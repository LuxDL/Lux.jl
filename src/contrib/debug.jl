"""
    DebugLayer(layer::AbstractExplicitLayer; nan_check::Symbol=:both,
        error_check::Bool=true, location::String="")

!!! danger

    This layer is only meant to be used for debugging. If used for actual training or
    inference, will lead to extremely bad performance.

A wrapper over Lux layers that adds checks for NaNs and errors. This is useful for
debugging.

## Arguments

  - `layer`: The layer to be wrapped.

## Keyword Arguments

  - `nan_check`: Whether to check for NaNs in the input, parameters, and states. Can be
    `:both`, `:forward`, `:backward`, or `:none`.
  - `error_check`: Whether to check for errors in the layer. If `true`, will throw an error
    if the layer fails.
  - `location`: The location of the layer. Use [`Lux.Experimental.@debug_mode`](@ref) to
    construct this layer to populate this value correctly.

## Inputs

  - `x`: The input to the layer.

## Outputs

  - `y`: The output of the layer.
  - `st`: The updated states of the layer.

If `nan_check` is enabled and NaNs are detected then a `DomainError` is thrown. If
`error_check` is enabled, then any errors in the layer are thrown with useful information to
track where the error originates.

!!! warning

    `nan_check` for the backward mode only works with ChainRules Compatible Reverse Mode AD
    Tools currently.

See [`Lux.Experimental.@debug_mode`](@ref) to construct this layer.
"""
@concrete struct DebugLayer{NaNCheck, ErrorCheck} <:
                 AbstractExplicitContainerLayer{(:layer,)}
    layer
    location::String
end

function DebugLayer(layer::AbstractExplicitLayer; nan_check::Symbol=:both,
        error_check::Bool=true, location::String="")
    @assert nan_check ∈ (:both, :forward, :backward, :none)
    return DebugLayer{nan_check, error_check}(layer, location)
end

nan_check(::DebugLayer{NaNCheck}) where {NaNCheck} = Val(NaNCheck)
function error_check(::DebugLayer{NaNCheck, ErrorCheck}) where {NaNCheck, ErrorCheck}
    return Val(ErrorCheck)
end

function (d::DebugLayer)(x, ps, st)
    return __debug_layer(nan_check(d), error_check(d), d.layer, x, ps, st, d.location)
end

function __any_nan(x)
    has_nan = Ref(false)
    function nan_check(x)
        x isa AbstractArray && (has_nan[] = has_nan[] || any(isnan, x))
        applicable(isnan, x) && (has_nan[] = has_nan[] || isnan(x))
        return x
    end
    fmap(nan_check, x)
    return has_nan[]
end

CRC.@non_differentiable __any_nan(::Any)

function __debug_layer(
        ::Val{NC}, ::Val{EC}, layer, x, ps, st, location::String) where {NC, EC}
    CRC.ignore_derivatives() do
        @info "Input Type: $(typeof(x)) | Input Structure: $(fmap(__size, x))"
        @info "Running Layer: $(layer) at location $(location)!"
    end
    if NC ∈ (:both, :forward)
        __any_nan(x) && throw(DomainError(
            x, "NaNs detected in input to layer $(layer) at location $(location)"))
        __any_nan(ps) && throw(DomainError(ps,
            "NaNs detected in parameters of layer $(layer) at location $(location)"))
        __any_nan(st) && throw(DomainError(
            st, "NaNs detected in states of layer $(layer) at location $(location)"))
    end
    y, st_ = __debug_layer_internal(layer, x, ps, st, location, EC, NC ∈ (:both, :backward))
    CRC.ignore_derivatives() do
        @info "Output Type: $(typeof(y)) | Output Structure: $(fmap(__size, y))"
    end
    return y, st_
end

__size(x::AbstractArray) = size(x)
@generated __size(x::T) where {T} = hasmethod(size, Tuple{T}) ? :(size(x)) : :(nothing)

function __debug_layer_internal(layer, x, ps, st, location, EC, NC)
    if EC
        try
            y, st_ = apply(layer, x, ps, st)
            return y, st_
        catch e
            @error "Layer $(layer) failed!! This layer is present at location $(location)"
            rethrow()
        end
    else
        return apply(layer, x, ps, st)
    end
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(__debug_layer_internal), layer, x, ps, st, location, EC, NC)
    result, ∇__debug_layer_internal = CRC.rrule_via_ad(cfg, apply, layer, x, ps, st)
    function ∇__debug_layer_internal_with_checks(Δ)
        if NC
            __any_nan(Δ) && throw(DomainError(Δ,
                "NaNs detected in pullback input for $(layer) at location $(location)!"))
        end
        if EC
            try
                gs = ∇__debug_layer_internal(Δ)
            catch e
                @error "Backward Pass for Layer $(layer) failed!! This layer is present at location $(location)"
                rethrow()
            end
            if NC
                for g in gs
                    __any_nan(g) && throw(DomainError(g,
                        "NaNs detected in pullback output for $(layer) at location $(location)!"))
                end
            end
            return (gs..., CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent())
        else
            gs = ∇__debug_layer_internal(Δ)
            if NC
                for g in gs
                    __any_nan(g) && throw(DomainError(g,
                        "NaNs detected in pullback output for $(layer) at location $(location)!"))
                end
            end
            return (gs..., CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent())
        end
    end
    return result, ∇__debug_layer_internal_with_checks
end

"""
    @debug_mode layer kwargs...

Recurses into the `layer` and replaces the inner most non Container Layers with a
[`Lux.Experimental.DebugLayer`](@ref).

See [`Lux.Experimental.DebugLayer`](@ref) for details about the Keyword Arguments.
"""
macro debug_mode(layer, kwargs...)
    kws = esc.(kwargs)
    return :(__debug_mode($(esc(layer)), $(string(layer)); $(kws...)))
end

function __debug_mode(layer, name::String; kwargs...)
    l_c, l_re = functor(layer)

    length(l_c) == 0 && return DebugLayer(layer; location=name, kwargs...)

    l_c_new = []
    for k in keys(l_c)
        l_c_new_ = __debug_mode(getproperty(l_c, k), join((name, k), "."); kwargs...)
        push!(l_c_new, k => l_c_new_)
    end

    return l_re((; l_c_new...))
end
