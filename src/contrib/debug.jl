"""
    DebugLayer(layer::AbstractExplicitLayer; nan_check::Symbol=:both,
        error_check::Bool=true, location::KeyPath=KeyPath())

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
    layer <: AbstractExplicitLayer
    location::KeyPath
end

function DebugLayer(layer::AbstractExplicitLayer; nan_check::Symbol=:both,
        error_check::Bool=true, location::Union{KeyPath, String}=KeyPath())
    @argcheck nan_check in (:both, :forward, :backward, :none)

    if location isa String
        Base.depwarn(
            "Using a String for location in DebugLayer is deprecated. Use \
            `Functors.KeyPath` instead.", :DebugLayer)
        location = KeyPath(Symbol.(split(location, "."))...)
    end

    return DebugLayer{nan_check, error_check}(layer, location)
end

function (d::DebugLayer{NaNCheck, ErrorCheck})(x, ps, st) where {NaNCheck, ErrorCheck}
    CRC.ignore_derivatives() do
        @info lazy"Input Type: $(typeof(x)) | Input Structure: $(fmapstructure(Lux.__size, x))"
        @info lazy"Running Layer: $(d.layer) at location $(d.location)!"
        if NaNCheck ∈ (:both, :forward)
            __check_nan_and_throw(x, "input", d.layer, d.location)
            __check_nan_and_throw(ps, "parameters", d.layer, d.location)
            __check_nan_and_throw(st, "states", d.layer, d.location)
        end
    end
    y, st_ = __debug_layer_internal(
        d.layer, x, ps, st, d.location, ErrorCheck, NaNCheck ∈ (:both, :backward))
    CRC.ignore_derivatives() do
        if NaNCheck ∈ (:both, :forward)
            __check_nan_and_throw(y, "output", d.layer, d.location)
            __check_nan_and_throw(st_, "states", d.layer, d.location)
        end
        @info lazy"Output Type: $(typeof(y)) | Output Structure: $(fmapstructure(Lux.__size, y))"
    end
    return y, st_
end

function __check_nan_and_throw(x, str::AbstractString, layer, location::KeyPath)
    function err(kp, x)
        loc_str = kp == KeyPath() ? " " : " (@ $(kp)) "
        return DomainError(
            x, "NaNs detected in $(str)$(loc_str)of layer $(layer) at location $(location)")
    end

    function nan_check(kp, x)
        x isa AbstractArray && any(isnan, x) && throw(err(kp, x))
        applicable(isnan, x) && isnan(x) && throw(err(kp, x))
        return x
    end

    return fmap_with_path(nan_check, x)
end

function __debug_layer_internal(layer, x, ps, st, location, EC, NC)
    y, st_ = try
        apply(layer, x, ps, st)
    catch
        EC && @error "Layer $(layer) failed!! This layer is present at location $(location)"
        rethrow()
    end
    return y, st_
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(__debug_layer_internal), layer, x, ps, st, location, EC, NC)
    result, ∇debug_layer_internal = CRC.rrule_via_ad(cfg, apply, layer, x, ps, st)
    syms = ["LuxCore.apply", "layer", "x", "ps", "st"]
    function ∇debug_layer_internal_with_checks(Δ)
        NC && __check_nan_and_throw(Δ, "pullback input", layer, location)

        gs = try
            ∇debug_layer_internal(Δ)
        catch
            EC &&
                @error "Backward Pass for Layer $(layer) failed!! This layer is present at location $(location)"
            rethrow()
        end

        if NC
            for (i, g) in enumerate(gs)
                __check_nan_and_throw(g, "pullback output ($(syms[i]))", layer, location)
            end
        end

        return (gs..., CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent())
    end
    return result, ∇debug_layer_internal_with_checks
end

"""
    @debug_mode layer kwargs...

Recurses into the `layer` and replaces the inner most non Container Layers with a
[`Lux.Experimental.DebugLayer`](@ref).

See [`Lux.Experimental.DebugLayer`](@ref) for details about the Keyword Arguments.
"""
macro debug_mode(layer, kwargs...)
    kws = esc.(kwargs)
    return :($(fmap_with_path)(
        (kp, l) -> DebugLayer(l; location=$(KeyPath)($(Meta.quot(layer)), kp), $(kws...)),
        $(esc(layer))))
end
