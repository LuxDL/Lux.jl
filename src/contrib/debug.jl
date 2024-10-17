"""
    DebugLayer(layer::AbstractLuxLayer;
        nan_check::Union{Symbol, StaticSymbol, Val}=static(:both),
        error_check::Union{StaticBool, Bool, Val{true}, Val{false}}=True(),
        location::KeyPath=KeyPath())

A wrapper over Lux layers that adds checks for NaNs and errors. This is useful for
debugging.

## Arguments

  - `layer`: The layer to be wrapped.

# Extended Help

## Keyword Arguments

  - `nan_check`: Whether to check for NaNs in the input, parameters, and states. Can be
    `:both`, `:forward`, `:backward`, or `:none`.
  - `error_check`: Whether to check for errors in the layer. If `true`, will throw an error
    if the layer fails.
  - `location`: The location of the layer. Use [`Lux.Experimental.@debug_mode`](@ref) to
    construct this layer to populate this value correctly.

## Input / Output

Inputs and outputs are the same as the `layer` unless one of the `nan_check` or
`error_check` criteria is met.

If `nan_check` is enabled and NaNs are detected then a `DomainError` is thrown. If
`error_check` is enabled, then any errors in the layer are thrown with useful information to
track where the error originates.

!!! warning "ChainRules Compatible Reverse Mode AD Tools"

    `nan_check` for the backward mode only works with ChainRules Compatible Reverse Mode AD
    Tools currently.

!!! danger "Disable After Debugging"

    This layer is only meant to be used for debugging. If used for actual training or
    inference, will lead to extremely bad performance.

See [`Lux.Experimental.@debug_mode`](@ref) to construct this layer.
"""
@concrete struct DebugLayer <: AbstractLuxWrapperLayer{:layer}
    nan_check <: StaticSymbol
    error_check <: StaticBool
    layer <: AbstractLuxLayer
    location::KeyPath
end

function DebugLayer(layer::AbstractLuxLayer; nan_check::SymbolType=static(:both),
        error_check::BoolType=True(), location::KeyPath=KeyPath())
    @argcheck dynamic(nan_check) in (:both, :forward, :backward, :none)
    return DebugLayer(static(nan_check), static(error_check), layer, location)
end

function (d::DebugLayer)(x, ps, st)
    CRC.ignore_derivatives() do
        @info lazy"Input Type: $(typeof(x)) | Input Structure: $(Utils.structure(x))."
        @info lazy"Running Layer: $(d.layer) at location $(d.location)!"
        if known(d.nan_check) ∈ (:both, :forward)
            check_nan_and_throw(x, "input", d.layer, d.location)
            check_nan_and_throw(ps, "parameters", d.layer, d.location)
            check_nan_and_throw(st, "states", d.layer, d.location)
        end
    end
    y,
    stₙ = debug_layer_impl(d.layer, x, ps, st, d.location, known(d.error_check),
        known(d.nan_check) ∈ (:both, :backward))
    CRC.ignore_derivatives() do
        if known(d.nan_check) ∈ (:both, :forward)
            check_nan_and_throw(y, "output", d.layer, d.location)
            check_nan_and_throw(stₙ, "states", d.layer, d.location)
        end
        @info lazy"Output Type: $(typeof(y)) | Output Structure: $(Utils.structure(y))."
    end
    return y, stₙ
end

function check_nan_and_throw(x, str::AbstractString, layer, location::KeyPath)
    function err(kp, x)
        loc_str = kp == KeyPath() ? " " : " (@ $(kp)) "
        return DomainError(x,
            "NaNs detected in $(str)$(loc_str) of layer $(layer) at location $(location).")
    end

    function nan_check(kp, x)
        x isa AbstractArray && any(isnan, x) && throw(err(kp, x))
        applicable(isnan, x) && isnan(x) && throw(err(kp, x))
        return x
    end

    return fmap_with_path(nan_check, x)
end

function debug_layer_impl(layer, x, ps, st, location, error_check, _)
    y,
    stₙ = try
        apply(layer, x, ps, st)
    catch
        error_check &&
            @error "Layer $(layer) failed!! This layer is present at location $(location)."
        rethrow()
    end
    return y, stₙ
end

function CRC.rrule(cfg::CRC.RuleConfig{>:CRC.HasReverseMode}, ::typeof(debug_layer_impl),
        layer, x, ps, st, location, error_check, nan_check_backward)
    result, ∇debug_layer_internal = CRC.rrule_via_ad(cfg, apply, layer, x, ps, st)
    syms = ("LuxCore.apply", "layer", "x", "ps", "st")
    function ∇debug_layer_internal_with_checks(Δ)
        nan_check_backward && check_nan_and_throw(Δ, "pullback input", layer, location)

        gs = try
            ∇debug_layer_internal(Δ)
        catch
            error_check &&
                @error "Backward Pass for Layer $(layer) failed!! This layer is present at location $(location)."
            rethrow()
        end

        if nan_check_backward
            foreach(enumerate(gs)) do (i, g)
                check_nan_and_throw(g, "pullback output ($(syms[i]))", layer, location)
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
