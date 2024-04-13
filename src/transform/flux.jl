"""
    FromFluxAdaptor(preserve_ps_st::Bool=false, force_preserve::Bool=false)

Convert a Flux Model to Lux Model.

!!! warning

    This always ignores the `active` field of some of the Flux layers. This is almost never
    going to be supported.

## Keyword Arguments

  - `preserve_ps_st`: Set to `true` to preserve the states and parameters of the layer. This
    attempts the best possible way to preserve the original model. But it might fail. If you
    need to override possible failures, set `force_preserve` to `true`.

  - `force_preserve`: Some of the transformations with state and parameters preservation
    haven't been implemented yet, in these cases, if `force_transform` is `false` a warning
    will be printed and a core Lux layer will be returned. Else, it will create a
    [`FluxLayer`](@ref).

## Example

```jldoctest
julia> import Flux, Metalhead

julia> using Adapt, Lux, Random

julia> m = Metalhead.ResNet(18);

julia> m2 = adapt(FromFluxAdaptor(), m.layers); # or FromFluxAdaptor()(m.layers)

julia> x = randn(Float32, 224, 224, 3, 1);

julia> ps, st = Lux.setup(Random.default_rng(), m2);

julia> size(first(m2(x, ps, st))) == (1000, 1)
true
```
"""
@kwdef struct FromFluxAdaptor <: AbstractToLuxAdaptor
    preserve_ps_st::Bool = false
    force_preserve::Bool = false
end

"""
    Adapt.adapt(from::FromFluxAdaptor, L)

Adapt a Flux model `L` to Lux model. See [`FromFluxAdaptor`](@ref) for more details.
"""
function Adapt.adapt(from::FromFluxAdaptor, L)
    if Base.get_extension(@__MODULE__, :LuxFluxExt) === nothing
        error("`FromFluxAdaptor` requires Flux.jl to be loaded.")
    end
    return __from_flux_adaptor(L; from.preserve_ps_st, from.force_preserve)
end

function __from_flux_adaptor end

Base.@deprecate transform(l; preserve_ps_st::Bool=false, force_preserve::Bool=false) adapt(
    FromFluxAdaptor(preserve_ps_st, force_preserve), l)

# Extend for AMDGPU in extensions
@inline _maybe_flip_conv_weight(x) = copy(x)

struct FluxModelConversionError <: Exception
    msg::String
end

function Base.showerror(io::IO, e::FluxModelConversionError)
    print(io, "FluxModelConversionError(", e.msg, ")")
    return
end
