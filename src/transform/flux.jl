"""
    FromFluxAdaptor(preserve_ps_st::Bool=false, force_preserve::Bool=false)

Convert a Flux Model to Lux Model.

!!! warning "`active` field"

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

```julia
julia> import Flux

julia> using Adapt, Lux, Random

julia> m = Flux.Chain(Flux.Dense(2 => 3, relu), Flux.Dense(3 => 2));

julia> m2 = adapt(FromFluxAdaptor(), m); # or FromFluxAdaptor()(m.layers)

julia> x = randn(Float32, 2, 32);

julia> ps, st = Lux.setup(Random.default_rng(), m2);

julia> size(first(m2(x, ps, st)))
(2, 32)
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
    if !is_extension_loaded(Val(:Flux))
        error("`FromFluxAdaptor` requires `Flux.jl` to be loaded.")
    end
    return convert_flux_model(L; from.preserve_ps_st, from.force_preserve)
end

function convert_flux_model end

Base.@deprecate transform(l; preserve_ps_st::Bool=false, force_preserve::Bool=false) adapt(
    FromFluxAdaptor(preserve_ps_st, force_preserve), l
)

function maybe_flip_conv_weight(x::AbstractArray)
    return maybe_flip_conv_weight(x, MLDataDevices.get_device_type(x))
end
maybe_flip_conv_weight(x::AbstractArray, _) = copy(x)
function maybe_flip_conv_weight(x::AbstractArray, ::Type{<:AMDGPUDevice})
    # This is a very rare operation, hence we dont mind allowing scalar operations
    return @allowscalar reverse(x; dims=ntuple(identity, ndims(x) - 2))
end
