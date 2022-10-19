# TODO(@avik-pal): Deprecation warnings once Flux2Lux.jl is registered.

import .Flux

"""
    transform(model)

Convert a Flux Model to Lux Model.

# Examples

```julia
using Lux, Metalhead, Random

m = ResNet(18)
m2 = Lux.transform(m.layers)

x = randn(Float32, 224, 224, 3, 1);

ps, st = Lux.setup(Random.default_rng(), m2);

m2(x, ps, st)
```
"""
transform(::T) where {T} = error("Transformation for type $T not implemented")

transform(model::Flux.Chain) = Chain(transform.(model.layers)...)

function transform(model::Flux.BatchNorm)
    return BatchNorm(model.chs, model.λ; affine=model.affine, track_stats=model.track_stats,
                     epsilon=model.ϵ, momentum=model.momentum)
end

function transform(model::Flux.Conv)
    return Conv(size(model.weight)[1:(end - 2)],
                size(model.weight, ndims(model.weight) - 1) * model.groups => size(model.weight,
                                                                                   ndims(model.weight)),
                model.σ; stride=model.stride, pad=model.pad,
                bias=model.bias isa Bool ? model.bias : !(model.bias isa Flux.Zeros),
                dilation=model.dilation, groups=model.groups)
end

function transform(model::Flux.SkipConnection)
    return SkipConnection(transform(model.layers), model.connection)
end

function transform(model::Flux.Dense)
    return Dense(size(model.weight, 2), size(model.weight, 1), model.σ)
end

function transform(model::Flux.MaxPool)
    return MaxPool(model.k, model.pad, model.stride)
end

function transform(model::Flux.MeanPool)
    return MeanPool(model.k, model.pad, model.stride)
end

function transform(::Flux.GlobalMaxPool)
    return GlobalMaxPool()
end

function transform(::Flux.GlobalMeanPool)
    return GlobalMeanPool()
end

function transform(p::Flux.AdaptiveMaxPool)
    return AdaptiveMaxPool(p.out)
end

function transform(p::Flux.AdaptiveMeanPool)
    return AdaptiveMeanPool(p.out)
end

function transform(model::Flux.Parallel)
    return Parallel(model.connection, transform.(model.layers)...)
end

function transform(d::Flux.Dropout)
    return Dropout(Float32(d.p); dims=d.dims)
end

transform(::typeof(identity)) = NoOpLayer()

transform(::typeof(Flux.flatten)) = FlattenLayer()

transform(f::Function) = WrappedFunction(f)
