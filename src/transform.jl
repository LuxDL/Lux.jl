import .Flux

"""
    transform(model)

Convert a Flux Model to Lux Model.

!!! tip
    
    It is recommended to use the package `Flux2Lux` instead of this function. It supports
    convertion of a wider variation of Flux models.

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
function transform(x)
    Base.depwarn("`Lux.transform` has been deprecated in favor of the package `Flux2Lux`." *
                 "This function will be removed in v0.5", :transform)
    return _transform(x)
end

_transform(::T) where {T} = error("Transformation for type $T not implemented")

_transform(model::Flux.Chain) = Chain(_transform.(model.layers)...)

function _transform(model::Flux.BatchNorm)
    return BatchNorm(model.chs, model.λ; affine=model.affine, track_stats=model.track_stats,
                     epsilon=model.ϵ, momentum=model.momentum)
end

function _transform(model::Flux.Conv)
    in_chs = size(model.weight, ndims(model.weight) - 1) * model.groups
    return Conv(size(model.weight)[1:(end - 2)],
                in_chs => size(model.weight, ndims(model.weight)), model.σ;
                stride=model.stride, pad=model.pad,
                bias=model.bias isa Bool ? model.bias : !(model.bias isa Flux.Zeros),
                dilation=model.dilation, groups=model.groups)
end

function _transform(model::Flux.SkipConnection)
    return SkipConnection(_transform(model.layers), model.connection)
end

_transform(model::Flux.Dense) = Dense(size(model.weight, 2), size(model.weight, 1), model.σ)

_transform(model::Flux.MaxPool) = MaxPool(model.k, model.pad, model.stride)

_transform(model::Flux.MeanPool) = MeanPool(model.k, model.pad, model.stride)

_transform(::Flux.GlobalMaxPool) = GlobalMaxPool()

_transform(::Flux.GlobalMeanPool) = GlobalMeanPool()

_transform(p::Flux.AdaptiveMaxPool) = AdaptiveMaxPool(p.out)

_transform(p::Flux.AdaptiveMeanPool) = AdaptiveMeanPool(p.out)

_transform(model::Flux.Parallel) = Parallel(model.connection, _transform.(model.layers)...)

_transform(d::Flux.Dropout) = Dropout(Float32(d.p); dims=d.dims)

_transform(::typeof(identity)) = NoOpLayer()

_transform(::typeof(Flux.flatten)) = FlattenLayer()

_transform(f::Function) = WrappedFunction(f)
