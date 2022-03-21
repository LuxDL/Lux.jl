transform(x::Any) = x

transform(model::Flux.Chain) = Chain(transform.(model.layers)...)

function transform(model::Flux.BatchNorm)
    return BatchNorm(
        model.chs, model.λ; affine=model.affine, track_stats=model.track_stats, ϵ=model.ϵ, momentum=model.momentum
    )
end

function transform(model::Flux.Conv)
    return Conv(
        size(model.weight)[1:(end - 2)],
        size(model.weight, ndims(model.weight) - 1) => size(model.weight, ndims(model.weight)),
        model.σ;
        stride=model.stride,
        pad=model.pad,
        bias=!(model.bias isa Flux.Zeros),
        dilation=model.dilation,
        groups=model.groups,
    )
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

function transform(model::Flux.Parallel)
    return Parallel(model.connection, transform.(model.layers)...)
end

transform(::typeof(identity)) = NoOpLayer()

transform(f::Function) = WrappedFunction(f)
