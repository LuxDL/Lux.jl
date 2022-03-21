struct Dense{bias,F1,F2,F3} <: AbstractExplicitLayer
    λ::F1
    in_dims::Int
    out_dims::Int
    initW::F2
    initb::F3
end

function Base.show(io::IO, d::Dense)
    print(io, "Dense($(d.in_dims) => $(d.out_dims)")
    (d.λ == identity) || print(io, ", $(d.λ)")
    return print(io, ")")
end

function Dense(mapping::Pair{<:Int,<:Int}, λ=identity; initW=glorot_uniform, initb=zeros32, bias::Bool=true)
    return Dense(first(mapping), last(mapping), λ, initW=initW, initb=initb, bias=bias)
end

function Dense(in_dims::Int, out_dims::Int, λ=identity; initW=glorot_uniform, initb=zeros32, bias::Bool=true)
    return Dense{bias,typeof(λ),typeof(initW),typeof(initb)}(λ, in_dims, out_dims, initW, initb)
end

function initialparameters(rng::AbstractRNG, d::Dense{true})
    return (weight=d.initW(rng, d.out_dims, d.in_dims), bias=d.initb(rng, d.out_dims, 1))
end
initialparameters(rng::AbstractRNG, d::Dense{false}) = (weight=d.initW(rng, d.out_dims, d.in_dims),)

parameterlength(d::Dense{true}) = d.out_dims * (d.in_dims + 1)
parameterlength(d::Dense{false}) = d.out_dims * d.in_dims
statelength(d::Dense) = 0

function (d::Dense)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    y, st = d(reshape(x, size(x, 1), :), ps, st)
    return reshape(y, :, size(x)[2:end]...), st
end

function (d::Dense{false})(x::AbstractVecOrMat, ps::NamedTuple, st::NamedTuple)
    return (NNlib.fast_act(d.λ, x)).(ps.weight * x), st
end

function (d::Dense{true})(x::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    return (NNlib.fast_act(d.λ, x)).(ps.weight * x .+ ps.bias), st
end

function (d::Dense{true})(x::AbstractVector, ps::NamedTuple, st::NamedTuple)
    return (NNlib.fast_act(d.λ, x)).(ps.weight * x .+ vec(ps.bias)), st
end
