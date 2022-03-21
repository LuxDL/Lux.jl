struct BatchNorm{F1,F2,F3,N} <: ExplicitLayer
    λ::F1
    ϵ::N
    momentum::N
    chs::Int
    initβ::F2
    initγ::F3
    affine::Bool
    track_stats::Bool
end

function BatchNorm(
    chs::Int, λ=identity; initβ=zeros32, initγ=ones32, affine=true, track_stats=true, ϵ=1.0f-5, momentum=0.1f0
)
    return BatchNorm(λ, ϵ, momentum, chs, initβ, initγ, affine, track_stats)
end

function initialparameters(rng::AbstractRNG, l::BatchNorm)
    return l.affine ? (γ=l.initγ(rng, l.chs), β=l.initβ(rng, l.chs)) : NamedTuple()
end
initialstates(::AbstractRNG, l::BatchNorm) = (μ=zeros32(l.chs), σ²=ones32(l.chs), training=true)

parameterlength(l::BatchNorm) = l.affine ? (l.chs * 2) : 0
statelength(l::BatchNorm) = 2 * l.chs + 1

function Base.show(io::IO, l::BatchNorm)
    print(io, "BatchNorm($(l.chs)")
    (l.λ == identity) || print(io, ", $(l.λ)")
    l.affine || print(io, ", affine=false")
    return print(io, ")")
end

function batchnorm_fallback(BN::BatchNorm, x::AbstractArray{T,N}, ps::NamedTuple, states::NamedTuple) where {T,N}
    @assert size(x, ndims(x) - 1) == BN.chs
    @assert !states.training || size(x, ndims(x)) > 1 "During `training`, `BatchNorm` can't handle Batch Size == 1"
    reduce_dims = [1:(N - 2); N]
    affine_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
    return norm_forward(BN, ps, states, x, reduce_dims, affine_shape)
end

function (BN::BatchNorm)(x::AbstractVector, ps::NamedTuple, states::NamedTuple) where {T}
    return vec(batchnorm_fallback(BN, reshape(x, :, 1), ps, states)), states
end

function (BN::BatchNorm)(x::AbstractArray{T}, ps::NamedTuple, states::NamedTuple) where {T}
    return batchnorm_fallback(BN, x, ps, states), states
end

function (BN::BatchNorm)(
    x::Union{CuArray{T,2},CuArray{T,4},CuArray{T,5}}, ps::NamedTuple, states::NamedTuple
) where {T<:Union{Float32,Float64}}
    (!BN.affine || !BN.track_stats) && return (batchnorm_fallback(BN, x, ps, states), states)
    return (
        BN.λ.(
            batchnorm(ps.γ, ps.β, x, states.μ, states.σ², BN.momentum; alpha=1, beta=0, eps=BN.ϵ, training=states.training)
        ),
        states
    )
end
