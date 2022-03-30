istraining() = false

function get_stats(::Val{true}, ::Val{false}, μ, σ², x::AbstractArray{T}, reduce_dims, momentum::T) where {T}
    # testmode with tracked stats
    N = ndims(x)
    stats_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
    return reshape(μ, stats_shape), reshape(σ², stats_shape)
end

function get_stats(::Val{false}, active, μ, σ², x, reduce_dims, momentum::T) where {T}
    # trainmode or testmode without tracked stats
    μ = mean(x; dims=reduce_dims)
    return μ, sum(abs2, x .- μ; dims=reduce_dims) ./ prod(map(i -> size(x, i), unique(reduce_dims)))
end

function get_stats(::Val{true}, active::Val{true}, μ, σ², x::AbstractArray{T}, reduce_dims, momentum::T) where {T}
    # trainmode with tracked stats
    _μ, _σ² = get_stats(Val(false), active, μ, σ², x, reduce_dims, momentum)
    _update_stats!(x, reduce_dims, ndims(x), _μ, _σ², momentum, μ, σ²)
    return _μ, _σ²
end

function _update_stats!(x, reduce_dims, N, _μ, _σ², momentum, μ, σ²)
    m = prod(size(x)[reduce_dims])  # needed for computing corrected var
    μnew = vec(N ∈ reduce_dims ? _μ : mean(_μ; dims=N))
    σ²new = vec(N ∈ reduce_dims ? _σ² : mean(_σ²; dims=N))
    @. μ = (1 - momentum) * μ + momentum * μnew
    @. σ² = (1 - momentum) * σ² + momentum * (m / (m - one(eltype(σ²)))) * σ²new
end

function norm_forward(
    l::AbstractExplicitLayer, ps::NamedTuple, states::NamedTuple, x::AbstractArray{T,N}, reduce_dims, affine_shape
) where {T,N}
    μ, σ² = get_stats(
        Val(l.track_stats),
        Val(states.training == :auto ? istraining() : states.training),
        states.μ,
        states.σ²,
        x,
        reduce_dims,
        l.momentum,
    )
    if l.affine
        γ = reshape(ps.γ, affine_shape)
        β = reshape(ps.β, affine_shape)
        return @. l.λ(γ * (x - μ) / sqrt(σ² + l.ϵ) + β)
    else
        return @. l.λ((x - μ) / sqrt(σ² + l.ϵ))
    end
end

struct BatchNorm{F1,F2,F3,N} <: AbstractExplicitLayer
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
initialstates(::AbstractRNG, l::BatchNorm) = (μ=zeros32(l.chs), σ²=ones32(l.chs), training=:auto)

parameterlength(l::BatchNorm) = l.affine ? (l.chs * 2) : 0
statelength(l::BatchNorm) = 2 * l.chs + 1

function Base.show(io::IO, l::BatchNorm)
    print(io, "BatchNorm($(l.chs)")
    (l.λ == identity) || print(io, ", $(l.λ)")
    l.affine || print(io, ", affine=false")
    return print(io, ")")
end

function batchnorm_fallback(BN::BatchNorm, x::AbstractArray{T}, ps::NamedTuple, states::NamedTuple) where {T}
    N = ndims(x)
    @assert size(x, ndims(x) - 1) == BN.chs
    @assert !(states.training == :auto ? istraining() : states.training) || size(x, ndims(x)) > 1 "During `training`, `BatchNorm` can't handle Batch Size == 1"
    reduce_dims = [1:(N - 2); N]
    affine_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
    return norm_forward(BN, ps, states, x, reduce_dims, affine_shape)
end

function (BN::BatchNorm)(x::AbstractVector, ps::NamedTuple, states::NamedTuple)
    y, states = BN(reshape(x, :, 1), ps, states)
    return vec(y), states
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
            batchnorm(
                ps.γ,
                ps.β,
                x,
                states.μ,
                states.σ²,
                BN.momentum;
                alpha=true,
                beta=false,
                eps=BN.ϵ,
                training=(states.training == :auto ? istraining() : states.training),
            ),
        ),
        states,
    )
end

struct WeightNorm{which_params,L<:AbstractExplicitLayer,D} <: AbstractExplicitLayer
    layer::L
    dims::D
end

function WeightNorm(
    layer::AbstractExplicitLayer, which_params::NTuple{N,Symbol}, dims::Union{Tuple,Nothing}=nothing
) where {N}
    return WeightNorm{Val{which_params},typeof(layer),typeof(dims)}(layer, dims)
end

function initialparameters(rng::AbstractRNG, wn::WeightNorm{Val{which_params}}) where {which_params}
    ps_layer = initialparameters(rng, wn.layer)
    ps_normalized = []
    ps_unnormalized = []
    i = 1
    for (k, v) in pairs(ps_layer)
        if k ∈ which_params
            dim = wn.dims === nothing ? ndims(v) : wn.dims[i]
            push!(ps_normalized, Symbol(string(k) * "_g") => _norm_except(v, dim))
            push!(ps_normalized, Symbol(string(k) * "_v") => v)
            i += 1
        else
            push!(ps_unnormalized, k => v)
        end
    end
    return (normalized=(; ps_normalized...), unnormalized=(; ps_unnormalized...))
end

initialstates(rng::AbstractRNG, wn::WeightNorm) = initialstates(rng, wn.layer)

function (wn::WeightNorm)(x, ps::NamedTuple, s::NamedTuple)
    _ps = get_normalized_parameters(wn, wn.dims, ps.normalized)
    return wn.layer(x, (; _ps..., ps.unnormalized...), s)
end

function get_normalized_parameters(::WeightNorm{Val{which_params}}, ::Nothing, ps::NamedTuple) where {which_params}
    wp_s = string.(which_params)
    _vs = getfield.((ps,), Symbol.(wp_s .* "_v"))
    _gs = getfield.((ps,), Symbol.(wp_s .* "_g"))
    return NamedTuple{which_params}(map((v, g) -> (v .* (g ./ _norm_except(v))), _vs, _gs))
end

function get_normalized_parameters(::WeightNorm{Val{which_params}}, dims::Tuple, ps::NamedTuple) where {which_params}
    wp_s = string.(which_params)
    _vs = getfield.((ps,), Symbol.(wp_s .* "_v"))
    _gs = getfield.((ps,), Symbol.(wp_s .* "_g"))
    return NamedTuple{which_params}(map((v, g, dim) -> (v .* (g ./ _norm_except(v, dim))), _vs, _gs, dims))
end

@inline _norm(x; dims=Colon()) = sqrt.(sum(abs2, x; dims=dims))

# Compute norm over all dimensions except `except_dim`
@inline _norm_except(x::AbstractArray{T,N}, except_dim) where {T,N} = _norm(x; dims=filter(i -> i != except_dim, 1:N))
@inline _norm_except(x::AbstractArray{T,N}) where {T,N} = _norm_except(x, N)
