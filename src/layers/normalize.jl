abstract type AbstractNormalizationLayer{affine,track_stats} <: AbstractExplicitLayer end

function get_stats!(
    ::Val{track_stats}, ::Val{active}, μ, σ², x::AbstractArray{T,N}, reduce_dims, momentum::T
) where {track_stats,active,T,N}
    if track_stats
        if active
            # Training
            μ_batch = mean(x; dims=reduce_dims)
            σ²_batch = std(x; mean=μ_batch, dims=reduce_dims)
            _update_stats!(reduce_dims, N, μ_batch, σ²_batch, momentum, μ, σ²)
            return μ_batch, σ²_batch
        else
            # Testing
            stats_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
            return reshape(μ, stats_shape), reshape(σ², stats_shape)
        end
    else
        # No Statistics Tracking
        μ = mean(x; dims=reduce_dims)
        return μ, std(x; mean=μ, dims=reduce_dims)
    end
end

function _update_stats!(reduce_dims, N, _μ, _σ², momentum, μ, σ²)
    μnew = vec(N == reduce_dims[end] ? _μ : mean(_μ; dims=N))
    σ²new = vec(N == reduce_dims[end] ? _σ² : mean(_σ²; dims=N))
    @. μ = (1 - momentum) * μ + momentum * μnew
    @. σ² = (1 - momentum) * σ² + momentum * σ²new
    return nothing
end

function norm_forward(
    l::AbstractNormalizationLayer{affine,track_stats},
    ps::NamedTuple,
    states::NamedTuple,
    x::AbstractArray{T,N},
    reduce_dims,
    affine_shape,
) where {T,N,affine,track_stats}
    μ, σ² = get_stats!(
        Val(track_stats),
        Val(states.training == :auto ? istraining() : states.training),
        states.μ,
        states.σ²,
        x,
        reduce_dims,
        l.momentum,
    )
    if affine
        γ = reshape(ps.γ, affine_shape)
        β = reshape(ps.β, affine_shape)
        return @. l.λ(γ * (x - μ) / sqrt(σ² + l.ϵ) + β)
    else
        return @. l.λ((x - μ) / sqrt(σ² + l.ϵ))
    end
end

function get_stats!(
    ::Val{track_stats},
    ::Val{active},
    μ::AbstractArray{T,1},
    σ²::AbstractArray{T,1},
    x::AbstractArray{T,N},
    momentum::T,
    μ_batch::AbstractArray{T,N},
    σ²_batch::AbstractArray{T,N},
    σ²_batch_intermediate::AbstractArray{T,N},
    l::AbstractNormalizationLayer,
) where {track_stats,active,T,N}
    if track_stats
        if active
            # Training
            mean!(μ_batch, x)
            var!(σ²_batch, σ²_batch_intermediate, x, μ_batch)
            _update_stats!!(μ_batch, σ²_batch, momentum, μ, σ², l)
        else
            # Testing
            selectdim(μ_batch, N - 1, :) .= μ
            selectdim(σ²_batch, N - 1, :) .= σ²
        end
    else
        # No Statistics Tracking
        mean!(μ_batch, x)
        var!(σ²_batch, σ²_batch_intermediate, x, μ_batch)
    end
    return μ_batch, σ²_batch
end

function _update_stats!!(μnew, σ²new, momentum, μ, σ², ::BatchNorm)
    @. μ = (1 - momentum) * μ + momentum * μnew
    @. σ² = (1 - momentum) * σ² + momentum * σ²new
    return nothing
end

function norm_forward!(
    y::AbstractArray{T,N},
    μ::AbstractArray{T,N},
    σ²::AbstractArray{T,N},
    σ²_intermediate::AbstractArray{T,N},
    γ::Union{AbstractArray{T,N},Nothing},
    β::Union{AbstractArray{T,N},Nothing},
    l::AbstractNormalizationLayer{affine,track_stats},
    ps::NamedTuple,
    states::NamedTuple,
    x::AbstractArray{T,N},
) where {T,N,affine,track_stats}
    get_stats!(
        Val(track_stats),
        Val(states.training == :auto ? istraining() : states.training),
        states.μ,
        states.σ²,
        x,
        l.momentum,
        μ,
        σ²,
        σ²_intermediate,
        l,
    )
    if affine
        selectdim(γ, N - 1, :) .= ps.γ
        selectdim(β, N - 1, :) .= ps.β
        @. y = l.λ(γ * (x - μ) / sqrt(σ² + l.ϵ) + β)
    else
        @. y = l.λ((x - μ) / sqrt(σ² + l.ϵ))
    end
    return y
end


# BatchNorm
struct BatchNorm{affine,track_stats,F1,F2,F3,N} <: AbstractNormalizationLayer{affine,track_stats}
    λ::F1
    ϵ::N
    momentum::N
    chs::Int
    initβ::F2
    initγ::F3
end

function BatchNorm(
    chs::Int,
    λ=identity;
    initβ=zeros32,
    initγ=ones32,
    affine::Bool=true,
    track_stats::Bool=true,
    ϵ=1.0f-5,
    momentum=0.1f0,
)
    λ = NNlib.fast_act(λ)
    return BatchNorm{affine,track_stats,typeof(λ),typeof(initβ),typeof(initγ),typeof(ϵ)}(
        λ, ϵ, momentum, chs, initβ, initγ
    )
end

function initialparameters(rng::AbstractRNG, l::BatchNorm{affine}) where {affine}
    return affine ? (γ=l.initγ(rng, l.chs), β=l.initβ(rng, l.chs)) : NamedTuple()
end
function initialstates(::AbstractRNG, l::BatchNorm{affine,track_stats}) where {affine,track_stats}
    return track_stats ? (μ=zeros32(l.chs), σ²=ones32(l.chs), training=:auto) : (training=:auto,)
end

function createcache(
    ::AbstractRNG, l::BatchNorm{affine}, x::AbstractArray{T,N}, ps::NamedTuple, states::NamedTuple
) where {T,N,affine}
    @assert size(x, N - 1) == l.chs
    return (
        normalized_input=similar(x),
        μ_cache=similar(x, ntuple(i -> i == N - 1 ? l.chs : 1, N)),
        σ²_cache=similar(x, ntuple(i -> i == N - 1 ? l.chs : 1, N)),
        σ²_intermediate=similar(x),
        γ_cache=affine ? similar(x, ntuple(i -> i == N - 1 ? l.chs : 1, N)) : nothing,
        β_cache=affine ? similar(x, ntuple(i -> i == N - 1 ? l.chs : 1, N)) : nothing,
    )
end

parameterlength(l::BatchNorm{affine}) where {affine} = affine ? (l.chs * 2) : 0
statelength(l::BatchNorm{affine,track_stats}) where {affine,track_stats} = (track_stats ? 2 * l.chs : 0) + 1
outputdescriptor(::BatchNorm, input::AbstractArray, ::NamedTuple, ::NamedTuple) = zero(input)

function Base.show(io::IO, l::BatchNorm{affine,track_stats}) where {affine,track_stats}
    print(io, "BatchNorm($(l.chs)")
    (l.λ == identity) || print(io, ", $(l.λ)")
    affine || print(io, ", affine=false")
    track_stats || print(io, ", track_stats=false")
    return print(io, ")")
end

function batchnorm_fallback(BN::BatchNorm, x::AbstractArray{T,N}, ps::NamedTuple, states::NamedTuple) where {T,N}
    @assert size(x, N - 1) == BN.chs
    @assert !(states.training == :auto ? istraining() : states.training) || size(x, N) > 1 "During `training`, `BatchNorm` can't handle Batch Size == 1"
    reduce_dims = [1:(N - 2); N]
    affine_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
    return norm_forward(BN, ps, states, x, reduce_dims, affine_shape)
end

function batchnorm_fallback!(
    BN::BatchNorm, x::AbstractArray{T,N}, ps::NamedTuple, states::NamedTuple, cache::NamedTuple
) where {T,N}
    @assert !(states.training == :auto ? istraining() : states.training) || size(x, N) > 1 "During `training`, `BatchNorm` can't handle Batch Size == 1"
    return norm_forward!(
        cache.normalized_input,
        cache.μ_cache,
        cache.σ²_cache,
        cache.σ²_intermediate,
        cache.γ_cache,
        cache.β_cache,
        BN,
        ps,
        states,
        x,
    )
end

function (BN::BatchNorm)(x::AbstractVector, ps::NamedTuple, states::NamedTuple)
    y, states = BN(reshape(x, :, 1), ps, states)
    return vec(y), states
end

function (BN::BatchNorm)(x::AbstractArray{T}, ps::NamedTuple, states::NamedTuple) where {T}
    return batchnorm_fallback(BN, x, ps, states), states
end

function (BN::BatchNorm)(x::AbstractVector, ps::NamedTuple, states::NamedTuple, cache::NamedTuple)
    y, states = BN(reshape(x, :, 1), ps, states, cache)
    return vec(y), states
end

function (BN::BatchNorm)(x::AbstractArray, ps::NamedTuple, states::NamedTuple, cache::NamedTuple)
    return batchnorm_fallback!(BN, x, ps, states, cache), states
end

function (BN::BatchNorm{affine,track_stats})(
    x::Union{CuArray{T,2},CuArray{T,4},CuArray{T,5}}, ps::NamedTuple, states::NamedTuple
) where {T<:Union{Float32,Float64},affine,track_stats}
    # TODO: Update to the latest CUDNN API
    (!affine || !track_stats) && return (batchnorm_fallback(BN, x, ps, states), states)
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

function (BN::BatchNorm{affine,track_stats})(
    x::Union{CuArray{T,2},CuArray{T,4},CuArray{T,5}}, ps::NamedTuple, states::NamedTuple, cache::NamedTuple
) where {T<:Union{Float32,Float64},affine,track_stats}
    # TODO: Update to the latest CUDNN API
    (!affine || !track_stats) && return (batchnorm_fallback!(BN, x, ps, states, cache), states)
    # TODO: Use the inplace batchnorm CUDA API
    #       Need to manually handle array reshaping
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

Base.@pure function (wn::WeightNorm)(x, ps::NamedTuple, s::NamedTuple)
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
