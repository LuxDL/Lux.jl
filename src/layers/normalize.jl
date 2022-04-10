abstract type AbstractNormalizationLayer{affine,track_stats} <: AbstractExplicitLayer end

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
    return if track_stats
        (μ=zeros32(l.chs), σ²=ones32(l.chs), training=:auto)
    else
        (μ=nothing, σ²=nothing, training=:auto)
    end
end

function createcache(
    ::AbstractRNG, l::BatchNorm{affine}, x::AbstractArray{T,_N}, ps::NamedTuple, states::NamedTuple
) where {T,_N,affine}
    x = _N == 1 ? @view(x[:, :]) : x
    N = ndims(x)
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
    @assert !istraining(states) || size(x, N) > 1 "During `training`, `BatchNorm` can't handle Batch Size == 1"
    reduce_dims = [1:(N - 2); N]
    affine_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
    return norm_forward(BN, ps, states, x, reduce_dims, affine_shape)
end

function batchnorm_fallback!(
    BN::BatchNorm, x::AbstractArray{T,N}, ps::NamedTuple, states::NamedTuple, cache::NamedTuple
) where {T,N}
    @assert !istraining(states) || size(x, N) > 1 "During `training`, `BatchNorm` can't handle Batch Size == 1"
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
    y, states = BN(x[:, :], ps, states)
    return y[:], states
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
                training=istraining(states),
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
                training=istraining(states),
            ),
        ),
        states,
    )
end

# GroupNorm
struct GroupNorm{affine,track_stats,F1,F2,F3,N} <: AbstractNormalizationLayer{affine,track_stats}
    λ::F1
    ϵ::N
    momentum::N
    chs::Int
    initβ::F2
    initγ::F3
    groups::Int
end

function GroupNorm(
    chs::Int,
    groups::Int,
    λ=identity;
    initβ=zeros32,
    initγ=ones32,
    affine::Bool=true,
    track_stats::Bool=true,
    ϵ=1.0f-5,
    momentum=0.1f0,
)
    @assert chs % groups == 0 "The number of groups ($(groups)) must divide the number of channels ($chs)"
    λ = NNlib.fast_act(λ)
    return GroupNorm{affine,track_stats,typeof(λ),typeof(initβ),typeof(initγ),typeof(ϵ)}(
        λ, ϵ, momentum, chs, initβ, initγ, groups
    )
end

function initialparameters(rng::AbstractRNG, l::GroupNorm{affine}) where {affine}
    return affine ? (γ=l.initγ(rng, l.chs), β=l.initβ(rng, l.chs)) : NamedTuple()
end
function initialstates(::AbstractRNG, l::GroupNorm{affine,track_stats}) where {affine,track_stats}
    return if track_stats
        (μ=zeros32(l.groups), σ²=ones32(l.groups), training=:auto)
    else
        (μ=nothing, σ²=nothing, training=:auto)
    end
end

parameterlength(l::GroupNorm{affine}) where {affine} = affine ? (l.chs * 2) : 0
statelength(l::GroupNorm{affine,track_stats}) where {affine,track_stats} = (track_stats ? 2 * l.groups : 0) + 1
outputdescriptor(::GroupNorm, input::AbstractArray, ::NamedTuple, ::NamedTuple) = zero(input)

function Base.show(io::IO, l::GroupNorm{affine,track_stats}) where {affine,track_stats}
    print(io, "GroupNorm($(l.chs), $(l.groups)")
    (l.λ == identity) || print(io, ", $(l.λ)")
    affine || print(io, ", affine=false")
    track_stats || print(io, ", track_stats=false")
    return print(io, ")")
end

function (GN::GroupNorm)(x::AbstractArray{T,N}, ps::NamedTuple, st::NamedTuple) where {T,N}
    sz = size(x)
    @assert N > 2
    @assert sz[N - 1] == GN.chs

    x_ = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ GN.groups, GN.groups, sz[N])
    N_ = N + 1

    reduce_dims = 1:(N_ - 2)
    affine_shape = ntuple(i -> i ∈ (N_ - 1, N_ - 2) ? size(x_, i) : 1, N_)

    return reshape(norm_forward(GN, ps, st, x_, reduce_dims, affine_shape), sz), st
end

# WeightNorm
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
    return wn.layer(x, merge(_ps, ps.unnormalized), s)
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

function Base.show(io::IO, w::WeightNorm{Val{which_params}}) where {which_params}
    return print(io, "WeightNorm{", which_params, "}(", w.layer, ")")
end
