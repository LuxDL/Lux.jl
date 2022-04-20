abstract type AbstractNormalizationLayer{affine,track_stats} <: AbstractExplicitLayer end

get_reduce_dims(::AbstractNormalizationLayer, ::AbstractArray) = error("Not Implemented Yet!!")

get_proper_shape(::AbstractNormalizationLayer, ::AbstractArray, y::Nothing, args...) = y
get_proper_shape(::AbstractNormalizationLayer, x::AbstractArray{T,N}, y::AbstractArray{T,N}, args...) where {T,N} = y

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
    return affine ? ComponentArray((γ=l.initγ(rng, l.chs), β=l.initβ(rng, l.chs))) : ComponentArray{Float32}()
end
function initialstates(::AbstractRNG, l::BatchNorm{affine,track_stats}) where {affine,track_stats}
    return track_stats ? (μ=zeros32(l.chs), σ²=ones32(l.chs), training=:auto) : (training=:auto,)
end

parameterlength(l::BatchNorm{affine}) where {affine} = affine ? (l.chs * 2) : 0
statelength(l::BatchNorm{affine,track_stats}) where {affine,track_stats} = (track_stats ? 2 * l.chs : 0) + 1

get_reduce_dims(::BatchNorm, x::AbstractArray{T,N}) where {T,N} = [1:(N - 2); N]

function get_proper_shape(::BatchNorm, x::AbstractArray{T,N}, y::AbstractVector) where {T,N}
    return reshape(y, ntuple(i -> i == N - 1 ? length(y) : 1, N)...)
end

function (BN::BatchNorm)(x::AbstractVector, ps, states::NamedTuple)
    y, states = BN(x[:, :], ps, states)
    return y[:], states
end

Base.@pure function (BN::BatchNorm{affine,track_stats})(
    x::AbstractArray{T,N}, ps, st::NamedTuple
) where {T,N,affine,track_stats}
    @assert size(x, N - 1) == BN.chs
    @assert !istraining(st) || size(x, N) > 1 "During `training`, `BatchNorm` can't handle Batch Size == 1"

    x_normalized, xmean, xvar = normalization_forward(
        BN,
        x,
        get_proper_shape(BN, x, track_stats ? st.μ : nothing),
        get_proper_shape(BN, x, track_stats ? st.σ² : nothing),
        get_proper_shape(BN, x, affine ? ps.γ : nothing),
        get_proper_shape(BN, x, affine ? ps.β : nothing),
        BN.λ;
        training=istraining(st),
    )

    st_ = track_stats ? (μ=xmean, σ²=xvar, training=st.training) : st

    return (x_normalized, st_)
end

function (BN::BatchNorm{affine,track_stats})(
    x::Union{CuArray{T,2},CuArray{T,4},CuArray{T,5}}, ps, st::NamedTuple
) where {T<:Union{Float32,Float64},affine,track_stats}
    # NNlibCUDA silently updates μ and σ² so copying them
    if istraining(st)
        μ2 = track_stats ? copy(st.μ) : nothing
        σ²2 = track_stats ? copy(st.σ²) : nothing
    else
        if track_stats
            μ2 = copy(st.μ)
            σ²2 = copy(st.σ²)
        else
            μ2 = mean(x; dims=get_reduce_dims(BN, x))
            σ²2 = var(x; mean=μ2, dims=reduce_dims, corrected=false)
        end
    end
    res = 
        BN.λ.(
            batchnorm(
                affine ? ps.γ : nothing,
                affine ? ps.β : nothing,
                x,
                μ2,
                σ²2,
                BN.momentum;
                eps=BN.ϵ,
                training=istraining(st),
            ),
        )
    if track_stats
        @set! st.μ = μ2
        @set! st.σ² = σ²2
    end
    return res, st
end

function Base.show(io::IO, l::BatchNorm{affine,track_stats}) where {affine,track_stats}
    print(io, "BatchNorm($(l.chs)")
    (l.λ == identity) || print(io, ", $(l.λ)")
    affine || print(io, ", affine=false")
    track_stats || print(io, ", track_stats=false")
    return print(io, ")")
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
    return affine ? ComponentArray((γ=l.initγ(rng, l.chs), β=l.initβ(rng, l.chs))) : ComponentArray{Float32}()
end
function initialstates(::AbstractRNG, l::GroupNorm{affine,track_stats}) where {affine,track_stats}
    return track_stats ? (μ=zeros32(l.groups), σ²=ones32(l.groups), training=:auto) : (training=:auto,)
end

parameterlength(l::GroupNorm{affine}) where {affine} = affine ? (l.chs * 2) : 0
statelength(l::GroupNorm{affine,track_stats}) where {affine,track_stats} = (track_stats ? 2 * l.groups : 0) + 1

get_reduce_dims(::GroupNorm, ::AbstractArray{T,N}) where {T,N} = 1:(N - 1)

function get_proper_shape(::GroupNorm, x::AbstractArray{T,N}, y::AbstractVector, mode::Symbol) where {T,N}
    if mode == :state
        return reshape(y, ntuple(i -> i == N - 1 ? size(x, i) : 1, N)...)
    else
        return reshape(y, ntuple(i -> i ∈ (N - 2, N - 1) ? size(x, i) : 1, N)...)
    end
end

Base.@pure function (GN::GroupNorm{affine,track_stats})(
    x::AbstractArray{T,N}, ps, st::NamedTuple
) where {T,N,affine,track_stats}
    sz = size(x)
    @assert N > 2
    @assert sz[N - 1] == GN.chs

    x_ = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ GN.groups, GN.groups, sz[N])

    x_normalized, xmean, xvar = normalization_forward(
        GN,
        x_,
        get_proper_shape(GN, x, track_stats ? st.μ : nothing, :state),
        get_proper_shape(GN, x, track_stats ? st.σ² : nothing, :state),
        get_proper_shape(GN, x, affine ? ps.γ : nothing, :param),
        get_proper_shape(GN, x, affine ? ps.β : nothing, :param),
        GN.λ;
        training=istraining(st),
    )

    st_ = if track_stats
        (μ=xmean, σ²=xvar, training=st.training)
    else
        st
    end

    return reshape(x_normalized, sz), st_
end

function Base.show(io::IO, l::GroupNorm{affine,track_stats}) where {affine,track_stats}
    print(io, "GroupNorm($(l.chs), $(l.groups)")
    (l.λ == identity) || print(io, ", $(l.λ)")
    affine || print(io, ", affine=false")
    track_stats || print(io, ", track_stats=false")
    return print(io, ")")
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
    for k in propertynames(ps_layer)
        v = ps_layer[k]
        if k ∈ which_params
            dim = wn.dims === nothing ? ndims(v) : wn.dims[i]
            push!(ps_normalized, Symbol(string(k) * "_g") => _norm_except(v, dim))
            push!(ps_normalized, Symbol(string(k) * "_v") => v)
            i += 1
        else
            push!(ps_unnormalized, k => v)
        end
    end
    ps_unnormalized = length(ps_unnormalized) == 0 ? ComponentArray{Float32}() : ComponentArray(; ps_unnormalized...)
    return ComponentArray(normalized=ComponentArray(; ps_normalized...), unnormalized=ps_unnormalized)
end

initialstates(rng::AbstractRNG, wn::WeightNorm) = initialstates(rng, wn.layer)

Base.@pure function (wn::WeightNorm)(x, ps::Union{ComponentArray,NamedTuple}, s::NamedTuple)
    _ps = get_normalized_parameters(wn, wn.dims, ps.normalized)
    return wn.layer(x, merge(_ps, ps.unnormalized), s)
end

function get_normalized_parameters(
    ::WeightNorm{Val{which_params}}, ::Nothing, ps::Union{ComponentArray,NamedTuple}
) where {which_params}
    wp_s = string.(which_params)
    _vs = getfield.((ps,), Symbol.(wp_s .* "_v"))
    _gs = getfield.((ps,), Symbol.(wp_s .* "_g"))
    return NamedTuple{which_params}(map((v, g) -> (v .* (g ./ _norm_except(v))), _vs, _gs))
end

function get_normalized_parameters(
    ::WeightNorm{Val{which_params}}, dims::Tuple, ps::Union{ComponentArray,NamedTuple}
) where {which_params}
    wp_s = string.(which_params)
    _vs = getfield.((ps,), Symbol.(wp_s .* "_v"))
    _gs = getfield.((ps,), Symbol.(wp_s .* "_g"))
    return NamedTuple{which_params}(map((v, g, dim) -> (v .* (g ./ _norm_except(v, dim))), _vs, _gs, dims))
end

function Base.show(io::IO, w::WeightNorm{Val{which_params}}) where {which_params}
    return print(io, "WeightNorm{", which_params, "}(", w.layer, ")")
end
