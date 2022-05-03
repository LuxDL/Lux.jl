abstract type AbstractNormalizationLayer{affine,track_stats} <: AbstractExplicitLayer end

get_reduce_dims(::AbstractNormalizationLayer, ::AbstractArray) = error("Not Implemented Yet!!")

get_proper_shape(::AbstractNormalizationLayer, ::AbstractArray, y::Nothing, args...) = y
get_proper_shape(::AbstractNormalizationLayer, x::AbstractArray{T,N}, y::AbstractArray{T,N}, args...) where {T,N} = y

"""
    BatchNorm(chs::Integer, λ=identity; initβ=zeros32, initγ=ones32,
              affine = true, track_stats = true, ϵ=1f-5, momentum= 0.1f0)

[Batch Normalization](https://arxiv.org/abs/1502.03167) layer.

`BatchNorm` computes the mean and variance for each `D_1×...×D_{N-2}×1×D_N` input slice and normalises the input accordingly.

# Arguments

* `chs` should be the size of the channel dimension in your data (see below). Given an array with `N` dimensions, call the `N-1`th the channel dimension. For a batch of feature vectors this is just the data dimension, for `WHCN` images it's the usual channel dimension.
* After normalisation, elementwise activation `λ` is applied.
* If `affine=true`, it also applies  a shift and a rescale to the input through to learnable per-channel bias β and scale γ parameters.
* If `track_stats=true`, accumulates mean and var statistics in training phase that will be used to renormalize the input in test phase.

Use [`Lux.testmode`](@ref) during inference.

# Examples

```julia
m = Chain(
        Dense(784 => 64),
        BatchNorm(64, relu),
        Dense(64 => 10),
        BatchNorm(10)
    )
```
"""
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
function initialstates(rng::AbstractRNG, l::BatchNorm{affine,track_stats}) where {affine,track_stats}
    return track_stats ? (μ=zeros32(rng, l.chs), σ²=ones32(rng, l.chs), training=true) : (training=true,)
end

parameterlength(l::BatchNorm{affine}) where {affine} = affine ? (l.chs * 2) : 0
statelength(l::BatchNorm{affine,track_stats}) where {affine,track_stats} = (track_stats ? 2 * l.chs : 0) + 1

get_reduce_dims(::BatchNorm, x::AbstractArray{T,N}) where {T,N} = [1:(N - 2); N]

function get_proper_shape(::BatchNorm, x::AbstractArray{T,N}, y::AbstractVector) where {T,N}
    return reshape(y, ntuple(i -> i == N - 1 ? length(y) : 1, N)...)
end

function (BN::BatchNorm{affine,track_stats})(x::AbstractArray{T,N}, ps, st::NamedTuple) where {T,N,affine,track_stats}
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
            reduce_dims = get_reduce_dims(BN, x)
            μ2 = mean(x; dims=reduce_dims)
            σ²2 = var(x; mean=μ2, dims=reduce_dims, corrected=false)
        end
    end
    res = applyactivation(
        BN.λ,
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
        Val(!istraining(st))
    )
    if track_stats
        st = merge(st, (μ=μ2, σ²=σ²2))
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

"""
    GroupNorm(chs::Integer, groups::Integer, λ=identity; initβ=zeros32, initγ=ones32,
              affine=true, track_stats=false, ϵ=1f-5, momentum=0.1f0)

[Group Normalization](https://arxiv.org/abs/1803.08494) layer.

# Arguments

* `chs` is the number of channels, the channel dimension of your input. For an array of N dimensions, the `N-1`th index is the channel dimension.
* `G` is the number of groups along which the statistics are computed. The number of channels must be an integer multiple of the number of groups.
* After normalisation, elementwise activation `λ` is applied.
* If `affine=true`, it also applies  a shift and a rescale to the input through to learnable per-channel bias `β` and scale `γ` parameters.
* If `track_stats=true`, accumulates mean and var statistics in training phase that will be used to renormalize the input in test phase.

!!! warn
    GroupNorm doesn't have CUDNN support. The GPU fallback is not very efficient.
"""
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
function initialstates(rng::AbstractRNG, l::GroupNorm{affine,track_stats}) where {affine,track_stats}
    return track_stats ? (μ=zeros32(rng, l.groups), σ²=ones32(rng, l.groups), training=true) : (training=true,)
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

function (GN::GroupNorm{affine,track_stats})(x::AbstractArray{T,N}, ps, st::NamedTuple) where {T,N,affine,track_stats}
    sz = size(x)
    @assert N > 2
    @assert sz[N - 1] == GN.chs

    x_ = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ GN.groups, GN.groups, sz[N])

    x_normalized, xmean, xvar = normalization_forward(
        GN,
        x_,
        get_proper_shape(GN, x_, track_stats ? st.μ : nothing, :state),
        get_proper_shape(GN, x_, track_stats ? st.σ² : nothing, :state),
        get_proper_shape(GN, x_, affine ? ps.γ : nothing, :param),
        get_proper_shape(GN, x_, affine ? ps.β : nothing, :param),
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

"""
    WeightNorm(layer::AbstractExplicitLayer, which_params::NTuple{N,Symbol}, dims::Union{Tuple,Nothing}=nothing)

Applies [weight normalization](https://arxiv.org/abs/1602.07868) to a parameter in the given layer.

``w = g\\frac{v}{\\|v\\|}``

Weight normalization is a reparameterization that decouples the magnitude of a weight tensor from its direction. This updates the parameters in `which_params` (e.g. `weight`) using two parameters: one specifying the magnitude (e.g. `weight_g`) and one specifying the direction (e.g. `weight_v`).

By default, a norm over the entire array is computed. Pass `dims` to modify the dimension.
"""
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
    ps_unnormalized = length(ps_unnormalized) == 0 ? NamedTuple() : (; ps_unnormalized...)
    return (normalized=(; ps_normalized...), unnormalized=ps_unnormalized)
end

initialstates(rng::AbstractRNG, wn::WeightNorm) = initialstates(rng, wn.layer)

function (wn::WeightNorm)(x, ps::Union{ComponentArray,NamedTuple}, s::NamedTuple)
    _ps = get_normalized_parameters(wn, wn.dims, ps.normalized)
    return wn.layer(x, merge(_ps, ps.unnormalized), s)
end

@inbounds @generated function get_normalized_parameters(
    ::WeightNorm{Val{which_params}}, dims::T, ps::Union{ComponentArray,NamedTuple}
) where {T,which_params}
    parameter_names = string.(which_params)
    v_parameter_names = Symbol.(parameter_names .* "_v")
    g_parameter_names = Symbol.(parameter_names .* "_g")
    normalized_params_symbol = [gensym(p) for p in parameter_names]

    function get_norm_except_invoke(i)
        return if T <: Tuple
            :(_norm_except(ps.$(v_parameter_names[i]), dims[$i]))
        else
            :(_norm_except(ps.$(v_parameter_names[i])))
        end
    end

    calls = []
    for i in 1:length(parameter_names)
        push!(
            calls,
            :(
                $(normalized_params_symbol[i]) =
                    ps.$(v_parameter_names[i]) .* (ps.$(g_parameter_names[i]) ./ $(get_norm_except_invoke(i)))
            ),
        )
    end
    push!(calls, :(return NamedTuple{$(which_params)}(tuple($(Tuple(normalized_params_symbol)...)))))

    return Expr(:block, calls...)
end

function Base.show(io::IO, w::WeightNorm{Val{which_params}}) where {which_params}
    return print(io, "WeightNorm{", which_params, "}(", w.layer, ")")
end
