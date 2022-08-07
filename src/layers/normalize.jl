abstract type AbstractNormalizationLayer{affine, track_stats} <: AbstractExplicitLayer end

"""
    BatchNorm(chs::Integer, activation=identity; init_bias=zeros32, init_scale=ones32,
              affine=true, track_stats=true, epsilon=1f-5, momentum=0.1f0)

[Batch Normalization](https://arxiv.org/abs/1502.03167) layer.

`BatchNorm` computes the mean and variance for each `D_1 × ... × D_{N-2} × 1 × D_N` input
slice and normalises the input accordingly.

## Arguments

  - `chs`: Size of the channel dimension in your data. Given an array with `N` dimensions,
    call the `N-1`th the channel dimension. For a batch of feature vectors this is
    just the data dimension, for `WHCN` images it's the usual channel dimension.
  - `activation`: After normalization, elementwise activation `activation` is applied.

## Keyword Arguments

  - If `affine=true`, it also applies  a shift and a rescale to the input through to
    learnable per-channel bias and scale parameters.
    
      + `init_bias`: Controls how the `bias` is initiliazed
      + `init_scale`: Controls how the `scale` is initiliazed

  - If `track_stats=true`, accumulates mean and variance statistics in training phase that
    will be used to renormalize the input in test phase.
  - `epsilon`: a value added to the denominator for numerical stability
  - `momentum`:  the value used for the `running_mean` and `running_var` computation

## Inputs

  - `x`: Array where `size(x, N - 1) = chs` and `ndims(x) > 2`

## Returns

  - `y`: Normalized Array
  - Update model state

## Parameters

  - `affine=true`
    
      + `bias`: Bias of shape `(chs,)`
      + `scale`: Scale of shape `(chs,)`

  - `affine=false` - Empty `NamedTuple()`

## States

  - Statistics if `track_stats=true`
    
      + `running_mean`: Running mean of shape `(chs,)`
      + `running_var`: Running variance of shape `(chs,)`

  - Statistics if `track_stats=false`
    
      + `running_mean`: nothing
      + `running_var`: nothing
  - `training`: Used to check if training/inference mode

Use [`Lux.testmode`](@ref) during inference.

## Example

```julia
m = Chain(Dense(784 => 64), BatchNorm(64, relu), Dense(64 => 10), BatchNorm(10))
```

!!! warning
    
    Passing a batch size of 1, during training will result in NaNs.

See also [`GroupNorm`](@ref)
"""
struct BatchNorm{affine, track_stats, F1, F2, F3, N} <:
       AbstractNormalizationLayer{affine, track_stats}
    activation::F1
    epsilon::N
    momentum::N
    chs::Int
    init_bias::F2
    init_scale::F3
end

function BatchNorm(chs::Int, activation=identity; init_bias=zeros32, init_scale=ones32,
                   affine::Bool=true, track_stats::Bool=true, epsilon=1.0f-5,
                   momentum=0.1f0)
    activation = NNlib.fast_act(activation)
    return BatchNorm{affine, track_stats, typeof(activation), typeof(init_bias),
                     typeof(init_scale), typeof(epsilon)}(activation, epsilon, momentum,
                                                          chs, init_bias, init_scale)
end

function initialparameters(rng::AbstractRNG, l::BatchNorm{affine}) where {affine}
    if affine
        return (scale=l.init_scale(rng, l.chs), bias=l.init_bias(rng, l.chs))
    else
        return (scale=nothing, bias=nothing)
    end
end

function initialstates(rng::AbstractRNG,
                       l::BatchNorm{affine, track_stats}) where {affine, track_stats}
    return if track_stats
        (running_mean=zeros32(rng, l.chs), running_var=ones32(rng, l.chs),
         training=Val(true))
    else
        (running_mean=nothing, running_var=nothing, training=Val(true))
    end
end

parameterlength(l::BatchNorm{affine}) where {affine} = affine ? (l.chs * 2) : 0
function statelength(l::BatchNorm{affine, track_stats}) where {affine, track_stats}
    return (track_stats ? 2 * l.chs : 0) + 1
end

function (BN::BatchNorm)(x::AbstractArray{T, N}, ps, st::NamedTuple) where {T, N}
    x_normalized, xmean, xvar = normalization(x, st.running_mean, st.running_var, ps.scale,
                                              ps.bias, BN.activation,
                                              collect([1:(N - 2); N]), st.training,
                                              BN.momentum, BN.epsilon)

    st = merge(st, (running_mean=xmean, running_var=xvar))

    return x_normalized, st
end

function (BN::BatchNorm{affine, track_stats})(x::Union{CuArray{T, 2}, CuArray{T, 4},
                                                       CuArray{T, 5}}, ps,
                                              st::NamedTuple) where {
                                                                     T <:
                                                                     Union{Float32, Float64
                                                                           }, affine,
                                                                     track_stats}
    # NNlibCUDA silently updates running_mean and running_var so copying them
    if istraining(st)
        running_mean2 = track_stats ? copy(st.running_mean) : nothing
        running_var2 = track_stats ? copy(st.running_var) : nothing
    else
        if track_stats
            running_mean2 = copy(st.running_mean)
            running_var2 = copy(st.running_var)
        else
            N = ndims(x)
            reduce_dims = collect([1:(N - 2); N])
            running_mean2 = mean(x; dims=reduce_dims)
            running_var2 = var(x; mean=running_mean2, dims=reduce_dims, corrected=false)
        end
    end
    res = BN.activation.(batchnorm(affine ? ps.scale : nothing, affine ? ps.bias : nothing,
                                   x, running_mean2, running_var2, BN.momentum;
                                   eps=BN.epsilon, training=istraining(st)))
    if track_stats
        st = merge(st, (running_mean=running_mean2, running_var=running_var2))
    end
    return res, st
end

function Base.show(io::IO, l::BatchNorm{affine, track_stats}) where {affine, track_stats}
    print(io, "BatchNorm($(l.chs)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    print(io, ", affine=$(affine)")
    print(io, ", track_stats=$(track_stats)")
    return print(io, ")")
end

"""
    GroupNorm(chs::Integer, groups::Integer, activation=identity; init_bias=zeros32,
              init_scale=ones32, affine=true, track_stats=true, epsilon=1f-5,
              momentum=0.1f0)

[Group Normalization](https://arxiv.org/abs/1803.08494) layer.

## Arguments

  - `chs`: Size of the channel dimension in your data. Given an array with `N` dimensions,
    call the `N-1`th the channel dimension. For a batch of feature vectors this is
    just the data dimension, for `WHCN` images it's the usual channel dimension.
  - `groups` is the number of groups along which the statistics are computed. The number of
    channels must be an integer multiple of the number of groups.
  - `activation`: After normalization, elementwise activation `activation` is applied.

## Keyword Arguments

  - If `affine=true`, it also applies  a shift and a rescale to the input through to
    learnable per-channel bias and scale parameters.
    
      + `init_bias`: Controls how the `bias` is initiliazed
      + `init_scale`: Controls how the `scale` is initiliazed

  - If `track_stats=true`, accumulates mean and variance statistics in training phase that
    will be used to renormalize the input in test phase. **(This feature has been
    deprecated and will be removed in v0.5)**
  - `epsilon`: a value added to the denominator for numerical stability
  - `momentum`:  the value used for the `running_mean` and `running_var` computation **(This
    feature has been deprecated and will be removed in v0.5)**

## Inputs

  - `x`: Array where `size(x, N - 1) = chs` and `ndims(x) > 2`

## Returns

  - `y`: Normalized Array
  - Update model state

## Parameters

  - `affine=true`
    
      + `bias`: Bias of shape `(chs,)`
      + `scale`: Scale of shape `(chs,)`

  - `affine=false` - Empty `NamedTuple()`

## States

  - Statistics if `track_stats=true` **(DEPRECATED)**
    
      + `running_mean`: Running mean of shape `(groups,)`
      + `running_var`: Running variance of shape `(groups,)`

  - Statistics if `track_stats=false`
    
      + `running_mean`: nothing
      + `running_var`: nothing
  - `training`: Used to check if training/inference mode

Use [`Lux.testmode`](@ref) during inference.

## Example

```julia
m = Chain(Dense(784 => 64), GroupNorm(64, 4, relu), Dense(64 => 10), GroupNorm(10, 5))
```

!!! warning
    
    GroupNorm doesn't have CUDNN support. The GPU fallback is not very efficient.

See also [`BatchNorm`](@ref)
"""
struct GroupNorm{affine, track_stats, F1, F2, F3, N} <:
       AbstractNormalizationLayer{affine, track_stats}
    activation::F1
    epsilon::N
    momentum::N
    chs::Int
    init_bias::F2
    init_scale::F3
    groups::Int
end

function GroupNorm(chs::Integer, groups::Integer, activation=identity; init_bias=zeros32,
                   init_scale=ones32, affine=true, track_stats=missing, epsilon=1.0f-5,
                   momentum=missing)
    @assert chs % groups==0 "The number of groups ($(groups)) must divide the number of channels ($chs)"
    activation = NNlib.fast_act(activation)

    # Deprecated Functionality (Remove in v0.5)
    if !ismissing(momentum)
        Base.depwarn("`momentum` for `GroupNorm` has been deprecated and will be removed " *
                     "in v0.5", :GroupNorm)
    else
        momentum = 0.1f0
    end
    if !ismissing(track_stats)
        if track_stats
            Base.depwarn("`track_stats` for `GroupNorm` has been deprecated and will be " *
                         "removed in v0.5", :GroupNorm)
        end
    else
        track_stats = true
    end

    return GroupNorm{affine, track_stats, typeof(activation), typeof(init_bias),
                     typeof(init_scale), typeof(epsilon)}(activation, epsilon, momentum,
                                                          chs, init_bias, init_scale,
                                                          groups)
end

function initialparameters(rng::AbstractRNG, l::GroupNorm{affine}) where {affine}
    if affine
        return (scale=l.init_scale(rng, l.chs), bias=l.init_bias(rng, l.chs))
    else
        return (scale=nothing, bias=nothing)
    end
end

function initialstates(rng::AbstractRNG,
                       l::GroupNorm{affine, track_stats}) where {affine, track_stats}
    return if track_stats
        (running_mean=zeros32(rng, l.groups), running_var=ones32(rng, l.groups),
         training=Val(true))
    else
        (running_mean=nothing, running_var=nothing, training=Val(true))
    end
end

parameterlength(l::GroupNorm{affine}) where {affine} = affine ? (l.chs * 2) : 0

function statelength(l::GroupNorm{affine, track_stats}) where {affine, track_stats}
    return (track_stats ? 2 * l.groups : 0) + 1
end

function (GN::GroupNorm)(x::AbstractArray{T, N}, ps, st::NamedTuple) where {T, N}
    sz = size(x)
    x_ = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ GN.groups, GN.groups, sz[N])

    x_normalized, xmean, xvar = normalization(x_, st.running_mean, st.running_var, ps.scale,
                                              ps.bias, GN.activation, collect(1:(N - 1)),
                                              st.training, GN.momentum, GN.epsilon)

    st = merge(st, (running_mean=xmean, running_var=xvar))

    return reshape(x_normalized, sz), st
end

function Base.show(io::IO, l::GroupNorm{affine, track_stats}) where {affine, track_stats}
    print(io, "GroupNorm($(l.chs), $(l.groups)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    print(io, ", affine=$(affine)")
    print(io, ", track_stats=$(track_stats)")
    return print(io, ")")
end

@doc doc"""
    WeightNorm(layer::AbstractExplicitLayer, which_params::NTuple{N,Symbol},
               dims::Union{Tuple,Nothing}=nothing)

Applies [weight normalization](https://arxiv.org/abs/1602.07868) to a parameter in the given
layer.

``w = g\frac{v}{\|v\|}``

Weight normalization is a reparameterization that decouples the magnitude of a weight tensor
from its direction. This updates the parameters in `which_params` (e.g. `weight`) using two
parameters: one specifying the magnitude (e.g. `weight_g`) and one specifying the direction
(e.g. `weight_v`).

## Arguments

  - `layer` whose parameters are being reparameterized
  - `which_params`: parameter names for the parameters being reparameterized
  - By default, a norm over the entire array is computed. Pass `dims` to modify the
    dimension.

## Inputs

  - `x`: Should be of valid type for input to `layer`

## Returns

  - Output from `layer`
  - Updated model state of `layer`

## Parameters

  - `normalized`: Parameters of `layer` that are being normalized
  - `unnormalized`: Parameters of `layer` that are not being normalized

## States

  - Same as that of `layer`
"""
struct WeightNorm{which_params, L <: AbstractExplicitLayer, D} <: AbstractExplicitLayer
    layer::L
    dims::D
end

function WeightNorm(layer::AbstractExplicitLayer, which_params::NTuple{N, Symbol},
                    dims::Union{Tuple, Nothing}=nothing) where {N}
    return WeightNorm{Val{which_params}, typeof(layer), typeof(dims)}(layer, dims)
end

@inline _norm(x; dims=Colon()) = sqrt.(sum(abs2, x; dims=dims))
@inline function _norm_except(x::AbstractArray{T, N};
                              dims::Union{Int, Tuple}=N) where {T, N}
    return _norm(x; dims=_get_norm_except_dims(N, dims))
end
@inline _get_norm_except_dims(N, dim::Int) = filter(i -> i != dim, 1:N)
@inline _get_norm_except_dims(N, dims::Tuple) = filter(i -> !(i in dims), 1:N)

function initialparameters(rng::AbstractRNG,
                           wn::WeightNorm{Val{which_params}}) where {which_params}
    ps_layer = initialparameters(rng, wn.layer)
    ps_normalized = []
    ps_unnormalized = []
    i = 1
    for k in propertynames(ps_layer)
        v = ps_layer[k]
        if k in which_params
            if all(iszero, v)
                msg = ("Parameter $(k) is completely zero. This will result in NaN " *
                       "gradients. Either remove this parameter from `which_params` or " *
                       "modify the initialization in the actual layer. Typically this is " *
                       "controlled using the `init_$(k)` keyword argument.")
                # FIXME(@avik-pal): This is not really an ArgumentError
                throw(ArgumentError(msg))
            end
            dim = wn.dims === nothing ? ndims(v) : wn.dims[i]
            push!(ps_normalized, Symbol(string(k) * "_g") => _norm_except(v; dims=dim))
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

function (wn::WeightNorm)(x, ps, s::NamedTuple)
    _ps = _get_normalized_parameters(wn, wn.dims, ps.normalized)
    return wn.layer(x, merge(_ps, ps.unnormalized), s)
end

@inbounds @generated function _get_normalized_parameters(::WeightNorm{Val{which_params}},
                                                         dims::T,
                                                         ps) where {T, which_params}
    parameter_names = string.(which_params)
    v_parameter_names = Symbol.(parameter_names .* "_v")
    g_parameter_names = Symbol.(parameter_names .* "_g")
    normalized_params_symbol = [gensym(p) for p in parameter_names]

    function get_norm_except_invoke(i)
        return if T <: Tuple
            :(_norm_except(ps.$(v_parameter_names[i]); dims=dims[$i]))
        else
            :(_norm_except(ps.$(v_parameter_names[i])))
        end
    end

    calls = []
    for i in 1:length(parameter_names)
        push!(calls,
              :($(normalized_params_symbol[i]) = ps.$(v_parameter_names[i]) .*
                                                 (ps.$(g_parameter_names[i]) ./
                                                  ($(get_norm_except_invoke(i)) .+
                                                   eps(eltype(ps.$(v_parameter_names[i])))))))
    end
    push!(calls,
          :(return NamedTuple{$(which_params)}(tuple($(Tuple(normalized_params_symbol)...)))))

    return Expr(:block, calls...)
end

function Base.show(io::IO, w::WeightNorm{Val{which_params}}) where {which_params}
    return print(io, "WeightNorm{", which_params, "}(", w.layer, ")")
end

@doc doc"""
    LayerNorm(shape::NTuple{N, Int}, activation=identity; epsilon=1f-5, dims=Colon(),
              affine::Bool=false, init_bias=zeros32, init_scale=ones32,)

Computes mean and standard deviation over the whole input array, and uses these to
normalize the whole array. Optionally applies an elementwise affine transformation
afterwards.

Given an input array ``x``, this layer computes

```math
y = \frac{x - \mathbb{E}[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta
```

where ``\gamma`` & ``\beta`` are trainable parameters if `affine=true`.

## Arguments

  - `shape`: Broadcastable shape of input array excluding the batch dimension.
  - `activation`: After normalization, elementwise activation `activation` is applied.

## Keyword Arguments

  - `epsilon`: a value added to the denominator for numerical stability.
  - `dims`: Dimensions to normalize the array over.
  - If `affine=true`, it also applies  a shift and a rescale to the input through to
    learnable per-channel bias and scale parameters.

      + `init_bias`: Controls how the `bias` is initiliazed
      + `init_scale`: Controls how the `scale` is initiliazed


## Inputs

  - `x`: AbstractArray

## Returns

  - `y`: Normalized Array
  - Empty NamedTuple()

## Parameters

  - `affine=false`: Empty `NamedTuple()`
  - `affine=true`
    
      + `bias`: Bias of shape `(shape..., 1)`
      + `scale`: Scale of shape `(shape..., 1)`
"""
struct LayerNorm{affine, F1, N, T, F2, F3, D} <: AbstractExplicitLayer
    shape::NTuple{N, Int}
    activation::F1
    epsilon::T
    init_bias::F2
    init_scale::F3
    dims::D
end

function LayerNorm(shape::NTuple{N, <:Int}, activation=identity; epsilon::T=1.0f-5,
                   dims=Colon(), affine::Bool=true, init_bias=zeros32,
                   init_scale=ones32) where {N, T}
    activation = NNlib.fast_act(activation)
    return LayerNorm{affine, typeof(activation), N, T, typeof(init_bias),
                     typeof(init_scale), typeof(dims)}(shape, activation, epsilon,
                                                       init_bias, init_scale, dims)
end

function initialparameters(rng::AbstractRNG, ln::LayerNorm{affine}) where {affine}
    if affine
        return (bias=ln.init_bias(rng, ln.shape..., 1),
                scale=ln.init_scale(rng, ln.shape..., 1))
    else
        return NamedTuple()
    end
end

@generated function (l::LayerNorm{affine, F1})(x::AbstractArray, ps,
                                               st::NamedTuple) where {affine, F1}
    calls = []
    push!(calls, :(_mean = mean(x; dims=l.dims);
                   _var = var(x; corrected=false, mean=_mean)))

    if affine
        if F1 == typeof(identity)
            push!(calls,
                  :(return ps.scale .* (x .- _mean) ./ sqrt.(_var .+ l.epsilon) .+ ps.bias,
                           st))
        else
            push!(calls,
                  :(return l.activation.(ps.scale .* (x .- _mean) ./
                                         sqrt.(_var .+ l.epsilon) .+ ps.bias), st))
        end
    else
        if F1 == typeof(identity)
            push!(calls, :(return (x .- _mean) ./ sqrt.(_var .+ l.epsilon), st))
        else
            push!(calls,
                  :(return l.activation.((x .- _mean) ./ sqrt.(_var .+ l.epsilon)), st))
        end
    end
    return Expr(:block, calls...)
end

function Base.show(io::IO, l::LayerNorm{affine}) where {affine}
    print(io, "LayerNorm($(l.shape)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    print(io, ", affine=$(affine), dims=$(l.dims)")
    return print(io, ")")
end
