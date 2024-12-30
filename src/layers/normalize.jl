@doc doc"""
    BatchNorm(chs::Integer, activation=identity; init_bias=zeros32, init_scale=ones32,
              affine=True(), track_stats=True(), epsilon=1f-5, momentum=0.1f0)

[Batch Normalization](https://arxiv.org/abs/1502.03167) layer.

`BatchNorm` computes the mean and variance for each ``D_1 × ... × D_{N-2} × 1 × D_N`` input
slice and normalises the input accordingly.

## Arguments

  - `chs`: Size of the channel dimension in your data. Given an array with `N` dimensions,
    call the `N-1`th the channel dimension. For a batch of feature vectors this is
    just the data dimension, for `WHCN` images it's the usual channel dimension.
  - `activation`: After normalization, elementwise activation `activation` is applied.

## Keyword Arguments

  - If `track_stats=true`, accumulates mean and variance statistics in training phase that
    will be used to renormalize the input in test phase.

  - `epsilon`: a value added to the denominator for numerical stability
  - `momentum`:  the value used for the `running_mean` and `running_var` computation
  - If `affine=true`, it also applies a shift and a rescale to the input through to
    learnable per-channel bias and scale parameters.

      + `init_bias`: Controls how the `bias` is initialized
      + `init_scale`: Controls how the `scale` is initialized

# Extended Help

## Inputs

  - `x`: Array where `size(x, N - 1) = chs`

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

Use `Lux.testmode` during inference.

## Example

```jldoctest
julia> Chain(Dense(784 => 64), BatchNorm(64, relu), Dense(64 => 10), BatchNorm(10))
Chain(
    layer_1 = Dense(784 => 64),         # 50_240 parameters
    layer_2 = BatchNorm(64, relu, affine=true, track_stats=true),  # 128 parameters, plus 129
    layer_3 = Dense(64 => 10),          # 650 parameters
    layer_4 = BatchNorm(10, affine=true, track_stats=true),  # 20 parameters, plus 21
)         # Total: 51_038 parameters,
          #        plus 150 states.
```

!!! warning

    Passing a batch size of 1, during training will result in an error.

See also [`BatchNorm`](@ref), [`InstanceNorm`](@ref), [`LayerNorm`](@ref),
[`WeightNorm`](@ref)
"""
@concrete struct BatchNorm <: AbstractLuxLayer
    activation
    epsilon <: Real
    momentum <: Real
    chs <: IntegerType
    init_bias
    init_scale
    affine <: StaticBool
    track_stats <: StaticBool
end

function BatchNorm(chs::IntegerType, activation=identity; init_bias=zeros32,
        init_scale=ones32, affine::BoolType=True(),
        track_stats::BoolType=True(), epsilon=1.0f-5, momentum=0.1f0)
    return BatchNorm(activation, epsilon, momentum, chs, init_bias,
        init_scale, static(affine), static(track_stats))
end

function initialparameters(rng::AbstractRNG, l::BatchNorm)
    has_affine(l) && return (; scale=l.init_scale(rng, l.chs), bias=l.init_bias(rng, l.chs))
    return (;)
end

function initialstates(rng::AbstractRNG, l::BatchNorm)
    if has_track_stats(l)
        return (running_mean=zeros32(rng, l.chs),
            running_var=ones32(rng, l.chs), training=Val(true))
    end
    return (; training=Val(true))
end

parameterlength(l::BatchNorm) = ifelse(has_affine(l), l.chs * 2, 0)
statelength(l::BatchNorm) = ifelse(has_track_stats(l), l.chs * 2, 0) + 1

function (BN::BatchNorm)(x::AbstractArray, ps, st::NamedTuple)
    CRC.ignore_derivatives() do
        if st.training isa Val{true}
            @argcheck size(x, ndims(x))!=1 "Batch size for BatchNorm cannot be 1 during training"
        end
    end

    x′ = match_eltype(BN, ps, st, x)
    σ = NNlib.fast_act(BN.activation, x′)
    y, stats = batchnorm(
        x′, safe_getproperty(ps, Val(:scale)), safe_getproperty(ps, Val(:bias)),
        safe_getproperty(st, Val(:running_mean)), safe_getproperty(st, Val(:running_var)),
        st.training, σ, convert(unwrapped_eltype(x′), BN.momentum),
        convert(unwrapped_eltype(x′), BN.epsilon))
    return y, update_batchnorm_state(BN, st, stats)
end

function update_batchnorm_state(BN::BatchNorm, st::NamedTuple, stats)
    has_track_stats(BN) && return merge(st,
        (; running_mean=Utils.vec(stats.running_mean),
            running_var=Utils.vec(stats.running_var)))
    return st
end

CRC.@non_differentiable update_batchnorm_state(::Any...)

function Base.show(io::IO, l::BatchNorm)
    print(io, "BatchNorm($(l.chs)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    print(io, ", affine=$(has_affine(l))")
    print(io, ", track_stats=$(has_track_stats(l))")
    return print(io, ")")
end

"""
    GroupNorm(chs::Integer, groups::Integer, activation=identity; init_bias=zeros32,
              init_scale=ones32, affine=true, epsilon=1f-5)

[Group Normalization](https://arxiv.org/abs/1803.08494) layer.

## Arguments

  - `chs`: Size of the channel dimension in your data. Given an array with `N` dimensions,
    call the `N-1`th the channel dimension. For a batch of feature vectors this is
    just the data dimension, for `WHCN` images it's the usual channel dimension.
  - `groups` is the number of groups along which the statistics are computed. The number of
    channels must be an integer multiple of the number of groups.
  - `activation`: After normalization, elementwise activation `activation` is applied.

## Keyword Arguments

  - `epsilon`: a value added to the denominator for numerical stability

  - If `affine=true`, it also applies  a shift and a rescale to the input through to
    learnable per-channel bias and scale parameters.

      + `init_bias`: Controls how the `bias` is initialized
      + `init_scale`: Controls how the `scale` is initialized

# Extended Help

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

  - `training`: Used to check if training/inference mode

Use `Lux.testmode` during inference.

## Example

```jldoctest
julia> Chain(Dense(784 => 64), GroupNorm(64, 4, relu), Dense(64 => 10), GroupNorm(10, 5))
Chain(
    layer_1 = Dense(784 => 64),         # 50_240 parameters
    layer_2 = GroupNorm(64, 4, relu, affine=true),  # 128 parameters
    layer_3 = Dense(64 => 10),          # 650 parameters
    layer_4 = GroupNorm(10, 5, affine=true),  # 20 parameters
)         # Total: 51_038 parameters,
          #        plus 0 states.
```

See also [`GroupNorm`](@ref), [`InstanceNorm`](@ref), [`LayerNorm`](@ref),
[`WeightNorm`](@ref)
"""
@concrete struct GroupNorm <: AbstractLuxLayer
    activation
    epsilon
    chs <: IntegerType
    init_bias
    init_scale
    groups <: IntegerType
    affine <: StaticBool
end

function GroupNorm(
        chs::IntegerType, groups::IntegerType, activation=identity; init_bias=zeros32,
        init_scale=ones32, affine::BoolType=True(), epsilon=1.0f-5)
    @argcheck chs % groups==0 "The number of groups ($(groups)) must divide the number of channels ($chs)"
    return GroupNorm(
        activation, epsilon, chs, init_bias, init_scale, groups, static(affine))
end

function initialparameters(rng::AbstractRNG, l::GroupNorm)
    return has_affine(l) ?
           (; scale=l.init_scale(rng, l.chs), bias=l.init_bias(rng, l.chs)) : (;)
end

parameterlength(l::GroupNorm) = has_affine(l) ? (l.chs * 2) : 0

function (GN::GroupNorm)(x::AbstractArray, ps, st::NamedTuple)
    x′ = match_eltype(GN, ps, st, x)
    σ = NNlib.fast_act(GN.activation, x′)
    y = groupnorm(x′, safe_getproperty(ps, Val(:scale)), safe_getproperty(ps, Val(:bias)),
        GN.groups, σ, convert(unwrapped_eltype(x′), GN.epsilon))
    return y, st
end

function Base.show(io::IO, l::GroupNorm)
    print(io, "GroupNorm($(l.chs), $(l.groups)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    print(io, ", affine=$(has_affine(l))")
    return print(io, ")")
end

@doc doc"""
    InstanceNorm(chs::Integer, activation=identity; init_bias=zeros32, init_scale=ones32,
                 affine=False(), track_stats=False(), epsilon=1f-5, momentum=0.1f0)

Instance Normalization. For details see [1].

Instance Normalization computes the mean and variance for each
``D_1 \times ... \times D_{N - 2} \times 1 \times 1``` input slice and normalises the input
accordingly.

## Arguments

  - `chs`: Size of the channel dimension in your data. Given an array with `N` dimensions,
    call the `N-1`th the channel dimension. For a batch of feature vectors this is
    just the data dimension, for `WHCN` images it's the usual channel dimension.
  - `activation`: After normalization, elementwise activation `activation` is applied.

## Keyword Arguments

  - If `track_stats=true`, accumulates mean and variance statistics in training phase that
    will be used to renormalize the input in test phase.

  - `epsilon`: a value added to the denominator for numerical stability
  - `momentum`:  the value used for the `running_mean` and `running_var` computation
  - If `affine=true`, it also applies  a shift and a rescale to the input through to
    learnable per-channel bias and scale parameters.

      + `init_bias`: Controls how the `bias` is initialized
      + `init_scale`: Controls how the `scale` is initialized

# Extended Help

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

Use `Lux.testmode` during inference.

## Example

```jldoctest
julia> Chain(Dense(784 => 64), InstanceNorm(64, relu; affine=true), Dense(64 => 10),
           InstanceNorm(10, relu; affine=true))
Chain(
    layer_1 = Dense(784 => 64),         # 50_240 parameters
    layer_2 = InstanceNorm(64, relu, affine=true, track_stats=false),  # 128 parameters, plus 1
    layer_3 = Dense(64 => 10),          # 650 parameters
    layer_4 = InstanceNorm(10, relu, affine=true, track_stats=false),  # 20 parameters, plus 1
)         # Total: 51_038 parameters,
          #        plus 2 states.
```

## References

[1] Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance normalization: The
    missing ingredient for fast stylization." arXiv preprint arXiv:1607.08022 (2016).

See also [`BatchNorm`](@ref), [`GroupNorm`](@ref), [`LayerNorm`](@ref), [`WeightNorm`](@ref)
"""
@concrete struct InstanceNorm <: AbstractLuxLayer
    activation
    epsilon <: Real
    momentum <: Real
    chs <: IntegerType
    init_bias
    init_scale
    affine <: StaticBool
    track_stats <: StaticBool
end

function InstanceNorm(chs::IntegerType, activation=identity; init_bias=zeros32,
        init_scale=ones32, affine::BoolType=False(),
        track_stats::BoolType=False(), epsilon=1.0f-5, momentum=0.1f0)
    return InstanceNorm(activation, epsilon, momentum, chs, init_bias,
        init_scale, static(affine), static(track_stats))
end

function initialparameters(rng::AbstractRNG, l::InstanceNorm)
    has_affine(l) && return (scale=l.init_scale(rng, l.chs), bias=l.init_bias(rng, l.chs))
    return (;)
end

function initialstates(rng::AbstractRNG, l::InstanceNorm)
    if has_track_stats(l)
        return (running_mean=zeros32(rng, l.chs),
            running_var=ones32(rng, l.chs), training=Val(true))
    end
    return (; training=Val(true))
end

parameterlength(l::InstanceNorm) = ifelse(has_affine(l), l.chs * 2, 0)
statelength(l::InstanceNorm) = ifelse(has_track_stats(l), l.chs * 2, 0) + 1

function (IN::InstanceNorm)(x::AbstractArray, ps, st::NamedTuple)
    x′ = match_eltype(IN, ps, st, x)
    σ = NNlib.fast_act(IN.activation, x′)
    y, stats = instancenorm(
        x′, safe_getproperty(ps, Val(:scale)), safe_getproperty(ps, Val(:bias)),
        safe_getproperty(st, Val(:running_mean)), safe_getproperty(st, Val(:running_var)),
        st.training, σ, convert(unwrapped_eltype(x′), IN.momentum),
        convert(unwrapped_eltype(x′), IN.epsilon))
    return y, update_instancenorm_state(IN, st, stats)
end

function update_instancenorm_state(IN::InstanceNorm, st::NamedTuple, stats)
    has_track_stats(IN) && return merge(st,
        (; running_mean=Utils.vec(stats.running_mean),
            running_var=Utils.vec(stats.running_var)))
    return st
end

CRC.@non_differentiable update_instancenorm_state(::Any...)

function Base.show(io::IO, l::InstanceNorm)
    print(io, "InstanceNorm($(l.chs)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    print(io, ", affine=$(has_affine(l))")
    print(io, ", track_stats=$(has_track_stats(l))")
    return print(io, ")")
end

@doc doc"""
    LayerNorm(shape::NTuple{N, Int}, activation=identity; epsilon=1f-5, dims=Colon(),
              affine=true, init_bias=zeros32, init_scale=ones32)

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
    learnable per-element bias and scale parameters.

      + `init_bias`: Controls how the `bias` is initialized
      + `init_scale`: Controls how the `scale` is initialized

# Extended Help

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
@concrete struct LayerNorm <: AbstractLuxLayer
    shape
    activation
    epsilon
    init_bias
    init_scale
    dims
    affine <: StaticBool
end

function LayerNorm(shape, activation=identity; epsilon=1.0f-5, dims=Colon(),
        affine::BoolType=True(), init_bias=zeros32, init_scale=ones32)
    return LayerNorm(
        shape, activation, epsilon, init_bias, init_scale, dims, static(affine))
end

function initialparameters(rng::AbstractRNG, ln::LayerNorm)
    if has_affine(ln)
        dims = (ln.shape..., 1)
        return (; bias=ln.init_bias(rng, dims...), scale=ln.init_scale(rng, dims...))
    end
    return (;)
end

function (l::LayerNorm)(x::AbstractArray, ps, st::NamedTuple)
    x′ = match_eltype(l, ps, st, x)
    σ = NNlib.fast_act(l.activation, x′)
    y = layernorm(x′, safe_getproperty(ps, Val(:scale)), safe_getproperty(ps, Val(:bias)),
        σ, l.dims, convert(unwrapped_eltype(x′), l.epsilon))
    return y, st
end

function Base.show(io::IO, l::LayerNorm)
    print(io, "LayerNorm($(l.shape)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    print(io, ", affine=$(has_affine(l)), dims=$(l.dims)")
    return print(io, ")")
end

@doc doc"""
    WeightNorm(layer::AbstractLuxLayer, which_params::NTuple{N, Symbol},
               dims::Union{Tuple, Nothing}=nothing)

Applies [weight normalization](https://arxiv.org/abs/1602.07868) to a parameter in the given
layer.

```math
w = g\frac{v}{\|v\|}
```

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
@concrete struct WeightNorm <: AbstractLuxLayer
    layer <: AbstractLuxLayer
    which_params
    dims

    function WeightNorm(
            layer::AbstractLuxLayer, which_params, dims::Union{Tuple, Nothing}=nothing)
        which_params = static(which_params)
        dims = static(dims)
        return new{typeof(layer), typeof(which_params), typeof(dims)}(
            layer, which_params, dims)
    end
end

function initialparameters(rng::AbstractRNG, wn::WeightNorm)
    ps_layer = initialparameters(rng, wn.layer)
    ps_normalized = []
    ps_unnormalized = []
    i = 1
    for k in propertynames(ps_layer)
        v = ps_layer[k]
        if k in known(wn.which_params)
            if all(iszero, v)
                throw(ArgumentError("Parameter $(k) is completely zero. This will result \
                                     in NaN gradients. Either remove this parameter from \
                                     `which_params` or modify the initialization in the \
                                     actual layer. Typically this is controlled using the \
                                     `init_$(k)` keyword argument."))
            end
            dim = wn.dims === nothing ? ndims(v) : known(wn.dims[i])
            push!(ps_normalized, Symbol(string(k) * "_g") => Utils.norm_except(v; dims=dim))
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

function (wn::WeightNorm)(x, ps, st::NamedTuple)
    y = match_eltype(wn, ps, st, x)
    psₙ = get_weight_normalized_parameters(wn, wn.dims, ps.normalized)
    return apply(wn.layer, y, Utils.merge(psₙ, ps.unnormalized), st)
end

@generated function get_weight_normalized_parameters(
        ::WeightNorm{L, WP}, dims::T, ps) where {L, WP, T}
    which_params = known(WP)
    v_parameter_names = Symbol.(which_params, :_v)
    g_parameter_names = Symbol.(which_params, :_g)
    normalized_params_symbol = [gensym(p) for p in which_params]

    function get_norm_except_invoke(i)
        return if T <: Tuple
            :(Utils.norm_except(ps.$(v_parameter_names[i]); dims=known(dims[$i])))
        else
            :(Utils.norm_except(ps.$(v_parameter_names[i])))
        end
    end

    calls = []
    for (i, (v_param, g_param)) in enumerate(zip(v_parameter_names, g_parameter_names))
        push!(calls,
            :($(normalized_params_symbol[i]) = ps.$(v_param) .* (ps.$(g_param) ./
                                                ($(get_norm_except_invoke(i)) .+
                                                 eps(eltype(ps.$(v_param)))))))
    end
    push!(calls,
        :(return NamedTuple{$(which_params)}(tuple($(Tuple(normalized_params_symbol)...)))))

    return Expr(:block, calls...)
end

function Base.show(io::IO, ::MIME"text/plain", w::WeightNorm)
    return print(io, "WeightNorm(", w.layer, ", dims = ", known(w.dims),
        ", normalized_parameters = ", known(w.which_params), ")")
end
