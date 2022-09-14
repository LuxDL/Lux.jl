"""
    ReshapeLayer(dims)

Reshapes the passed array to have a size of `(dims..., :)`

## Arguments

  - `dims`: The new dimensions of the array (excluding the last dimension).

## Inputs

  - `x`: AbstractArray of any shape which can be reshaped in `(dims..., size(x, ndims(x)))`

## Returns

  - AbstractArray of size `(dims..., size(x, ndims(x)))`
  - Empty `NamedTuple()`
"""
struct ReshapeLayer{N} <: AbstractExplicitLayer
    dims::NTuple{N, Int}
end

@inline function (r::ReshapeLayer)(x::AbstractArray, ps, st::NamedTuple)
    return reshape(x, r.dims..., size(x, ndims(x))), st
end

function Base.show(io::IO, r::ReshapeLayer)
    return print(io, "ReshapeLayer(output_dims = (", join(r.dims, ", "), ", :))")
end

"""
    FlattenLayer()

Flattens the passed array into a matrix.

## Inputs

  - `x`: AbstractArray

## Returns

  - AbstractMatrix of size `(:, size(x, ndims(x)))`
  - Empty `NamedTuple()`
"""
struct FlattenLayer <: AbstractExplicitLayer end

@inline function (f::FlattenLayer)(x::AbstractArray{T, N}, ps, st::NamedTuple) where {T, N}
    return reshape(x, :, size(x, N)), st
end

"""
    SelectDim(dim, i)

Return a view of all the data of the input `x` where the index for dimension `dim` equals
`i`. Equivalent to `view(x,:,:,...,i,:,:,...)` where `i` is in position `d`.

## Arguments

  - `dim`: Dimension for indexing
  - `i`: Index for dimension `dim`

## Inputs

  - `x`: AbstractArray that can be indexed with `view(x,:,:,...,i,:,:,...)`

## Returns

  - `view(x,:,:,...,i,:,:,...)` where `i` is in position `d`
  - Empty `NamedTuple()`
"""
struct SelectDim{dim, index} <: AbstractExplicitLayer end

SelectDim(dim, index) = SelectDim{Val(dim), Val(index)}()

@inline function (s::SelectDim{dim, index})(x, ps, st::NamedTuple) where {dim, index}
    return selectdim(x, get_known(dim), get_known(index)), st
end

function Base.show(io::IO, s::SelectDim{dim, index}) where {dim, index}
    return print(io, "SelectDim(dim = ", get_known(dim), ", index = ", get_known(index),
                 ")")
end

"""
    NoOpLayer()

As the name suggests does nothing but allows pretty printing of layers. Whatever input is
passed is returned.
"""
struct NoOpLayer <: AbstractExplicitLayer end

@inline (noop::NoOpLayer)(x, ps, st::NamedTuple) = x, st

"""
    WrappedFunction(f)

Wraps a stateless and parameter less function. Might be used when a function is added to
`Chain`. For example, `Chain(x -> relu.(x))` would not work and the right thing to do would
be `Chain((x, ps, st) -> (relu.(x), st))`. An easier thing to do would be
`Chain(WrappedFunction(Base.Fix1(broadcast, relu)))`

## Arguments

  - `f::Function`: A stateless and parameterless function

## Inputs

  - `x`: s.t `hasmethod(f, (typeof(x),))` is `true`

## Returns

  - Output of `f(x)`
  - Empty `NamedTuple()`
"""
struct WrappedFunction{F} <: AbstractExplicitLayer
    func::F
end

(wf::WrappedFunction)(x, ps, st::NamedTuple) = wf.func(x), st

function Base.show(io::IO, w::WrappedFunction)
    return print(io, "WrappedFunction(", w.func, ")")
end

"""
    Dense(in_dims => out_dims, activation=identity; init_weight=glorot_uniform,
          init_bias=zeros32, bias::Bool=true)

Create a traditional fully connected layer, whose forward pass is given by:
`y = activation.(weight * x .+ bias)`

## Arguments

  - `in_dims`: number of input dimensions
  - `out_dims`: number of output dimensions
  - `activation`: activation function

## Keyword Arguments

  - `init_weight`: initializer for the weight matrix
    (`weight = init_weight(rng, out_dims, in_dims)`)
  - `init_bias`: initializer for the bias vector (ignored if `use_bias=false`)
  - `use_bias`: Trainable bias can be disabled entirely by setting this to `false`
  - `allow_fast_activation`: If `true`, then certain activations can be approximated with
    a faster version. The new activation function will be given by
    `NNlib.fast_act(activation)`

## Input

  - `x` must be an AbstractArray with `size(x, 1) == in_dims`

## Returns

  - AbstractArray with dimensions `(out_dims, ...)` where `...` are the dimensions of `x`
  - Empty `NamedTuple()`

## Parameters

  - `weight`: Weight Matrix of size `(out_dims, in_dims)`
  - `bias`: Bias of size `(out_dims, 1)` (present if `use_bias=true`)
"""
struct Dense{use_bias, F1, F2, F3} <: AbstractExplicitLayer
    activation::F1
    in_dims::Int
    out_dims::Int
    init_weight::F2
    init_bias::F3
end

function Base.show(io::IO, d::Dense{use_bias}) where {use_bias}
    print(io, "Dense($(d.in_dims) => $(d.out_dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    use_bias || print(io, ", bias=false")
    return print(io, ")")
end

function Dense(mapping::Pair{<:Int, <:Int}, activation=identity; init_weight=glorot_uniform,
               init_bias=zeros32, use_bias::Bool=true, bias::Union{Missing, Bool}=missing)
    return Dense(first(mapping), last(mapping), activation; init_weight, init_bias,
                 use_bias, bias)
end

function Dense(in_dims::Int, out_dims::Int, activation=identity; init_weight=glorot_uniform,
               init_bias=zeros32, use_bias::Bool=true, bias::Union{Missing, Bool}=missing,
               allow_fast_activation::Bool=true)
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation

    # Deprecated Functionality (Remove in v0.5)
    if !ismissing(bias)
        Base.depwarn("`bias` argument to `Dense` has been deprecated and will be removed" *
                     " in v0.5. Use `use_bias` kwarg instead.", :Dense)
        if !use_bias
            throw(ArgumentError("Both `bias` and `use_bias` are set. Please only use " *
                                "the `use_bias` keyword argument."))
        end
        use_bias = bias
    end

    dtype = (use_bias, typeof(activation), typeof(init_weight), typeof(init_bias))
    return Dense{dtype...}(activation, in_dims, out_dims, init_weight, init_bias)
end

function initialparameters(rng::AbstractRNG, d::Dense{use_bias}) where {use_bias}
    if use_bias
        return (weight=d.init_weight(rng, d.out_dims, d.in_dims),
                bias=d.init_bias(rng, d.out_dims, 1))
    else
        return (weight=d.init_weight(rng, d.out_dims, d.in_dims),)
    end
end

function parameterlength(d::Dense{use_bias}) where {use_bias}
    return use_bias ? d.out_dims * (d.in_dims + 1) : d.out_dims * d.in_dims
end
statelength(d::Dense) = 0

@inline function (d::Dense{false})(x::AbstractVecOrMat, ps, st::NamedTuple)
    return d.activation.(ps.weight * x), st
end

@inline function (d::Dense{false, typeof(identity)})(x::AbstractVecOrMat, ps,
                                                     st::NamedTuple)
    return ps.weight * x, st
end

@inline function (d::Dense{false})(x::AbstractArray, ps, st::NamedTuple)
    sz = size(x)
    x_reshaped = reshape(x, sz[1], :)
    return reshape(d.activation.(ps.weight * x_reshaped), d.out_dims, sz[2:end]...), st
end

@inline function (d::Dense{false, typeof(identity)})(x::AbstractArray, ps, st::NamedTuple)
    sz = size(x)
    x_reshaped = reshape(x, sz[1], :)
    return reshape(ps.weight * x_reshaped, d.out_dims, sz[2:end]...), st
end

@inline function (d::Dense{true})(x::AbstractVector, ps, st::NamedTuple)
    return d.activation.(ps.weight * x .+ vec(ps.bias)), st
end

@inline function (d::Dense{true, typeof(identity)})(x::AbstractVector, ps, st::NamedTuple)
    return ps.weight * x .+ vec(ps.bias), st
end

@inline function (d::Dense{true})(x::AbstractMatrix, ps, st::NamedTuple)
    return d.activation.(ps.weight * x .+ ps.bias), st
end

@inline function (d::Dense{true, typeof(identity)})(x::AbstractMatrix, ps, st::NamedTuple)
    return ps.weight * x .+ ps.bias, st
end

@inline function (d::Dense{true})(x::AbstractArray, ps, st::NamedTuple)
    sz = size(x)
    x_reshaped = reshape(x, sz[1], :)
    return (reshape(d.activation.(ps.weight * x_reshaped .+ ps.bias), d.out_dims,
                    sz[2:end]...), st)
end

@inline function (d::Dense{true, typeof(identity)})(x::AbstractArray, ps, st::NamedTuple)
    sz = size(x)
    x_reshaped = reshape(x, sz[1], :)
    return (reshape(ps.weight * x_reshaped .+ ps.bias, d.out_dims, sz[2:end]...), st)
end

"""
    Scale(dims, activation=identity; init_weight=ones32, init_bias=zeros32, bias::Bool=true)

Create a Sparsely Connected Layer with a very specific structure (only Diagonal
Elements are non-zero). The forward pass is given by: `y = activation.(weight .* x .+ bias)`

## Arguments

  - `dims`: size of the learnable scale and bias parameters.
  - `activation`: activation function

## Keyword Arguments

  - `init_weight`: initializer for the weight matrix
    (`weight = init_weight(rng, out_dims, in_dims)`)
  - `init_bias`: initializer for the bias vector (ignored if `use_bias=false`)
  - `use_bias`: Trainable bias can be disabled entirely by setting this to `false`
  - `allow_fast_activation`: If `true`, then certain activations can be approximated with
    a faster version. The new activation function will be given by
    `NNlib.fast_act(activation)`

## Input

  - `x` must be an Array of size `(dims..., B)` or `(dims...[0], ..., dims[k])`
    for `k ≤ size(dims)`

## Returns

  - Array of size `(dims..., B)` or `(dims...[0], ..., dims[k])` for `k ≤ size(dims)`
  - Empty `NamedTuple()`

## Parameters

  - `weight`: Weight Array of size `(dims...)`
  - `bias`: Bias of size `(dims...)`

!!! compat "Lux 0.4.3"
    
    `Scale` with multiple dimensions requires at least Lux 0.4.3.
"""
struct Scale{use_bias, F1, D, F2, F3} <: AbstractExplicitLayer
    activation::F1
    dims::D
    init_weight::F2
    init_bias::F3
end

function Base.show(io::IO, d::Scale)
    print(io, "Scale($(d.dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    return print(io, ")")
end

function Scale(dims::Tuple{Vararg{Integer}}, activation=identity;
               init_weight=glorot_uniform, init_bias=zeros32, use_bias::Bool=true,
               bias::Union{Missing, Bool}=missing, allow_fast_activation::Bool=true)
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation

    # Deprecated Functionality (Remove in v0.5)
    if !ismissing(bias)
        Base.depwarn("`bias` argument to `Scale` has been deprecated and will be removed" *
                     " in v0.5. Use `use_bias` kwarg instead.", :Scale)
        if !use_bias
            throw(ArgumentError("Both `bias` and `use_bias` are set. Please only use " *
                                "the `use_bias` keyword argument."))
        end
        use_bias = bias
    end

    return Scale{use_bias, typeof(activation), typeof(dims), typeof(init_weight),
                 typeof(init_bias)}(activation, dims, init_weight, init_bias)
end

function Scale(s1::Integer, s23::Integer...; _act=identity, kw...)
    return Scale(tuple(s1, s23...), _act; kw...)
end
Scale(size_act...; kw...) = Scale(size_act[1:(end - 1)]...; _act=size_act[end], kw...)

function initialparameters(rng::AbstractRNG, d::Scale{use_bias}) where {use_bias}
    if use_bias
        return (weight=d.init_weight(rng, d.dims...), bias=d.init_bias(rng, d.dims...))
    else
        return (weight=d.init_weight(rng, d.dims...),)
    end
end

parameterlength(d::Scale{use_bias}) where {use_bias} = (1 + use_bias) * prod(d.dims)
statelength(d::Scale) = 0

function (d::Scale{true})(x::AbstractArray, ps, st::NamedTuple)
    return d.activation.(ps.weight .* x .+ ps.bias), st
end

function (d::Scale{true, typeof(identity)})(x::AbstractArray, ps, st::NamedTuple)
    return ps.weight .* x .+ ps.bias, st
end

function (d::Scale{false})(x::AbstractArray, ps, st::NamedTuple)
    return d.activation.(ps.weight .* x), st
end

function (d::Scale{false, typeof(identity)})(x::AbstractArray, ps, st::NamedTuple)
    return ps.weight .* x, st
end
