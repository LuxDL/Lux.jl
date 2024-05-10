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

outputsize(r::ReshapeLayer) = r.dims

@inline function (r::ReshapeLayer)(x::AbstractArray, ps, st::NamedTuple)
    return reshape(x, r.dims..., size(x, ndims(x))), st
end

function Base.show(io::IO, r::ReshapeLayer)
    return print(io, "ReshapeLayer(output_dims = (", join(r.dims, ", "), ", :))")
end

"""
    FlattenLayer(N = nothing)

Flattens the passed array into a matrix.

## Arguments

  - `N`: Flatten the first `N` dimensions of the input array. If `nothing`, then all
    dimensions (except) are flattened. Note that the batch dimension is never flattened.

## Inputs

  - `x`: AbstractArray

## Returns

  - AbstractMatrix of size `(:, size(x, ndims(x)))`
  - Empty `NamedTuple()`
"""
@kwdef @concrete struct FlattenLayer <: AbstractExplicitLayer
    N = nothing
end

@inline function (f::FlattenLayer)(x::AbstractArray{T, N}, ps, st::NamedTuple) where {T, N}
    f.N === nothing && return reshape(x, :, size(x, N)), st
    @assert f.N < N
    return reshape(x, :, size(x)[(f.N + 1):end]...), st
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

SelectDim(dim, index) = SelectDim{dim, index}()

@inline function (s::SelectDim{dim, index})(x, ps, st::NamedTuple) where {dim, index}
    return selectdim(x, dim, index), st
end

function Base.show(io::IO, s::SelectDim{dim, index}) where {dim, index}
    return print(io, "SelectDim(dim = ", dim, ", index = ", index, ")")
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
@concrete struct WrappedFunction <: AbstractExplicitLayer
    func
end

(wf::WrappedFunction)(x, ps, st::NamedTuple) = wf.func(x), st

function Base.show(io::IO, w::WrappedFunction)
    return print(io, "WrappedFunction(", w.func, ")")
end

"""
    Dense(in_dims => out_dims, activation=identity; init_weight=glorot_uniform,
          init_bias=zeros32, use_bias::Bool=true, allow_fast_activation::Bool=true)

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
@concrete struct Dense{use_bias} <: AbstractExplicitLayer
    activation
    in_dims::Int
    out_dims::Int
    init_weight
    init_bias
end

function Base.show(io::IO, d::Dense{use_bias}) where {use_bias}
    print(io, "Dense($(d.in_dims) => $(d.out_dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    use_bias || print(io, ", bias=false")
    return print(io, ")")
end

function Dense(mapping::Pair{<:Int, <:Int}, activation=identity; kwargs...)
    return Dense(first(mapping), last(mapping), activation; kwargs...)
end

function Dense(in_dims::Int, out_dims::Int, activation=identity; init_weight=glorot_uniform,
        init_bias=zeros32, use_bias::Bool=true, allow_fast_activation::Bool=true)
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
    return Dense{use_bias}(activation, in_dims, out_dims, init_weight, init_bias)
end

function initialparameters(rng::AbstractRNG, d::Dense{use_bias}) where {use_bias}
    if use_bias
        return (weight=d.init_weight(rng, d.out_dims, d.in_dims),
            bias=d.init_bias(rng, d.out_dims, 1)) #TODO: In v0.6 make it a vector
    else
        return (weight=d.init_weight(rng, d.out_dims, d.in_dims),)
    end
end

function parameterlength(d::Dense{use_bias}) where {use_bias}
    return use_bias ? d.out_dims * (d.in_dims + 1) : d.out_dims * d.in_dims
end
statelength(d::Dense) = 0

outputsize(d::Dense) = (d.out_dims,)

@inline function (d::Dense)(x::AbstractVector, ps, st::NamedTuple)
    return vec(first(d(reshape(x, :, 1), ps, st))), st
end

@inline function (d::Dense)(x::AbstractMatrix, ps, st::NamedTuple)
    return (
        fused_dense_bias_activation(
            d.activation, ps.weight, x, _vec(_getproperty(ps, Val(:bias)))),
        st)
end

@inline function (d::Dense)(x::AbstractArray, ps, st::NamedTuple)
    return reshape(first(d(reshape(x, size(x, 1), :), ps, st)), :, size(x)[2:end]...), st
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
"""
@concrete struct Scale{use_bias} <: AbstractExplicitLayer
    activation
    dims
    init_weight
    init_bias
end

function Base.show(io::IO, d::Scale)
    print(io, "Scale($(d.dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    return print(io, ")")
end

function Scale(
        dims::Tuple{Vararg{Integer}}, activation=identity; init_weight=glorot_uniform,
        init_bias=zeros32, use_bias::Bool=true, allow_fast_activation::Bool=true)
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
    return Scale{use_bias}(activation, dims, init_weight, init_bias)
end

function Scale(s1::Integer, s23::Integer...; _act=identity, kwargs...)
    return Scale(tuple(s1, s23...), _act; kwargs...)
end
function Scale(size_act...; kwargs...)
    return Scale(size_act[1:(end - 1)]...; _act=size_act[end], kwargs...)
end

function initialparameters(rng::AbstractRNG, d::Scale{use_bias}) where {use_bias}
    if use_bias
        return (weight=d.init_weight(rng, d.dims...), bias=d.init_bias(rng, d.dims...))
    else
        return (weight=d.init_weight(rng, d.dims...),)
    end
end

parameterlength(d::Scale{use_bias}) where {use_bias} = (1 + use_bias) * prod(d.dims)
statelength(d::Scale) = 0

outputsize(d::Scale) = d.dims

function (d::Scale{true})(x::AbstractArray, ps, st::NamedTuple)
    return apply_bias_activation(d.activation, ps.weight .* x, ps.bias), st
end

function (d::Scale{false})(x::AbstractArray, ps, st::NamedTuple)
    return apply_activation(d.activation, ps.weight .* x), st
end

"""
    Bilinear((in1_dims, in2_dims) => out, activation=identity; init_weight=glorot_uniform,
             init_bias=zeros32, use_bias::Bool=true, allow_fast_activation::Bool=true)
    Bilinear(in12_dims => out, activation=identity; init_weight=glorot_uniform,
             init_bias=zeros32, use_bias::Bool=true, allow_fast_activation::Bool=true)

Create a fully connected layer between two inputs and an output, and otherwise similar to
[`Dense`](@ref). Its output, given vectors `x` & `y`, is another vector `z` with, for all
`i in 1:out`:

`z[i] = activation(x' * W[i, :, :] * y + bias[i])`

If `x` and `y` are matrices, then each column of the output `z = B(x, y)` is of this form,
with `B` the Bilinear layer.

## Arguments

  - `in1_dims`: number of input dimensions of `x`
  - `in2_dims`: number of input dimensions of `y`
  - `in12_dims`: If specified, then `in1_dims = in2_dims = in12_dims`
  - `out`: number of output dimensions
  - `activation`: activation function

## Keyword Arguments

  - `init_weight`: initializer for the weight matrix
    (`weight = init_weight(rng, out_dims, in1_dims, in2_dims)`)
  - `init_bias`: initializer for the bias vector (ignored if `use_bias=false`)
  - `use_bias`: Trainable bias can be disabled entirely by setting this to `false`
  - `allow_fast_activation`: If `true`, then certain activations can be approximated with
    a faster version. The new activation function will be given by
    `NNlib.fast_act(activation)`

## Input

  - A 2-Tuple containing

      + `x` must be an AbstractArray with `size(x, 1) == in1_dims`
      + `y` must be an AbstractArray with `size(y, 1) == in2_dims`

  - If the input is an AbstractArray, then `x = y`

## Returns

  - AbstractArray with dimensions `(out_dims, size(x, 2))`
  - Empty `NamedTuple()`

## Parameters

  - `weight`: Weight Matrix of size `(out_dims, in1_dims, in2_dims)`
  - `bias`: Bias of size `(out_dims, 1)` (present if `use_bias=true`)
"""
@concrete struct Bilinear{use_bias} <: AbstractExplicitLayer
    activation
    in1_dims::Int
    in2_dims::Int
    out_dims::Int
    init_weight
    init_bias
end

function Base.show(io::IO, b::Bilinear{use_bias}) where {use_bias}
    print(io, "Bilinear(($(b.in1_dims), $(b.in2_dims)) => $(b.out_dims)")
    (b.activation == identity) || print(io, ", $(b.activation)")
    use_bias || print(io, ", bias=false")
    return print(io, ")")
end

function Bilinear(((in1_dims, in2_dims), out)::Pair{<:Tuple, <:Integer},
        activation=identity; init_weight=glorot_uniform, init_bias=zeros32,
        use_bias::Bool=true, allow_fast_activation::Bool=true)
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
    return Bilinear{use_bias}(activation, in1_dims, in2_dims, out, init_weight, init_bias)
end
function Bilinear(
        (in12_dims, out)::Pair{<:Integer, <:Integer}, activation=identity; kwargs...)
    return Bilinear((in12_dims, in12_dims) => out, activation; kwargs...)
end

function initialparameters(rng::AbstractRNG, b::Bilinear{use_bias}) where {use_bias}
    if use_bias
        return (weight=b.init_weight(rng, b.out_dims, b.in1_dims, b.in2_dims),
            bias=b.init_bias(rng, b.out_dims, 1))
    else
        return (weight=b.init_weight(rng, b.out_dims, b.in1_dims, b.in2_dims),)
    end
end

function parameterlength(b::Bilinear{use_bias}) where {use_bias}
    return b.out_dims * b.in1_dims * b.in2_dims + use_bias * b.out_dims
end
statelength(b::Bilinear) = 0

outputsize(b::Bilinear) = (b.out_dims,)

function (b::Bilinear{use_bias})((x, y)::Tuple{<:AbstractVecOrMat, <:AbstractVecOrMat},
        ps, st::NamedTuple) where {use_bias}
    d_z, d_x, d_y = size(ps.weight)
    if d_x != size(x, 1) || d_y != size(y, 1)
        throw(DimensionMismatch(lazy"number of rows in data must match `ps.weight`"))
    end
    if size(x, 2) != size(y, 2)
        throw(DimensionMismatch(lazy"data inputs must agree on batch size, got $(size(x, 2)) and $(size(y, 2))"))
    end

    Wy = reshape(reshape(ps.weight, (:, d_y)) * y, (d_z, d_x, :))
    Wyx = reshape(batched_mul(Wy, reshape(x, (d_x, 1, :))), (d_z, :))

    if use_bias
        return apply_bias_activation(b.activation, Wyx, ps.bias), st
    else
        return apply_activation(b.activation, Wyx), st
    end
end

function (b::Bilinear)((x, y)::Tuple{<:AbstractArray, <:AbstractArray}, ps, st::NamedTuple)
    if size(x)[2:end] != size(y)[2:end]
        throw(DimensionMismatch("data arrays must agree on batch dimensions, " *
                                "got sizes $(size(x)), and $(size(y))"))
    end

    d_z, d_x, d_y = size(ps.weight)

    x_reshaped = reshape(x, d_x, :)
    y_reshaped = reshape(y, d_y, :)

    z, st = b((x_reshaped, y_reshaped), ps, st)

    return reshape(z, d_z, size(x)[2:end]...), st
end

(b::Bilinear)(x::AbstractArray, ps, st::NamedTuple) = b((x, x), ps, st)

"""
    Embedding(in_dims => out_dims; init_weight=randn32)

A lookup table that stores embeddings of dimension `out_dims` for a vocabulary of size
`in_dims`.

This layer is often used to store word embeddings and retrieve them using indices.

!!! warning

    Unlike `Flux.Embedding`, this layer does not support using `OneHotArray` as an input.

## Arguments

  - `in_dims`: number of input dimensions
  - `out_dims`: number of output dimensions

## Keyword Arguments

  - `init_weight`: initializer for the weight matrix
    (`weight = init_weight(rng, out_dims, in_dims)`)

## Input

  - Integer OR
  - Abstract Vector of Integers OR
  - Abstract Array of Integers

## Returns

  - Returns the embedding corresponding to each index in the input. For an N dimensional
    input, an N + 1 dimensional output is returned.
  - Empty `NamedTuple()`
"""
@concrete struct Embedding <: AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    init_weight
end

function Embedding((in_dims, out_dims)::Pair{<:Integer, <:Integer}; init_weight=randn32)
    return Embedding(in_dims, out_dims, init_weight)
end

function initialparameters(rng::AbstractRNG, e::Embedding)
    return (weight=e.init_weight(rng, e.out_dims, e.in_dims),)
end

(e::Embedding)(x::Integer, ps, st::NamedTuple) = view(ps.weight, :, x), st
function (e::Embedding)(x::AbstractVector{<:Integer}, ps, st::NamedTuple)
    return NNlib.gather(ps.weight, x), st
end
function (e::Embedding)(x::AbstractArray{<:Integer}, ps, st::NamedTuple)
    return reshape(e(vec(x), ps, st)[1], :, size(x)...), st
end

function Base.show(io::IO, e::Embedding)
    return print(io, "Embedding(", e.in_dims, " => ", e.out_dims, ")")
end

outputsize(e::Embedding) = (e.out_dims,)

"""
    PeriodicEmbedding(idxs, periods)

Create an embedding periodic in some dimensions with specified periods. Dimensions not in
`idxs` are passed through unchanged, but dimensions in `idxs` are moved to the end of the
output and replaced with their sines, followed by their cosines (scaled appropriately to
have the specified periods). This smooth embedding preserves phase information and enforces
periodicity.

## Arguments

  - `idxs`: Indices of the periodic inputs
  - `periods`: Periods of the periodic inputs, in the same order as in `idxs`

## Inputs

  - `x` must be an `AbstractArray` with `issubset(idxs, axes(x, 1))`
  - `st` must be a `NamedTuple` where `st.k = 2 ./ periods`, but on the same device as `x`

## Returns

  - `AbstractArray` of size `(size(x, 1) + length(idxs), ...)` where `...` are the other
    dimensions of `x`.
  - `st`, unchanged
"""
@concrete struct PeriodicEmbedding <:AbstractExplicitLayer
    idxs
    periods
end

Lux.initialstates(::AbstractRNG, p::PeriodicEmbedding) = (k = 2 ./ p.periods,)

@inline function (p::PeriodicEmbedding)(x::AbstractVector, ps, st::NamedTuple)
    return vec(first(p(reshape(x, :, 1), ps, st))), st
end

@inline function (p::PeriodicEmbedding)(x::AbstractMatrix, ps, st::NamedTuple)
    other_idxs = ChainRulesCore.@ignore_derivatives setdiff(axes(x, 1), p.idxs)
    return (
        vcat(
            x[other_idxs, :],
            sinpi.(st.k .* x[p.idxs, :]),
            cospi.(st.k .* x[p.idxs, :])
        ),
        st)
end

@inline function (p::PeriodicEmbedding)(x::AbstractArray, ps, st::NamedTuple)
    return reshape(first(p(reshape(x, size(x, 1), :), ps, st)), :, size(x)[2:end]...), st
end

function Base.show(io::IO, p::PeriodicEmbedding)
    return print(io, "PeriodicEmbedding(idxs = ", p.idxs, ", periods = ", p.periods, ")")
end
