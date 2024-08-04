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

## Example

```jldoctest
julia> model = ReshapeLayer((2, 2))
ReshapeLayer(output_dims = (2, 2, :))

julia> rng = Random.default_rng();
       Random.seed!(rng, 0);
       ps, st = Lux.setup(rng, model);
       x = randn(rng, Float32, (4, 1, 3));

julia> y, st_new = model(x, ps, st);
       size(y)
(2, 2, 3)
```
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
    ReverseSequence(dim = nothing)

Reverse the specified dimension `dims` of the passed array

## Arguments

  - `dim`: Dimension that need to be reversed. If `nothing`, for AbstractVector{T}
    it reverses itself (dimension 1), for other arrays, reverse the dimension `ndims(x) - 1`.

## Inputs

  - `x`: AbstractArray.

## Returns

  - AbstractArray with the same dimensions as the input
  - Empty `NamedTuple()`

## Example

```jldoctest
julia> model = ReverseSequence()
ReverseSequence{Nothing}(nothing)

julia> rng = Random.default_rng();
       Random.seed!(rng, 0);
       ps, st = Lux.setup(rng, model);
       x = [1.0, 2.0, 3.0];

julia> y, st_new = model(x, ps, st)
([3.0, 2.0, 1.0], NamedTuple())
```
"""
@kwdef struct ReverseSequence{D <: Union{Int, Nothing}} <: AbstractExplicitLayer
    dim::D = nothing
end

@inline function (r::ReverseSequence{Nothing})(
        x::AbstractVector{T}, ps, st::NamedTuple) where {T}
    return __reverse(x), st
end

@inline function (r::ReverseSequence{Nothing})(
        x::AbstractArray{T, N}, ps, st::NamedTuple) where {T, N}
    return __reverse(x; dims=ndims(x) - 1), st
end

@inline function (r::ReverseSequence)(x::AbstractVector{T}, ps, st::NamedTuple) where {T}
    r.dim == 1 && return __reverse(x), st
    throw(ArgumentError("Cannot specify a dimension other than 1 for AbstractVector{T}"))
end

@inline function (r::ReverseSequence)(
        x::AbstractArray{T, N}, ps, st::NamedTuple) where {T, N}
    return __reverse(x; dims=r.dim), st
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

## Example

```jldoctest
julia> model = FlattenLayer()
FlattenLayer{Nothing}(nothing)

julia> rng = Random.default_rng();
       Random.seed!(rng, 0);
       ps, st = Lux.setup(rng, model);
       x = randn(rng, Float32, (2, 2, 2, 2));

julia> y, st_new = model(x, ps, st);
       size(y)
(8, 2)
```
"""
struct FlattenLayer{NT <: Union{Nothing, Int}} <: AbstractExplicitLayer
    N::NT
end

FlattenLayer(; N=nothing) = FlattenLayer(N)

@inline function (::FlattenLayer{Nothing})(
        x::AbstractArray{T, N}, ps, st::NamedTuple) where {T, N}
    return reshape(x, :, size(x, N)), st
end

@inline function (f::FlattenLayer)(x::AbstractArray{T, N}, ps, st::NamedTuple) where {T, N}
    @argcheck f.N < N
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

function Base.show(io::IO, ::SelectDim{dim, index}) where {dim, index}
    return print(io, "SelectDim(", dim, ", ", index, ")")
end

"""
    NoOpLayer()

As the name suggests does nothing but allows pretty printing of layers. Whatever input is
passed is returned.

# Example

```jldoctest
julia> model = NoOpLayer()
NoOpLayer()

julia> rng = Random.default_rng();
       Random.seed!(rng, 0);
       ps, st = Lux.setup(rng, model);
       x = 1
1

julia> y, st_new = model(x, ps, st)
(1, NamedTuple())
```
"""
struct NoOpLayer <: AbstractExplicitLayer end

@inline (noop::NoOpLayer)(x, ps, st::NamedTuple) = x, st

"""
    WrappedFunction{DC}(f)
    WrappedFunction(f) -> WrappedFunction{:direct_call}(f)

Wraps a stateless and parameter less function. Might be used when a function is added to
`Chain`. For example, `Chain(x -> relu.(x))` would not work and the right thing to do would
be `Chain((x, ps, st) -> (relu.(x), st))`. An easier thing to do would be
`Chain(WrappedFunction(Base.Fix1(broadcast, relu)))`

## Arguments

  - `DC`: If `:runtime_check`, then we check if the function can be called with the input
    `x`, `ps`, and `st` using `hasmethod`. If `:direct_call`, we call `f(x)` directly.
    For all other values, we call `f(x, ps, st)` which must return a tuple. **(In future
    versions, we will default to `:runtime_check`)**
  - `f`: Some function.

## Inputs

  - `x`: s.t `hasmethod(f, (typeof(x),))` is `true`

## Returns

  - Output of `f(x)`
  - Empty `NamedTuple()`
"""
@concrete struct WrappedFunction{DC} <: AbstractExplicitLayer
    func
end

function WrappedFunction(f::F) where {F}
    # Not a depwarn but helpful to call this
    Base.depwarn("The current default of `:direct_call` will be replaced with \
                  `:runtime_check` from v0.6). Please make sure that the assumptions of \
                  this function are correct or specify `WrappedFunction{:direct_call}(f)`",
        :WrappedFunction)
    return WrappedFunction{:direct_call}(f)
end

function (wf::WrappedFunction{:direct_call})(x, ps, st::NamedTuple)
    return __maybe_direct_call(wf.func, x, ps, st, Val(true))
end

function (wf::WrappedFunction)(x, ps, st::NamedTuple)
    return __maybe_direct_call(wf.func, x, ps, st, Val(false))
end

function (wf::WrappedFunction{:runtime_check})(x, ps, st::NamedTuple)
    return __maybe_direct_call(
        wf.func, x, ps, st, Val(!hasmethod(wf.func, (typeof(x), typeof(ps), typeof(st)))))
end

@inline __maybe_direct_call(f, x, ps, st, ::Val{false}) = f(x, ps, st)
@inline __maybe_direct_call(f, x, ps, st, ::Val{true}) = f(x), st

function Base.show(io::IO, w::WrappedFunction{T}) where {T}
    print(io, "WrappedFunction{$(Meta.quot(T))}(")
    show(io, w.func)
    print(io, ")")
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
    use_bias || print(io, ", use_bias=false")
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

@inline function (d::Dense)(x::AbstractArray, ps, st::NamedTuple)
    return reshape(first(d(reshape(x, size(x, 1), :), ps, st)), :, size(x)[2:end]...), st
end

@inline function (d::Dense)(x::AbstractMatrix, ps, st::NamedTuple)
    y = match_eltype(d, ps, st, x)
    return (
        fused_dense_bias_activation(
            d.activation, ps.weight, y, _vec(_getproperty(ps, Val(:bias)))),
        st)
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

function Base.show(io::IO, d::Scale{use_bias}) where {use_bias}
    print(io, "Scale($(d.dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    use_bias || print(io, ", use_bias=false")
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

function (d::Scale{false})(x::AbstractArray, ps, st::NamedTuple)
    y = match_eltype(d, ps, st, x)
    return @.(d.activation(y .* ps.weight)), st
end
function (d::Scale{true})(x::AbstractArray, ps, st::NamedTuple)
    y = match_eltype(d, ps, st, x)
    return @.(d.activation(y * ps.weight + ps.bias)), st
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
    use_bias || print(io, ", use_bias=false")
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
            bias=b.init_bias(rng, b.out_dims, 1)) # TODO: In v1.0 make it a vector
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
    @argcheck d_x == size(x, 1) && d_y == size(y, 1)
    @argcheck size(x, 2) == size(y, 2)

    Wy = reshape(reshape(ps.weight, (:, d_y)) * y, (d_z, d_x, :))
    Wyx = reshape(batched_matmul(Wy, reshape(x, (d_x, 1, :))), (d_z, :))

    return bias_activation!!(b.activation, Wyx, _vec(_getproperty(ps, Val(:bias)))), st
end

function (b::Bilinear)((x, y)::Tuple{<:AbstractArray, <:AbstractArray}, ps, st::NamedTuple)
    @argcheck size(x)[2:end] == size(y)[2:end]

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
`in_dims`. When the vocabulary is multi-dimensional, the input is expected to be a tuple
of Cartesian indices.

This layer is often used to store word embeddings and retrieve them using indices.

!!! warning

    Unlike `Flux.Embedding`, this layer does not support using `OneHotArray` as an input.

## Arguments

  - `in_dims`: number(s) of input dimensions
  - `out_dims`: number of output dimensions

## Keyword Arguments

  - `init_weight`: initializer for the weight matrix
    (`weight = init_weight(rng, out_dims, in_dims...)`)

## Input

  - Integer OR
  - Abstract Vector of Integers OR
  - Abstract Array of Integers OR
  - Tuple of Integers OR
  - Tuple of Abstract Vectors of Integers OR
  - Tuple of Abstract Arrays of Integers

## Returns

  - Returns the embedding corresponding to each index in the input. For an N dimensional
    input, an N + 1 dimensional output is returned.
  - Empty `NamedTuple()`
"""
@concrete struct Embedding <: AbstractExplicitLayer
    in_dims
    out_dims::Int
    init_weight
end

function Embedding(
        (in_dims, out_dims)::Pair{<:Union{Integer, NTuple{<:Any, <:Integer}}, <:Integer};
        init_weight=randn32)
    return Embedding(in_dims, out_dims, init_weight)
end

function initialparameters(rng::AbstractRNG, e::Embedding)
    return (weight=e.init_weight(rng, e.out_dims, e.in_dims...),)
end

(e::Embedding)(x::Integer, ps, st::NamedTuple) = view(ps.weight, :, x), st
function (e::Embedding)(x::AbstractVector{<:Integer}, ps, st::NamedTuple)
    return NNlib.gather(ps.weight, x), st
end
function (e::Embedding)(x::AbstractArray{<:Integer}, ps, st::NamedTuple)
    return reshape(e(vec(x), ps, st)[1], :, size(x)...), st
end
function (e::Embedding)(x::NTuple{<:Any, <:Integer}, ps, st::NamedTuple)
    view(ps.weight, :, x...), st
end
function (e::Embedding)(x::NTuple{<:Any, <:AbstractVector{<:Integer}}, ps, st::NamedTuple)
    sizes = size.(x)
    @argcheck allequal(sizes) DimensionMismatch("Input vectors must have the same shape")
    return NNlib.gather(ps.weight, x...), st
end
function (e::Embedding)(x::NTuple{<:Any, <:AbstractArray{<:Integer}}, ps, st::NamedTuple)
    sizes = size.(x)
    @argcheck allequal(sizes) DimensionMismatch("Input arrays must have the same shape")
    return reshape(e(vec.(x), ps, st)[1], :, first(sizes)...), st
end
function (e::Embedding)(x::Tuple{}, ps, st::NamedTuple)
    throw(ArgumentError("Input tuple must contain at least one element"))
end

function Base.show(io::IO, e::Embedding)
    return print(io, "Embedding(", e.in_dims, " => ", e.out_dims, ")")
end

outputsize(e::Embedding) = (e.out_dims,)

"""
    PeriodicEmbedding(idxs, periods)

Create an embedding periodic in some inputs with specified periods. Input indices not in
`idxs` are passed through unchanged, but inputs in `idxs` are moved to the end of the
output and replaced with their sines, followed by their cosines (scaled appropriately to
have the specified periods). This smooth embedding preserves phase information and enforces
periodicity.

For example, `layer = PeriodicEmbedding([2, 3], [3.0, 1.0])` will create a layer periodic in
the second input with period 3.0 and periodic in the third input with period 1.0. In this
case, `layer([a, b, c, d], st) == ([a, d, sinpi(2 / 3.0 * b), sinpi(2 / 1.0 * c), cospi(2 / 3.0 * b), cospi(2 / 1.0 * c)], st)`.

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

## Example

```jldoctest
julia> layer = PeriodicEmbedding([2], [4.0])
PeriodicEmbedding([2], [4.0])

julia> using Random;
       rng = Random.seed!(123);

julia> ps, st = Lux.setup(rng, layer)
(NamedTuple(), (k = [0.5],))

julia> all(layer([1.1, 2.2, 3.3], ps, st)[1] .==
           [1.1, 3.3, sinpi(2 / 4.0 * 2.2), cospi(2 / 4.0 * 2.2)])
true
```
"""
@concrete struct PeriodicEmbedding <: AbstractExplicitLayer
    idxs
    periods
end

initialstates(::AbstractRNG, p::PeriodicEmbedding) = (k=2 ./ p.periods,)

@inline function (p::PeriodicEmbedding)(x::AbstractVector, ps, st::NamedTuple)
    return vec(first(p(reshape(x, :, 1), ps, st))), st
end

@inline function (p::PeriodicEmbedding)(x::AbstractMatrix, ps, st::NamedTuple)
    other_idxs = CRC.@ignore_derivatives setdiff(axes(x, 1), p.idxs)
    return (
        vcat(x[other_idxs, :], sinpi.(st.k .* x[p.idxs, :]), cospi.(st.k .* x[p.idxs, :])),
        st)
end

@inline function (p::PeriodicEmbedding)(x::AbstractArray, ps, st::NamedTuple)
    return reshape(first(p(reshape(x, size(x, 1), :), ps, st)), :, size(x)[2:end]...), st
end

function Base.show(io::IO, p::PeriodicEmbedding)
    return print(io, "PeriodicEmbedding(", p.idxs, ", ", p.periods, ")")
end
