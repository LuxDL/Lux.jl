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

function (r::ReshapeLayer)(x::AbstractArray, _, st::NamedTuple)
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
    it reverses itself (dimension 1), for other arrays, reverse the dimension
    `ndims(x) - 1`.

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
@concrete struct ReverseSequence <: AbstractExplicitLayer
    dim <: Union{Nothing, StaticInt}
end

ReverseSequence(dim) = ReverseSequence(static(dim))
ReverseSequence(; dim=nothing) = ReverseSequence(static(dim))

function (r::ReverseSequence{Nothing})(x::AbstractArray, _, st::NamedTuple)
    return safe_reverse(x; dims=max(ndims(x) - 1, 1)), st
end

function (r::ReverseSequence{StaticInt{1}})(x::AbstractVector, _, st::NamedTuple)
    return safe_reverse(x), st
end

function (r::ReverseSequence{StaticInt{N}})(::AbstractVector, _, st::NamedTuple) where {N}
    throw(ArgumentError("Cannot specify a dimension ($(N) != 1) for AbstractVector"))
end

function (r::ReverseSequence{StaticInt{N}})(x::AbstractArray, _, st::NamedTuple) where {N}
    return safe_reverse(x; dims=N), st
end

"""
    FlattenLayer(; N = nothing)

Flattens the passed array into a matrix.

## Keyword Arguments

  - `N`: Flatten the first `N` dimensions of the input array. If `nothing`, then all
    dimensions (except the last) are flattened. Note that the batch dimension is never
    flattened.

## Inputs

  - `x`: AbstractArray

## Returns

  - AbstractMatrix of size `(:, size(x, ndims(x)))` if `N` is `nothing` else the first
    `N` dimensions of the input array are flattened.
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
@concrete struct FlattenLayer <: AbstractExplicitLayer
    N <: Union{Nothing, StaticInt}
end

FlattenLayer(N) = FlattenLayer(static(N))
FlattenLayer(; N=nothing) = FlattenLayer(static(N))

function (::FlattenLayer{Nothing})(x::AbstractArray{T, N}, _, st::NamedTuple) where {T, N}
    return reshape(x, :, size(x, N)), st
end

function (f::FlattenLayer)(x::AbstractArray{T, N}, _, st::NamedTuple) where {T, N}
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
@concrete struct SelectDim <: AbstractExplicitLayer
    dim <: StaticInt
    index <: StaticInt
end

SelectDim(dim, index) = SelectDim(static(dim), static(index))

(s::SelectDim)(x, _, st::NamedTuple) = selectdim(x, known(s.dim), known(s.index)), st

function Base.show(io::IO, s::SelectDim)
    return print(io, "SelectDim(dim = ", s.dim, ", index = ", s.index, ")")
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

(noop::NoOpLayer)(x, _, st::NamedTuple) = x, st

"""
    WrappedFunction{DC}(f)
    WrappedFunction(f) -> WrappedFunction{:runtime_check}(f)

Wraps a stateless and parameter less function. Might be used when a function is added to
`Chain`. For example, `Chain(x -> relu.(x))` would not work and the right thing to do would
be `Chain((x, ps, st) -> (relu.(x), st))`. An easier thing to do would be
`Chain(WrappedFunction(Base.Fix1(broadcast, relu)))`

## Arguments

  - `DC`: If `:runtime_check`, then we check if the function can be called with the input
    `x`, `ps`, and `st` using `hasmethod`. If `:direct_call`, we call `f(x)` directly.
    For all other values, we call `f(x, ps, st)` which must return a tuple.
  - `f`: Some function.

## Inputs

  - `x`: s.t `hasmethod(f, (typeof(x),))` is `true` if :direct_call else
    `hasmethod(f, (typeof(x), NamedTuple, NamedTuple))` is `true`

## Returns

  - Output of `f(x)`
  - Empty `NamedTuple()`
"""
struct WrappedFunction{DC, F} <: AbstractExplicitLayer
    call_mode::StaticSymbol{DC}
    func::F
end

function WrappedFunction{call_mode}(f::F) where {call_mode, F}
    return WrappedFunction(static(call_mode), f)
end

WrappedFunction(f::F) where {F} = WrappedFunction{:runtime_check}(f)

function (wf::WrappedFunction{:direct_call})(x, ps, st::NamedTuple)
    return wrapped_function_call(wf.func, x, ps, st, True())
end

function (wf::WrappedFunction)(x, ps, st::NamedTuple)
    return wrapped_function_call(wf.func, x, ps, st, False())
end

function (wf::WrappedFunction{:runtime_check})(x, ps, st::NamedTuple)
    return wrapped_function_call(wf.func, x, ps, st,
        static(!hasmethod(wf.func, (typeof(x), typeof(ps), typeof(st)))))
end

wrapped_function_call(f, x, ps, st, ::False) = f(x, ps, st)
wrapped_function_call(f, x, _, st, ::True) = f(x), st

function Base.show(io::IO, w::WrappedFunction{T}) where {T}
    print(io, "WrappedFunction(", static(w.call_mode), ", ")
    show(io, w.func)
    print(io, ")")
end

"""
    Dense(in_dims => out_dims, activation=identity; init_weight=glorot_uniform,
          init_bias=zeros32, use_bias=True(), allow_fast_activation=True())

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
@concrete struct Dense <: AbstractExplicitLayer
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_weight
    init_bias
    use_bias <: StaticBool
end

function Base.show(io::IO, d::Dense)
    print(io, "Dense($(d.in_dims) => $(d.out_dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    has_bias(d) || print(io, ", use_bias=false")
    return print(io, ")")
end

function Dense(mapping::Pair{<:IntegerType, <:IntegerType}, activation=identity; kwargs...)
    return Dense(first(mapping), last(mapping), activation; kwargs...)
end

function Dense(in_dims::IntegerType, out_dims::IntegerType, activation=identity;
        init_weight=glorot_uniform, init_bias=zeros32,
        use_bias::BoolType=True(), allow_fast_activation::BoolType=True())
    activation = dynamic(allow_fast_activation) ? NNlib.fast_act(activation) : activation
    return Dense(activation, in_dims, out_dims, init_weight, init_bias, static(use_bias))
end

function initialparameters(rng::AbstractRNG, d::Dense)
    if has_bias(d)
        return (weight=d.init_weight(rng, d.out_dims, d.in_dims),
            bias=d.init_bias(rng, d.out_dims, 1)) #TODO: In v1 make it a vector
    else
        return (weight=d.init_weight(rng, d.out_dims, d.in_dims),)
    end
end

parameterlength(d::Dense) = d.out_dims * d.in_dims + has_bias(d) * d.out_dims
statelength(d::Dense) = 0

outputsize(d::Dense) = (d.out_dims,)

function (d::Dense)(x::AbstractArray, ps, st::NamedTuple)
    y = match_eltype(d, ps, st, x)
    bias = safe_vec(safe_getproperty(ps, Val(:bias)))
    z = matrix_to_array(
        fused_dense_bias_activation(d.activation, ps.weight, make_abstract_matrix(y), bias),
        y)
    return z, st
end

"""
    Scale(dims, activation=identity; init_weight=ones32, init_bias=zeros32, use_bias=True(),
          allow_fast_activation=True())

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
@concrete struct Scale{UB <: StaticBool} <: AbstractExplicitLayer
    activation
    dims <: Tuple{Vararg{IntegerType}}
    init_weight
    init_bias
    use_bias::UB
end

function Base.show(io::IO, d::Scale)
    print(io, "Scale($(d.dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    has_bias(d) || print(io, ", use_bias=false")
    return print(io, ")")
end

function Scale(dims::Tuple{Vararg{IntegerType}}, activation=identity;
        init_weight=glorot_uniform, init_bias=zeros32,
        use_bias::BoolType=True(), allow_fast_activation::BoolType=True())
    activation = dynamic(allow_fast_activation) ? NNlib.fast_act(activation) : activation
    return Scale(activation, dims, init_weight, init_bias, static(use_bias))
end

function Scale(s1::IntegerType, s23::IntegerType...; _act=identity, kwargs...)
    return Scale(tuple(s1, s23...), _act; kwargs...)
end
function Scale(size_act...; kwargs...)
    return Scale(size_act[1:(end - 1)]...; _act=size_act[end], kwargs...)
end

function initialparameters(rng::AbstractRNG, d::Scale)
    if has_bias(d)
        return (; weight=d.init_weight(rng, d.dims...), bias=d.init_bias(rng, d.dims...))
    end
    return (; weight=d.init_weight(rng, d.dims...),)
end

parameterlength(d::Scale) = (1 + has_bias(d)) * prod(d.dims)
statelength(d::Scale) = 0

outputsize(d::Scale) = d.dims

function (d::Scale{False})(x::AbstractArray, ps, st::NamedTuple)
    y = match_eltype(d, ps, st, x)
    return @.(d.activation(y .* ps.weight)), st
end
function (d::Scale{True})(x::AbstractArray, ps, st::NamedTuple)
    y = match_eltype(d, ps, st, x)
    return @.(d.activation(y * ps.weight + ps.bias)), st
end

"""
    Bilinear((in1_dims, in2_dims) => out, activation=identity; init_weight=glorot_uniform,
             init_bias=zeros32, use_bias=True(), allow_fast_activation=True())
    Bilinear(in12_dims => out, activation=identity; init_weight=glorot_uniform,
             init_bias=zeros32, use_bias=True(), allow_fast_activation=True())

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
@concrete struct Bilinear <: AbstractExplicitLayer
    activation
    in1_dims <: IntegerType
    in2_dims <: IntegerType
    out_dims <: IntegerType
    init_weight
    init_bias
    use_bias <: StaticBool
end

function Base.show(io::IO, b::Bilinear)
    print(io, "Bilinear(($(b.in1_dims), $(b.in2_dims)) => $(b.out_dims)")
    (b.activation == identity) || print(io, ", $(b.activation)")
    has_bias(b) || print(io, ", use_bias=false")
    return print(io, ")")
end

function Bilinear((in12_dims, out)::Pair{<:IntegerType, <:IntegerType},
        activation=identity; kwargs...)
    return Bilinear((in12_dims, in12_dims) => out, activation; kwargs...)
end

function Bilinear(((in1_dims, in2_dims), out)::Pair{<:Tuple, <:IntegerType},
        activation=identity; init_weight=glorot_uniform, init_bias=zeros32,
        use_bias::BoolType=True(), allow_fast_activation::BoolType=True())
    activation = dynamic(allow_fast_activation) ? NNlib.fast_act(activation) : activation
    return Bilinear(
        activation, in1_dims, in2_dims, out, init_weight, init_bias, static(use_bias))
end

function initialparameters(rng::AbstractRNG, b::Bilinear)
    if has_bias(b)
        return (weight=b.init_weight(rng, b.out_dims, b.in1_dims, b.in2_dims),
            bias=b.init_bias(rng, b.out_dims, 1)) # TODO: In v1.0 make it a vector
    else
        return (weight=b.init_weight(rng, b.out_dims, b.in1_dims, b.in2_dims),)
    end
end

function parameterlength(b::Bilinear)
    return b.out_dims * b.in1_dims * b.in2_dims + has_bias(b) * b.out_dims
end
statelength(b::Bilinear) = 0

outputsize(b::Bilinear) = (b.out_dims,)

function (b::Bilinear)(
        (x, y)::Tuple{<:AbstractVecOrMat, <:AbstractVecOrMat}, ps, st::NamedTuple)
    s₁, s₂, s₃ = size(ps.weight)
    @argcheck s₂ == size(x, 1) && s₃ == size(y, 1)
    @argcheck size(x, 2) == size(y, 2)

    Wy = reshape(reshape(ps.weight, (:, s₃)) * y, (s₁, s₂, :))
    Wyx = reshape(batched_matmul(Wy, reshape(x, (s₂, 1, :))), (s₁, :))

    bias = safe_vec(safe_getproperty(ps, Val(:bias)))
    return (bias_activation!!(b.activation, Wyx, bias), st)
end

function (b::Bilinear)((x, y)::Tuple{<:AbstractArray, <:AbstractArray}, ps, st::NamedTuple)
    @argcheck size(x)[2:end] == size(y)[2:end]

    s₁, s₂, s₃ = size(ps.weight)
    x′ = reshape(x, s₂, :)
    y′ = reshape(y, s₃, :)

    z, stₙ = b((x′, y′), ps, st)

    return reshape(z, s₁, size(x)[2:end]...), stₙ
end

(b::Bilinear)(x::AbstractArray, ps, st::NamedTuple) = b((x, x), ps, st)

"""
    Embedding(in_dims => out_dims; init_weight=randn32)

A lookup table that stores embeddings of dimension `out_dims` for a vocabulary of size
`in_dims`. When the vocabulary is multi-dimensional, the input is expected to be a tuple
of Cartesian indices.

This layer is often used to store word embeddings and retrieve them using indices.

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
    in_dims <: Union{IntegerType, Tuple{Vararg{IntegerType}}}
    out_dims <: IntegerType
    init_weight
end

function Embedding((in_dims, out_dims)::Pair; init_weight=randn32)
    return Embedding(in_dims, out_dims, init_weight)
end

function initialparameters(rng::AbstractRNG, e::Embedding)
    return (weight=e.init_weight(rng, e.out_dims, e.in_dims...),)
end

function Base.show(io::IO, e::Embedding)
    return print(io, "Embedding(", e.in_dims, " => ", e.out_dims, ")")
end

outputsize(e::Embedding) = (e.out_dims,)

(e::Embedding)(x::Integer, ps, st::NamedTuple) = view(ps.weight, :, x), st
function (e::Embedding)(x::AbstractVector{<:Integer}, ps, st::NamedTuple)
    return NNlib.gather(ps.weight, x), st
end
function (e::Embedding)(x::AbstractArray{<:Integer}, ps, st::NamedTuple)
    y, stₙ = e(vec(x), ps, st)
    return reshape(y, :, size(x)...), stₙ
end
function (e::Embedding)(x::NTuple{<:Any, <:Integer}, ps, st::NamedTuple)
    return view(ps.weight, :, x...), st
end
function (e::Embedding)(x::NTuple{<:Any, <:AbstractVector{<:Integer}}, ps, st::NamedTuple)
    @argcheck allequal(size, x) DimensionMismatch("Input vectors must have the same shape")
    return NNlib.gather(ps.weight, x...), st
end
function (e::Embedding)(x::NTuple{<:Any, <:AbstractArray{<:Integer}}, ps, st::NamedTuple)
    @argcheck allequal(size, x) DimensionMismatch("Input arrays must have the same shape")
    y, stₙ = e(vec.(x), ps, st)
    return reshape(y, :, size(first(x))...), stₙ
end
function (e::Embedding)(::Tuple{}, _, ::NamedTuple)
    throw(ArgumentError("Input tuple must contain at least one element"))
end

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

!!! danger "Deprecation Notice"

    This layer is deprecated and will be removed in v1. Please use the version in
    [`Boltz.jl`](https://github.com/LuxDL/Boltz.jl) instead.

# Extended Help

## Inputs

  - `x` must be an `AbstractArray` with `issubset(idxs, axes(x, 1))`
  - `st` must be a `NamedTuple` where `st.k = 2 ./ periods`, but on the same device as `x`

## Returns

  - `AbstractArray` of size `(size(x, 1) + length(idxs), ...)` where `...` are the other
    dimensions of `x`.
  - `st`, unchanged
"""
struct PeriodicEmbedding{I, P} <: AbstractExplicitLayer
    idxs::I
    periods::P

    function PeriodicEmbedding(idxs::I, periods::P) where {I, P}
        Base.depwarn("`PeriodicEmbedding` is deprecated and will be removed in v1. Please \
                      use the corresponding version in `Boltz.jl` instead.",
            :PeriodicEmbedding)
        return new{I, P}(idxs, periods)
    end
end

initialstates(::AbstractRNG, p::PeriodicEmbedding) = (k=2 ./ p.periods,)

function (p::PeriodicEmbedding)(x::AbstractVector, ps, st::NamedTuple)
    return vec(first(p(reshape(x, :, 1), ps, st))), st
end

function (p::PeriodicEmbedding)(x::AbstractMatrix, ps, st::NamedTuple)
    other_idxs = CRC.@ignore_derivatives setdiff(axes(x, 1), p.idxs)
    return (
        vcat(x[other_idxs, :], sinpi.(st.k .* x[p.idxs, :]), cospi.(st.k .* x[p.idxs, :])),
        st)
end

function (p::PeriodicEmbedding)(x::AbstractArray, ps, st::NamedTuple)
    return reshape(first(p(reshape(x, size(x, 1), :), ps, st)), :, size(x)[2:end]...), st
end

function Base.show(io::IO, p::PeriodicEmbedding)
    return print(io, "PeriodicEmbedding(", p.idxs, ", ", p.periods, ")")
end
