function init_linear_bias(
    rng::AbstractRNG, init_bias::F, fan_in::IntegerType, bias_len::IntegerType
) where {F}
    if init_bias === nothing # Default from PyTorch
        bound = inv(sqrt(fan_in))
        y = rand32(rng, bias_len)
        @. y = (y - 0.5f0) * 2 * bound
        return y
    end
    return init_bias(rng, bias_len)
end

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
struct ReshapeLayer{N} <: AbstractLuxLayer
    dims::NTuple{N,Int}
end

outputsize(r::ReshapeLayer, _, ::AbstractRNG) = r.dims

@trace function (r::ReshapeLayer)(x::AbstractArray, _, st::NamedTuple)
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
@concrete struct ReverseSequence <: AbstractLuxLayer
    dim <: Union{Nothing,StaticInt}
end

ReverseSequence(dim) = ReverseSequence(static(dim))
ReverseSequence(; dim=nothing) = ReverseSequence(static(dim))

@trace function (r::ReverseSequence{Nothing})(x::AbstractArray, _, st::NamedTuple)
    return safe_reverse(x; dims=max(ndims(x) - 1, 1)), st
end

@trace function (r::ReverseSequence{StaticInt{1}})(x::AbstractVector, _, st::NamedTuple)
    return safe_reverse(x), st
end

@trace function (r::ReverseSequence{StaticInt{N}})(
    ::AbstractVector, _, st::NamedTuple
) where {N}
    throw(ArgumentError("Cannot specify a dimension ($(N) != 1) for AbstractVector"))
end

@trace function (r::ReverseSequence{StaticInt{N}})(
    x::AbstractArray, _, st::NamedTuple
) where {N}
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
@concrete struct FlattenLayer <: AbstractLuxLayer
    N <: Union{Nothing,StaticInt}
end

FlattenLayer(N) = FlattenLayer(static(N))
FlattenLayer(; N=nothing) = FlattenLayer(static(N))

@trace function (::FlattenLayer{Nothing})(
    x::AbstractArray{T,N}, _, st::NamedTuple
) where {T,N}
    return reshape(x, :, size(x, N)), st
end

@trace function (f::FlattenLayer)(x::AbstractArray{T,N}, _, st::NamedTuple) where {T,N}
    @assert f.N < N
    return reshape(x, :, size(x)[(f.N + 1):end]...), st
end

"""
    SelectDim(dim, i)

Return a view of all the data of the input `x` where the index for dimension `dim` equals
`i`. Equivalent to `view(x,:,:,...,i,:,:,...)` where `i` is any valid index for index slot `d`
(e.g. an integer or a unit range).  Note that it may be inefficient to use non-contiguous
views.

## Arguments

  - `dim`: Dimension for indexing
  - `i`: Index or indices for dimension `dim`

## Inputs

  - `x`: AbstractArray that can be indexed with `view(x,:,:,...,i,:,:,...)`

## Returns

  - `view(x,:,:,...,i,:,:,...)` where `i` is in position `d`
  - Empty `NamedTuple()`
"""
@concrete struct SelectDim <: AbstractLuxLayer
    dim <: StaticInt
    index <: Union{StaticInt,AbstractVector}
end

SelectDim(dim::Integer, index::Integer) = SelectDim(static(dim), static(index))
SelectDim(dim::Integer, index::AbstractVector) = SelectDim(static(dim), index)

@trace function (s::SelectDim{D,<:StaticInt})(x, _, st::NamedTuple) where {D}
    return selectdim(x, known(s.dim), known(s.index)), st
end
(s::SelectDim)(x, _, st::NamedTuple) = selectdim(x, known(s.dim), s.index), st

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
struct NoOpLayer <: AbstractLuxLayer end

(::NoOpLayer)(x, _, st::NamedTuple) = x, st

"""
    WrappedFunction(f)

Wraps a stateless and parameter less function. Might be used when a function is added to
`Chain`. For example, `Chain(x -> relu.(x))` would not work and the right thing to do would
be `Chain((x, ps, st) -> (relu.(x), st))`. An easier thing to do would be
`Chain(WrappedFunction(Base.Fix1(broadcast, relu)))`

## Arguments

  - `f`: Some function.

## Inputs

  - `x`: will be directly passed to `f`

## Returns

  - Output of `f(x)`
  - Empty `NamedTuple()`
"""
@concrete struct WrappedFunction <: AbstractLuxLayer
    func <: Function
end

(wf::WrappedFunction)(x, _ps, st::NamedTuple{}) = wf.func(x), st

Base.show(io::IO, w::WrappedFunction) = print(io, "WrappedFunction(", w.func, ")")

"""
    Dense(in_dims => out_dims, activation=identity; init_weight=nothing,
          init_bias=nothing, use_bias=True())

Create a traditional fully connected layer, whose forward pass is given by:
`y = activation.(weight * x .+ bias)`

## Arguments

  - `in_dims`: number of input dimensions
  - `out_dims`: number of output dimensions
  - `activation`: activation function

## Keyword Arguments

  - `init_weight`: initializer for the weight matrix
    (`weight = init_weight(rng, out_dims, in_dims)`). If `nothing`, then we use
    [`kaiming_uniform`](@ref) with gain computed on the basis of the activation
    function (taken from Pytorch
    [`nn.init.calculate_gain`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain)).
  - `init_bias`: initializer for the bias vector (ignored if `use_bias=false`). If
    `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(in_dims))`.
  - `use_bias`: Trainable bias can be disabled entirely by setting this to `false`

## Input

  - `x` must be an AbstractArray with `size(x, 1) == in_dims`

## Returns

  - AbstractArray with dimensions `(out_dims, ...)` where `...` are the dimensions of `x`
  - Empty `NamedTuple()`

## Parameters

  - `weight`: Weight Matrix of size `(out_dims, in_dims)`
  - `bias`: Bias of size `(out_dims, 1)` (present if `use_bias=true`)
"""
@concrete struct Dense <: AbstractLuxLayer
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

function Dense(mapping::Pair{<:IntegerType,<:IntegerType}, activation=identity; kwargs...)
    return Dense(first(mapping), last(mapping), activation; kwargs...)
end

function Dense(
    in_dims::IntegerType,
    out_dims::IntegerType,
    activation=identity;
    init_weight=nothing,
    init_bias=nothing,
    use_bias::BoolType=True(),
)
    return Dense(activation, in_dims, out_dims, init_weight, init_bias, static(use_bias))
end

function initialparameters(rng::AbstractRNG, d::Dense)
    weight = if d.init_weight === nothing
        kaiming_uniform(
            rng,
            Float32,
            d.out_dims,
            d.in_dims;
            gain=Utils.calculate_gain(d.activation, √5.0f0),
        )
    else
        d.init_weight(rng, d.out_dims, d.in_dims)
    end
    has_bias(d) || return (; weight)
    return (; weight, bias=init_linear_bias(rng, d.init_bias, d.in_dims, d.out_dims))
end

parameterlength(d::Dense) = d.out_dims * d.in_dims + has_bias(d) * d.out_dims
statelength(::Dense) = 0

function outputsize(d::Dense, x::AbstractArray, ::AbstractRNG)
    return (d.out_dims, size(x)[2:(end - 1)]...)
end

@trace function (d::Dense)(x::AbstractArray, ps, st::NamedTuple)
    y = match_eltype(d, ps, st, x)
    bias = safe_getproperty(ps, Val(:bias))
    σ = NNlib.fast_act(d.activation, x)
    z = matrix_to_array(
        fused_dense_bias_activation(σ, ps.weight, make_abstract_matrix(y), bias), y
    )
    return z, st
end

"""
    Scale(dims, activation=identity; init_weight=ones32, init_bias=zeros32, use_bias=True())

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
@concrete struct Scale{UB<:StaticBool} <: AbstractLuxLayer
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

function Scale(
    dims::Tuple{Vararg{IntegerType}},
    activation=identity;
    init_weight=glorot_uniform,
    init_bias=zeros32,
    use_bias::BoolType=True(),
)
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
    return (; weight=d.init_weight(rng, d.dims...))
end

parameterlength(d::Scale) = (1 + has_bias(d)) * prod(d.dims)
statelength(::Scale) = 0

outputsize(d::Scale, _, ::AbstractRNG) = d.dims

@trace function (d::Scale{False})(x::AbstractArray, ps, st::NamedTuple)
    y = match_eltype(d, ps, st, x)
    σ = NNlib.fast_act(d.activation, y)
    return @.(σ(y .* ps.weight)), st
end
@trace function (d::Scale{True})(x::AbstractArray, ps, st::NamedTuple)
    y = match_eltype(d, ps, st, x)
    σ = NNlib.fast_act(d.activation, y)
    return @.(σ(y * ps.weight + ps.bias)), st
end

"""
    Bilinear((in1_dims, in2_dims) => out, activation=identity; init_weight=nothing,
             init_bias=nothing, use_bias=True())
    Bilinear(in12_dims => out, activation=identity; init_weight=nothing,
             init_bias=nothing, use_bias=True())

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
    (`weight = init_weight(rng, out_dims, in1_dims, in2_dims)`). If `nothing`, then we
    use uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(in1_dims))`.
  - `init_bias`: initializer for the bias vector (ignored if `use_bias=false`). If
    `nothing`, then we use uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(in1_dims))`.
  - `use_bias`: Trainable bias can be disabled entirely by setting this to `false`

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
@concrete struct Bilinear <: AbstractLuxLayer
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

function Bilinear(
    (in12_dims, out)::Pair{<:IntegerType,<:IntegerType}, activation=identity; kwargs...
)
    return Bilinear((in12_dims, in12_dims) => out, activation; kwargs...)
end

function Bilinear(
    ((in1_dims, in2_dims), out)::Pair{<:Tuple,<:IntegerType},
    activation=identity;
    init_weight=nothing,
    init_bias=nothing,
    use_bias::BoolType=True(),
)
    return Bilinear(
        activation, in1_dims, in2_dims, out, init_weight, init_bias, static(use_bias)
    )
end

function initialparameters(rng::AbstractRNG, b::Bilinear)
    weight = if b.init_weight === nothing
        bound = inv(sqrt(b.in1_dims))
        y = randn32(rng, b.out_dims, b.in1_dims, b.in2_dims)
        @. y = (y - 0.5f0) * 2 * bound
        y
    else
        b.init_weight(rng, b.out_dims, b.in1_dims, b.in2_dims)
    end
    has_bias(b) || return (; weight)
    return (; weight, bias=init_linear_bias(rng, b.init_bias, b.in1_dims, b.out_dims))
end

function parameterlength(b::Bilinear)
    return b.out_dims * b.in1_dims * b.in2_dims + has_bias(b) * b.out_dims
end
statelength(::Bilinear) = 0

outputsize(b::Bilinear, _, ::AbstractRNG) = (b.out_dims,)

@trace function (b::Bilinear)(
    (x, y)::Tuple{<:AbstractVecOrMat,<:AbstractVecOrMat}, ps, st::NamedTuple
)
    s₁, s₂, s₃ = size(ps.weight)
    @assert s₂ == size(x, 1) && s₃ == size(y, 1)
    @assert size(x, 2) == size(y, 2)

    Wy = reshape(reshape(ps.weight, (:, s₃)) * y, (s₁, s₂, :))
    Wyx = reshape(batched_matmul(Wy, reshape(x, (s₂, 1, :))), (s₁, :))

    σ = NNlib.fast_act(b.activation, Wyx)
    return bias_activation!!(σ, Wyx, safe_getproperty(ps, Val(:bias))), st
end

@trace function (b::Bilinear)(
    (x, y)::Tuple{<:AbstractArray,<:AbstractArray}, ps, st::NamedTuple
)
    @assert size(x)[2:end] == size(y)[2:end]

    s₁, s₂, s₃ = size(ps.weight)
    x′ = reshape(x, s₂, :)
    y′ = reshape(y, s₃, :)

    z, stₙ = b((x′, y′), ps, st)

    return reshape(z, s₁, size(x)[2:end]...), stₙ
end

(b::Bilinear)(x::AbstractArray, ps, st::NamedTuple) = b((x, x), ps, st)

"""
    AlternatePrecision{T}(layer)
    AlternatePrecision(::Type{T}, layer)

This layer is used to convert the input to a different precision (`T`), execute the layer,
and then convert the output back to the original precision.

## Arguments

  - `T`: The eltype of the input to the layer
  - `layer`: The layer to execute

## Inputs

  - `x`: AbstractArray

## Returns

  - `y`: Output of the layer
  - State of the output
"""
@concrete struct AlternatePrecision{T} <: AbstractLuxWrapperLayer{:layer}
    layer
end

AlternatePrecision(::Type{T}, layer) where {T} = AlternatePrecision{T}(layer)

LuxCore.display_name(::AlternatePrecision{T}) where {T} = "AlternatePrecision{$T}"

@trace function (model::AlternatePrecision{T})(x::AbstractArray{T2}, ps, st) where {T,T2}
    y, stₙ = model.layer(T.(x), ps, st)
    return T2.(y), stₙ
end
