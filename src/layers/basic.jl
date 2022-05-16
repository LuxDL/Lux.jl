"""
    ReshapeLayer(dims)

Reshapes the passed array to have a size of `(dims..., :)`
"""
struct ReshapeLayer{N} <: AbstractExplicitLayer
    dims::NTuple{N,Int}
end

@inline function (r::ReshapeLayer)(x::AbstractArray, ps, st::NamedTuple)
    return reshape(x, r.dims..., :), st
end

"""
    FlattenLayer()

Flattens the passed array into a matrix.
"""
struct FlattenLayer <: AbstractExplicitLayer end

@inline function (f::FlattenLayer)(x::AbstractArray{T,N}, ps, st::NamedTuple) where {T,N}
    return reshape(x, :, size(x, N)), st
end

"""
    SelectDim(dim, i)

See the documentation for `selectdim` for more information.
"""
struct SelectDim{I} <: AbstractExplicitLayer
    dim::Int
    i::I
end

@inline (s::SelectDim)(x, ps, st::NamedTuple) = selectdim(x, s.dim, s.i), st

"""
    NoOpLayer()

As the name suggests does nothing but allows pretty printing of layers.
"""
struct NoOpLayer <: AbstractExplicitLayer end

@inline (noop::NoOpLayer)(x, ps, st::NamedTuple) = x, st

"""
    WrappedFunction(f)

Wraps a stateless and parameter less function. Might be used when a function is
added to `Chain`. For example, `Chain(x -> relu.(x))` would not work and the
right thing to do would be `Chain((x, ps, st) -> (relu.(x), st))`. An easier thing
to do would be `Chain(WrappedFunction(Base.Fix1(broadcast, relu)))`
"""
struct WrappedFunction{F} <: AbstractExplicitLayer
    func::F
end

(wf::WrappedFunction)(x, ps, st::NamedTuple) = wf.func(x), st

function Base.show(io::IO, w::WrappedFunction)
    return print(io, "WrappedFunction(", w.func, ")")
end

"""
    ActivationFunction(f)

Broadcast `f` on the input but fallback to CUDNN for Backward Pass
"""
struct ActivationFunction{F} <: AbstractExplicitLayer
    func::F
end

(af::ActivationFunction)(x, ps, st::NamedTuple) = applyactivation(af.func, x), st

function Base.show(io::IO, af::ActivationFunction)
    return print(io, "ActivationFunction(", af.func, ")")
end

"""
    SkipConnection(layer, connection)

Create a skip connection which consists of a layer or `Chain` of consecutive layers and a shortcut connection linking the block's input to the output through a user-supplied 2-argument callable. The first argument to the callable will be propagated through the given `layer` while the second is the unchanged, "skipped" input.

The simplest "ResNet"-type connection is just `SkipConnection(layer, +)`.
"""
struct SkipConnection{T<:AbstractExplicitLayer,F} <: AbstractExplicitContainerLayer{(:layers,)}
    layers::T
    connection::F
end

@inline function (skip::SkipConnection)(input, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple)
    mx, st = skip.layers(input, ps, st)
    return skip.connection(mx, input), st
end

"""
    Parallel(connection, layers...)

Behaves differently on different input types:
* If `x` is a Tuple then each element is passed to each layer
* Otherwise, `x` is directly passed to all layers
"""
struct Parallel{F,T<:NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    connection::F
    layers::T
end

function Parallel(connection, layers...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return Parallel(connection, NamedTuple{names}(layers))
end

function Parallel(connection; kw...)
    layers = NamedTuple(kw)
    if :layers in Base.keys(layers) || :connection in Base.keys(layers)
        throw(ArgumentError("a Parallel layer cannot have a named sub-layer called `connection` or `layers`"))
    end
    isempty(layers) && throw(ArgumentError("a Parallel layer must have at least one sub-layer"))
    return Parallel(connection, layers)
end

function (m::Parallel)(x, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple)
    return applyparallel(m.layers, m.connection, x, ps, st)
end

@generated function applyparallel(
    layers::NamedTuple{names}, connection::C, x::T, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
) where {names,C,T}
    N = length(names)
    y_symbols = [gensym() for _ in 1:(N + 1)]
    st_symbols = [gensym() for _ in 1:N]
    getinput(i) = T <: Tuple ? :(x[$i]) : :x
    calls = []
    append!(
        calls,
        [
            :(($(y_symbols[i]), $(st_symbols[i])) = layers[$i]($(getinput(i)), ps.$(names[i]), st.$(names[i]))) for
            i in 1:N
        ],
    )
    append!(calls, [:(st = NamedTuple{$names}((($(Tuple(st_symbols)...),))))])
    if C == Nothing
        append!(calls, [:($(y_symbols[N + 1]) = tuple($(Tuple(y_symbols[1:N])...)))])
    else
        append!(calls, [:($(y_symbols[N + 1]) = connection($(Tuple(y_symbols[1:N])...)))])
    end
    append!(calls, [:(return $(y_symbols[N + 1]), st)])
    return Expr(:block, calls...)
end

Base.keys(m::Parallel) = Base.keys(getfield(m, :layers))

"""
    BranchLayer(layers...)

Takes an input `x` and passes it through all the `layers` and returns a tuple of the outputs.

## Comparison with [`Parallel`](@ref)

This is slightly different from `Parallel(nothing, layers...)`

    * If the input is a tuple Parallel will pass each element individually to each layer
    * `BranchLayer` essentially assumes 1 input comes in and is branched out into `N` outputs

## Example

An easy way to replicate an input to an NTuple is to do

```julia
l = BranchLayer(
    NoOpLayer(),
    NoOpLayer(),
    NoOpLayer(),
)
```
"""
struct BranchLayer{T<:NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    layers::T
end

function BranchLayer(layers...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return BranchLayer(NamedTuple{names}(layers))
end

function BranchLayer(; kwargs...)
    layers = NamedTuple(kwargs)
    if :layers in Base.keys(layers)
        throw(ArgumentError("A BranchLayer cannot have a named sub-layer called `layers`"))
    end
    isempty(layers) && throw(ArgumentError("A BranchLayer must have at least one sub-layer"))
    return BranchLayer(layers)
end

(m::BranchLayer)(x, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple) = applybranching(m.layers, x, ps, st)

@generated function applybranching(
    layers::NamedTuple{names}, x, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
) where {names}
    N = length(names)
    y_symbols = [gensym() for _ in 1:N]
    st_symbols = [gensym() for _ in 1:N]
    calls = []
    append!(
        calls, [:(($(y_symbols[i]), $(st_symbols[i])) = layers[$i](x, ps.$(names[i]), st.$(names[i]))) for i in 1:N]
    )
    append!(calls, [:(st = NamedTuple{$names}((($(Tuple(st_symbols)...),))))])
    append!(calls, [:(return tuple($(Tuple(y_symbols)...)), st)])
    return Expr(:block, calls...)
end

Base.keys(m::BranchLayer) = Base.keys(getfield(m, :layers))

"""
    PairwiseFusion(connection, layers...)

```
x1 --> layer1 --> y1
                  |
                  connection --> layer2 --> y2
                  |                         |
                  x2                        connection --> layer3 --> y3
                                            |                         |
                                            x3                        connection --> y4
                                                                      |
                                                                      x4
```

## Arguments

* `connection`: Takes 2 inputs and combines them
* `layers`: [`AbstractExplicitLayer`](@ref)s 

## Input

Layer behaves differently based on input type:
1. Input `x` is a tuple of length `N` then the `layers` must be a tuple of length `N`. The computation is as follows

```julia
y = x[1]
for i in 1:N
    y = connection(x[i], layers[i](y))
end
```

2. Any other kind of input

```julia
y = x
for i in 1:N
    y = connection(x, layers[i](y))
end
```
"""
struct PairwiseFusion{F,T<:NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    connection::F
    layers::T
end

function PairwiseFusion(connection, layers...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return PairwiseFusion(connection, NamedTuple{names}(layers))
end

function PairwiseFusion(connection; kw...)
    layers = NamedTuple(kw)
    if :layers in Base.keys(layers) || :connection in Base.keys(layers)
        throw(ArgumentError("a PairwiseFusion layer cannot have a named sub-layer called `connection` or `layers`"))
    end
    isempty(layers) && throw(ArgumentError("a PairwiseFusion layer must have at least one sub-layer"))
    return PairwiseFusion(connection, layers)
end

function (m::PairwiseFusion)(x, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple)
    return applypairwisefusion(m.layers, m.connection, x, ps, st)
end

@generated function applypairwisefusion(
    layers::NamedTuple{names}, connection::C, x::T, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
) where {names,C,T}
    N = length(names)
    y_symbols = [gensym() for _ in 1:(N + 1)]
    st_symbols = [gensym() for _ in 1:N]
    getinput(i) = T <: Tuple ? :(x[$i]) : :x
    calls = [:($(y_symbols[N + 1]) = $(getinput(1)))]
    for i in 1:N
        push!(
            calls,
            :(($(y_symbols[i]), $(st_symbols[i])) = layers[$i]($(y_symbols[N + 1]), ps.$(names[i]), st.$(names[i]))),
        )
        push!(calls, :($(y_symbols[N + 1]) = connection($(y_symbols[i]), $(getinput(i + 1)))))
    end
    append!(calls, [:(st = NamedTuple{$names}((($(Tuple(st_symbols)...),))))])
    append!(calls, [:(return $(y_symbols[N + 1]), st)])
    return Expr(:block, calls...)
end

Base.keys(m::PairwiseFusion) = Base.keys(getfield(m, :layers))

"""
    Chain(layers...; disable_optimizations::Bool = false)

Collects multiple layers / functions to be called in sequence on a given input.

## Optimizations

Performs a few optimizations to generate reasonable architectures. Can be disabled using keyword argument `disable_optimizations`.
* All sublayers are recursively optimized.
* If a function `f` is passed as a layer and it doesn't take 3 inputs, it is converted to a WrappedFunction(`f`) which takes only one input.
* If the layer is a Chain, it is flattened.
* `NoOpLayer`s are removed.
* If there is only 1 layer (left after optimizations), then it is returned without the `Chain` wrapper.
* If there are no layers (left after optimizations), a `NoOpLayer` is returned.

## Input

Input `x` is passed sequentially to each layer, and must conform to the input requirements of the internal layers.

## Example

```julia
c = Chain(
    Dense(2, 3, relu),
    BatchNorm(3),
    Dense(3, 2)
)
```
"""
struct Chain{T} <: AbstractExplicitContainerLayer{(:layers,)}
    layers::T
    function Chain(xs...; disable_optimizations::Bool=false)
        xs = disable_optimizations ? xs : flatten_model(xs)
        length(xs) == 0 && return NoOpLayer()
        length(xs) == 1 && return first(xs)
        names = ntuple(i -> Symbol("layer_$i"), length(xs))
        layers = NamedTuple{names}(xs)
        return new{typeof(layers)}(layers)
    end
    Chain(xs::AbstractVector; disable_optimizations::Bool=false) = Chain(xs...; disable_optimizations)
end

function flatten_model(layers::Union{AbstractVector,Tuple})
    new_layers = []
    for l in layers
        f = flatten_model(l)
        if f isa Tuple || f isa AbstractVector
            append!(new_layers, f)
        elseif f isa Function
            if !hasmethod(f, (Any, Union{ComponentArray,NamedTuple}, NamedTuple))
                push!(new_layers, WrappedFunction(f))
            else
                push!(new_layers, f)
            end
        elseif f isa Chain
            append!(new_layers, f.layers)
        elseif f isa NoOpLayer
            continue
        else
            push!(new_layers, f)
        end
    end
    return layers isa AbstractVector ? new_layers : Tuple(new_layers)
end

flatten_model(x) = x

(c::Chain)(x, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple) = applychain(c.layers, x, ps, st)

@generated function applychain(
    layers::NamedTuple{fields}, x, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple{fields}
) where {fields}
    N = length(fields)
    x_symbols = [gensym() for _ in 1:N]
    st_symbols = [gensym() for _ in 1:N]
    calls = [:(($(x_symbols[1]), $(st_symbols[1])) = layers[1](x, ps.layer_1, st.layer_1))]
    append!(
        calls,
        [
            :(($(x_symbols[i]), $(st_symbols[i])) = layers[$i]($(x_symbols[i - 1]), ps.$(fields[i]), st.$(fields[i])))
            for i in 2:N
        ],
    )
    append!(calls, [:(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),))))])
    append!(calls, [:(return $(x_symbols[N]), st)])
    return Expr(:block, calls...)
end

Base.keys(m::Chain) = Base.keys(getfield(m, :layers))

"""
    Dense(in_dims => out_dims, activation=identity; init_weight=glorot_uniform, init_bias=zeros32, bias::Bool=true)

Create a traditional fully connected layer, whose forward pass is given by: `y = activation.(weight * x .+ bias)`

## Arguments

* `in_dims`: number of input dimensions
* `out_dims`: number of output dimensions
* `activation`: activation function
* `init_weight`: initializer for the weight matrix (`weight = init_weight(rng, out_dims, in_dims)`)
* `init_bias`: initializer for the bias vector (ignored if `bias=false`)
* `bias`: whether to include a bias vector

## Input

Input `x` must be a Matrix of size `in_dims × B` or a Vector of length `in_dims`
"""
struct Dense{bias,F1,F2,F3} <: AbstractExplicitLayer
    activation::F1
    in_dims::Int
    out_dims::Int
    init_weight::F2
    init_bias::F3
end

function Base.show(io::IO, d::Dense{bias}) where {bias}
    print(io, "Dense($(d.in_dims) => $(d.out_dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    bias || print(io, ", bias=false")
    return print(io, ")")
end

function Dense(mapping::Pair{<:Int,<:Int}, activation=identity; init_weight=glorot_uniform, init_bias=zeros32, bias::Bool=true)
    return Dense(first(mapping), last(mapping), activation; init_weight=init_weight, init_bias=init_bias, bias=bias)
end

function Dense(in_dims::Int, out_dims::Int, activation=identity; init_weight=glorot_uniform, init_bias=zeros32, bias::Bool=true)
    activation = NNlib.fast_act(activation)
    return Dense{bias,typeof(activation),typeof(init_weight),typeof(init_bias)}(activation, in_dims, out_dims, init_weight, init_bias)
end

function initialparameters(rng::AbstractRNG, d::Dense{bias}) where {bias}
    if bias
        return (weight=d.init_weight(rng, d.out_dims, d.in_dims), bias=d.init_bias(rng, d.out_dims, 1))
    else
        return (weight=d.init_weight(rng, d.out_dims, d.in_dims),)
    end
end

parameterlength(d::Dense{bias}) where {bias} = bias ? d.out_dims * (d.in_dims + 1) : d.out_dims * d.in_dims
statelength(d::Dense) = 0

@inline function (d::Dense{false})(x::AbstractArray, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple)
    return applyactivation(d.activation, ps.weight * x), st
end

@inline function (d::Dense{false,typeof(identity)})(
    x::AbstractArray, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
)
    return ps.weight * x, st
end

@inline function (d::Dense{true})(x::AbstractArray, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple)
    return applyactivation(d.activation, elementwise_add(ps.weight * x, ps.bias)), st
end

@inline function (d::Dense{true,typeof(identity)})(
    x::AbstractArray, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
)
    return elementwise_add(ps.weight * x, ps.bias), st
end

@inline function (d::Dense{true})(x::AbstractVector, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple)
    return applyactivation(d.activation, elementwise_add(ps.weight * x, vec(ps.bias))), st
end

@inline function (d::Dense{true,typeof(identity)})(
    x::AbstractVector, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
)
    return elementwise_add(ps.weight * x, vec(ps.bias)), st
end

"""
    Scale(dims, activation=identity; init_weight=ones32, init_bias=zeros32, bias::Bool=true)

Create a Sparsely Connected Layer with a very specific structure (only Diagonal Elements are non-zero). The forward pass is given by: `y = activation.(weight .* x .+ bias)`

## Arguments

* `dims`: dimension of the input
* `activation`: activation function
* `init_weight`: initializer for the weight matrix (`weight = init_weight(rng, out, in)`)
* `init_bias`: initializer for the bias vector (ignored if `bias=false`)
* `bias`: whether to include a bias vector

## Input

Input `x` must be a Matrix of size `dims × B` or a Vector of length `dims`
"""
struct Scale{bias,F1,D,F2,F3} <: AbstractExplicitLayer
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

function Scale(dims, activation=identity; init_weight=glorot_uniform, init_bias=zeros32, bias::Bool=true)
    activation = NNlib.fast_act(activation)
    return Scale{bias,typeof(activation),typeof(dims),typeof(init_weight),typeof(init_bias)}(activation, dims, init_weight, init_bias)
end

function initialparameters(rng::AbstractRNG, d::Scale{true})
    return (weight=d.init_weight(rng, d.dims), bias=d.init_bias(rng, d.dims))
end
initialparameters(rng::AbstractRNG, d::Scale{false}) = (weight=d.init_weight(rng, d.dims),)

parameterlength(d::Scale{bias}) where {bias} = (1 + bias) * d.dims
statelength(d::Scale) = 0

function (d::Scale{true})(x::AbstractArray, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple)
    return applyactivation(d.activation, elementwise_add(elementwise_mul(ps.weight, x), ps.bias)), st
end

function (d::Scale{true,typeof(identity)})(x::AbstractArray, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple)
    return elementwise_add(elementwise_mul(ps.weight, x), ps.bias), st
end

function (d::Scale{false})(x::AbstractArray, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple)
    return applyactivation(d.activation, elementwise_mul(ps.weight, x)), st
end

function (d::Scale{false,typeof(identity)})(x::AbstractArray, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple)
    return elementwise_mul(ps.weight, x), st
end
