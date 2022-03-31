"""
    ReshapeLayer(dims)

Reshapes the passed array to have a size of `(dims..., :)`
"""
struct ReshapeLayer{N} <: AbstractExplicitLayer
    dims::NTuple{N,Int}
end

Base.@pure (r::ReshapeLayer)(x::AbstractArray, ::NamedTuple, st::NamedTuple) = reshape(x, r.dims..., :), st


"""
    FlattenLayer()

Flattens the passed array into a matrix.
"""
struct FlattenLayer <: AbstractExplicitLayer end

Base.@pure (f::FlattenLayer)(x::AbstractArray{T,N}, ::NamedTuple, st::NamedTuple) where {T,N} = reshape(x, :, size(x, N)), st


"""
    SelectDim(dim, i)

See the documentation for `selectdim` for more information.
"""
struct SelectDim{I} <: AbstractExplicitLayer
    dim::Int
    i::I
end

(s::SelectDim)(x, ::NamedTuple, st::NamedTuple) = selectdim(x, s.dim, s.i), st


"""
    NoOpLayer()

As the name suggests does nothing but allows pretty printing of layers.
"""
struct NoOpLayer <: AbstractExplicitLayer end

Base.@pure (noop::NoOpLayer)(x, ::NamedTuple, st::NamedTuple) = x, st


"""
    WrappedFunction(f)

Wraps a stateless and parameter less function. Might be used when a function is
added to [Chain](@doc). For example, `Chain(x -> relu.(x))` would not work and the
right thing to do would be `Chain((x, ps, st) -> (relu.(x), st))`. An easier thing
to do would be `Chain(WrappedFunction(Base.Fix1(broadcast, relu)))`
"""
struct WrappedFunction{F} <: AbstractExplicitLayer
    func::F
end

(wf::WrappedFunction)(x, ::NamedTuple, st::NamedTuple) = wf.func(x), st

## SkipConnection
struct SkipConnection{T<:AbstractExplicitLayer,F} <: AbstractExplicitLayer
    layers::T
    connection::F
end

initialparameters(rng::AbstractRNG, s::SkipConnection) = initialparameters(rng, s.layers)
initialstates(rng::AbstractRNG, s::SkipConnection) = initialstates(rng, s.layers)

parameterlength(s::SkipConnection) = parameterlength(s.layers)
statelength(s::SkipConnection) = statelength(s.layers)

function (skip::SkipConnection)(input, ps::NamedTuple, st::NamedTuple)
    mx, st = skip.layers(input, ps, st)
    return skip.connection(mx, input), st
end
  
function Base.show(io::IO, b::SkipConnection)
    print(io, "SkipConnection(", b.layers, ", ", b.connection, ")")
end


## Parallel
struct Parallel{F, T<:NamedTuple} <: AbstractExplicitLayer
    connection::F
    layers::T
end

initialparameters(rng::AbstractRNG, p::Parallel) = initialparameters(rng, p.layers)
initialstates(rng::AbstractRNG, p::Parallel) = initialstates(rng, p.layers)

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

function (m::Parallel)(x, ps::NamedTuple, st::NamedTuple)
    return applyparallel(m.layers, m.connection, x, ps, st)
end

@generated function applyparallel(layers::NamedTuple{names}, connection, x, ps::NamedTuple, st::NamedTuple) where {names}
    N = length(names)
    y_symbols = [gensym() for _ in 1:(N+1)]
    st_symbols = [gensym() for _ in 1:N]
    calls = []
    append!(calls, [:(($(y_symbols[i]), $(st_symbols[i])) = layers[$i](x, ps[$i], st[$i])) for i in 1:N])
    append!(calls, [:(st = NamedTuple{$names}((($(Tuple(st_symbols)...),))))])
    append!(calls, [:($(y_symbols[N + 1]) = connection($(Tuple(y_symbols[1:N])...),))])
    append!(calls, [:(return $(y_symbols[N + 1]), st)])
    return Expr(:block, calls...)
end

@generated function applyparallel(layers::NamedTuple{names}, connection, x::Tuple, ps::NamedTuple, st::NamedTuple) where {names}
    N = length(names)
    y_symbols = [gensym() for _ in 1:(N+1)]
    st_symbols = [gensym() for _ in 1:N]
    calls = []
    append!(calls, [:(($(y_symbols[i]), $(st_symbols[i])) = layers[$i](x[$i], ps[$i], st[$i])) for i in 1:N])
    append!(calls, [:(st = NamedTuple{$names}((($(Tuple(st_symbols)...),))))])
    append!(calls, [:($(y_symbols[N + 1]) = connection($(Tuple(y_symbols[1:N])...),))])
    append!(calls, [:(return $(y_symbols[N + 1]), st)])
    return Expr(:block, calls...)
end
  
Base.keys(m::Parallel) = Base.keys(getfield(m, :layers))
  
function Base.show(io::IO, m::Parallel)
    print(io, "Parallel(", m.connection, ", ", m.layers, ")")
end


## Chain
struct Chain{T} <: AbstractExplicitLayer
    layers::T
    function Chain(xs...)
        length(xs) == 1 && return first(xs)
        xs = flatten_model(xs)
        return new{typeof(xs)}(xs)
    end
    Chain(xs::AbstractVector) = Chain(xs...)
end

function Base.show(io::IO, c::Chain)
    print(io, "Chain(\n")
    for l in c.layers
        show(io, l)
        print(io, "\n")
    end
    return print(io, ")")
end

function flatten_model(layers::Union{AbstractVector,Tuple})
    new_layers = []
    for l in layers
        f = flatten_model(l)
        if f isa Tuple || f isa AbstractVector
            append!(new_layers, f)
        elseif f isa Chain
            append!(new_layers, f.layers)
        else
            push!(new_layers, f)
        end
    end
    return layers isa AbstractVector ? new_layers : Tuple(new_layers)
end

flatten_model(x) = x

function initialparameters(rng::AbstractRNG, c::Chain)
    return (; zip(ntuple(i -> Symbol("layer_$i"), length(c.layers)), initialparameters.(rng, c.layers))...)
end

function initialstates(rng::AbstractRNG, c::Chain)
    return (; zip(ntuple(i -> Symbol("layer_$i"), length(c.layers)), initialstates.(rng, c.layers))...)
end

(c::Chain)(x, ps::NamedTuple, s::NamedTuple) = applychain(c.layers, x, ps, s)

@generated function applychain(layers::Tuple{Vararg{<:Any,N}}, x, ps, st::NamedTuple{fields}) where {N, fields}
    x_symbols = [gensym() for _ in 1:N]
    st_symbols = [gensym() for _ in 1:N]
    calls = [:(($(x_symbols[1]), $(st_symbols[1])) = layers[1](x, ps[1], st[1]))]
    append!(calls, [:(($(x_symbols[i]), $(st_symbols[i])) = layers[$i]($(x_symbols[i - 1]), ps[$i], st[$i])) for i in 2:N])
    append!(calls, [:(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),))))])
    append!(calls, [:(return $(x_symbols[N]), st)])
    return Expr(:block, calls...)
end


## Linear
struct Dense{bias,F1,F2,F3} <: AbstractExplicitLayer
    λ::F1
    in_dims::Int
    out_dims::Int
    initW::F2
    initb::F3
end

function Base.show(io::IO, d::Dense)
    print(io, "Dense($(d.in_dims) => $(d.out_dims)")
    (d.λ == identity) || print(io, ", $(d.λ)")
    return print(io, ")")
end

function Dense(mapping::Pair{<:Int,<:Int}, λ=identity; initW=glorot_uniform, initb=zeros32, bias::Bool=true)
    return Dense(first(mapping), last(mapping), λ, initW=initW, initb=initb, bias=bias)
end

function Dense(in_dims::Int, out_dims::Int, λ=identity; initW=glorot_uniform, initb=zeros32, bias::Bool=true)
    return Dense{bias,typeof(λ),typeof(initW),typeof(initb)}(λ, in_dims, out_dims, initW, initb)
end

function initialparameters(rng::AbstractRNG, d::Dense{true})
    return (weight=d.initW(rng, d.out_dims, d.in_dims), bias=d.initb(rng, d.out_dims, 1))
end
initialparameters(rng::AbstractRNG, d::Dense{false}) = (weight=d.initW(rng, d.out_dims, d.in_dims),)

parameterlength(d::Dense{true}) = d.out_dims * (d.in_dims + 1)
parameterlength(d::Dense{false}) = d.out_dims * d.in_dims
statelength(d::Dense) = 0

function (d::Dense)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    y, st = d(reshape(x, size(x, 1), :), ps, st)
    return reshape(y, :, size(x)[2:end]...), st
end

function (d::Dense{false})(x::AbstractVecOrMat, ps::NamedTuple, st::NamedTuple)
    return (NNlib.fast_act(d.λ, x)).(ps.weight * x), st
end

function (d::Dense{true})(x::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    return (NNlib.fast_act(d.λ, x)).(ps.weight * x .+ ps.bias), st
end

function (d::Dense{true})(x::AbstractVector, ps::NamedTuple, st::NamedTuple)
    return (NNlib.fast_act(d.λ, x)).(ps.weight * x .+ vec(ps.bias)), st
end


## Diagonal
struct Diagonal{bias,F1,F2,F3} <: AbstractExplicitLayer
    λ::F1
    dims::Int
    initW::F2
    initb::F3
end

function Base.show(io::IO, d::Diagonal)
    print(io, "Diagonal($(d.dims)")
    (d.λ == identity) || print(io, ", $(d.λ)")
    return print(io, ")")
end

function Diagonal(dims, λ=identity; initW=glorot_uniform, initb=zeros32, bias::Bool=true)
    return Diagonal{bias,typeof(λ),typeof(initW),typeof(initb)}(λ, dims, initW, initb)
end

function initialparameters(rng::AbstractRNG, d::Diagonal{true})
    return (weight=d.initW(rng, d.dims), bias=d.initb(rng, d.dims))
end
initialparameters(rng::AbstractRNG, d::Diagonal{false}) = (weight=d.initW(rng, d.dims),)

parameterlength(d::Diagonal{true}) = 2 * d.dims
parameterlength(d::Diagonal{false}) = d.dims
statelength(d::Diagonal) = 0

function (d::Diagonal{false})(x::AbstractVecOrMat, ps::NamedTuple, st::NamedTuple)
    return (NNlib.fast_act(d.λ, x)).(ps.weight .* x), st
end

function (d::Diagonal{true})(x::AbstractVecOrMat, ps::NamedTuple, st::NamedTuple)
    return (NNlib.fast_act(d.λ, x)).(ps.weight .* x .+ ps.bias), st
end
