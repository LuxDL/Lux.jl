"""
    ReshapeLayer(dims)

Reshapes the passed array to have a size of `(dims..., :)`
"""
struct ReshapeLayer{N} <: AbstractExplicitLayer
    dims::NTuple{N,Int}
end

Base.@pure function (r::ReshapeLayer)(x::AbstractArray, ps, st::NamedTuple)
    return reshape(x, r.dims..., :), st
end

"""
    FlattenLayer()

Flattens the passed array into a matrix.
"""
struct FlattenLayer <: AbstractExplicitLayer end

Base.@pure function (f::FlattenLayer)(
    x::AbstractArray{T,N}, ps, st::NamedTuple
) where {T,N}
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

Base.@pure (s::SelectDim)(x, ps, st::NamedTuple) = selectdim(x, s.dim, s.i), st

"""
    NoOpLayer()

As the name suggests does nothing but allows pretty printing of layers.
"""
struct NoOpLayer <: AbstractExplicitLayer end

Base.@pure (noop::NoOpLayer)(x, ps, st::NamedTuple) = x, st

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

(wf::WrappedFunction)(x, ps, st::NamedTuple) = wf.func(x), st

function Base.show(io::IO, w::WrappedFunction)
    return print(io, "WrappedFunction(", w.func, ")")
end

## SkipConnection
struct SkipConnection{T<:AbstractExplicitLayer,F} <: AbstractExplicitContainerLayer{(:layers,)}
    layers::T
    connection::F
end

function (skip::SkipConnection)(input, ps::NamedTuple, st::NamedTuple)
    mx, st = skip.layers(input, ps, st)
    return skip.connection(mx, input), st
end

## Parallel
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

## Branching Layer
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

## PairwiseFusion
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

## Chain
struct Chain{T} <: AbstractExplicitContainerLayer{(:layers,)}
    layers::T
    function Chain(xs...)
        xs = flatten_model(xs)
        length(xs) == 0 && return NoOpLayer()
        length(xs) == 1 && return first(xs)
        names = ntuple(i -> Symbol("layer_$i"), length(xs))
        layers = NamedTuple{names}(xs)
        return new{typeof(layers)}(layers)
    end
    Chain(xs::AbstractVector) = Chain(xs...)
end

function flatten_model(layers::Union{AbstractVector,Tuple})
    new_layers = []
    for l in layers
        f = flatten_model(l)
        if f isa Tuple || f isa AbstractVector
            append!(new_layers, f)
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
    return Dense(first(mapping), last(mapping), λ; initW=initW, initb=initb, bias=bias)
end

function Dense(in_dims::Int, out_dims::Int, λ=identity; initW=glorot_uniform, initb=zeros32, bias::Bool=true)
    λ = NNlib.fast_act(λ)
    return Dense{bias,typeof(λ),typeof(initW),typeof(initb)}(λ, in_dims, out_dims, initW, initb)
end

function initialparameters(rng::AbstractRNG, d::Dense{bias}) where {bias}
    if bias
        return (weight=d.initW(rng, d.out_dims, d.in_dims), bias=d.initb(rng, d.out_dims, 1))
    else
        return (weight=d.initW(rng, d.out_dims, d.in_dims),)
    end
end

parameterlength(d::Dense{bias}) where {bias} = bias ? d.out_dims * (d.in_dims + 1) : d.out_dims * d.in_dims
statelength(d::Dense) = 0

Base.@pure function (d::Dense{bias,λT})(
    x::AbstractArray{T,N}, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
) where {bias,T,N,λT}
    if bias
        b = N == 1 ? view(ps.bias, :, 1) : ps.bias
        if λT == typeof(identity)
            return fast_matmul(ps.weight, x) .+ b, st
        else
            return d.λ.(fast_matmul(ps.weight, x) .+ b), st
        end
    else
        if λT == typeof(identity)
            return fast_matmul(ps.weight, x), st
        else
            return d.λ.(fast_matmul(ps.weight, x)), st
        end
    end
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
    λ = NNlib.fast_act(λ)
    return Diagonal{bias,typeof(λ),typeof(initW),typeof(initb)}(λ, dims, initW, initb)
end

function initialparameters(rng::AbstractRNG, d::Diagonal{true})
    return (weight=d.initW(rng, d.dims), bias=d.initb(rng, d.dims))
end
initialparameters(rng::AbstractRNG, d::Diagonal{false}) = (weight=d.initW(rng, d.dims),)

parameterlength(d::Diagonal{bias}) where {bias} = (1 + bias) * d.dims
statelength(d::Diagonal) = 0

Base.@pure function (d::Diagonal{bias})(x::AbstractVecOrMat, ps::NamedTuple, st::NamedTuple) where {bias}
    if bias
        return d.λ.(ps.weight .* x .+ ps.bias), st
    else
        return d.λ.(ps.weight .* x), st
    end
end
