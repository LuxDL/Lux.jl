struct ReshapeLayer{N} <: ExplicitLayer
    dims::NTuple{N,Int}
end

(r::ReshapeLayer)(x::AbstractArray, ::NamedTuple, st::NamedTuple) = reshape(x, r.dims..., :), st


struct FlattenLayer <:ExplicitLayer end

(f::FlattenLayer)(x::AbstractArray{T,N}, ::NamedTuple, st::NamedTuple) where {T,N} = reshape(x, :, size(x, N)), st


struct SelectDim{I} <: ExplicitLayer
    dim::Int
    i::I
end

(s::SelectDim)(x, ::NamedTuple, st::NamedTuple) = selectdim(x, s.dim, s.i), st


struct NoOpLayer <: ExplicitLayer end

(noop::NoOpLayer)(x, ::NamedTuple, st::NamedTuple) = x, st


struct SkipConnection{T<:ExplicitLayer,F} <: ExplicitLayer
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


struct Parallel{F, T<:NamedTuple} <: ExplicitLayer
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
