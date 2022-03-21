struct Chain{T} <: ExplicitLayer
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
