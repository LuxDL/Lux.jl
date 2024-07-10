function Base.show(io::IO, ::MIME"text/plain", x::AbstractExplicitContainerLayer)
    if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        _big_show(io, x)
    elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
        _layer_show(io, x)
    else
        show(io, x)
    end
end

function Base.show(io::IO, ::MIME"text/plain", x::AbstractExplicitLayer)
    !get(io, :compact, false) && return _layer_show(io, x)
    show(io, x)
end

function _big_show(io::IO, obj, indent::Int=0, name=nothing)
    if obj isa Function || obj isa Nothing
        print(io, " "^indent, isnothing(name) ? "" : "$name = ", obj)
        indent != 0 && println(io, ",")
        return
    end

    children = _printable_children(obj)
    if all(_show_leaflike, children)
        _layer_show(io, obj, indent, name)
    else
        println(io, " "^indent, isnothing(name) ? "" : "$name = ", display_name(obj), "(")
        for (k, c) in pairs(children)
            _big_show(io, c, indent + 4, k)
        end
        if indent == 0  # i.e. this is the outermost container
            print(io, rpad(")", 2))
            _big_finale(io, obj)
        else
            println(io, " "^indent, ")", ",")
        end
    end
end

_printable_children(x) = Functors.children(x)
function _printable_children(m::AbstractExplicitContainerLayer{layers}) where {layers}
    children = Functors.children(m)
    length(layers) ≥ 2 && return children
    field = first(layers)
    hasfield(typeof(children), field) || return children
    nt = getfield(children, field)
    nt isa NamedTuple || (nt = NamedTuple{(field,)}((nt,)))
    return merge(Base.structdiff(children, NamedTuple{(field,)}), nt)
end
function _printable_children(l::Union{PairwiseFusion, Parallel})
    children = Functors.children(l)
    l.connection === nothing && return children.layers
    return merge((; l.connection), children.layers)
end
_printable_children(l::SkipConnection) = (; l.connection, l.layers)
function _printable_children(l::BidirectionalRNN)
    merge_mode = l.model.connection isa Broadcast.BroadcastFunction ? l.model.connection.f :
                 nothing
    return (; merge_mode, forward_cell=l.model.layers.forward_rnn.cell,
        backward_cell=l.model.layers.backward_rnn.rnn.cell)
end

_show_leaflike(x) = Functors.isleaf(x)  # mostly follow Functors, except for:
_show_leaflike(x::AbstractExplicitLayer) = false

function _layer_show(io::IO, layer, indent::Int=0, name=nothing)
    _str = isnothing(name) ? "" : "$name = "
    str = _str * sprint(show, layer; context=io)
    print(io, " "^indent, str, indent == 0 ? "" : ",")
    _show_parameters_count(io, layer, indent, str)
    indent == 0 || println(io)
    return
end

function _show_parameters_count(io::IO, layer, indent, str::String)
    paramlength = parameterlength(layer)
    if paramlength > 0
        print(io, " "^max(2, (indent == 0 ? 20 : 39) - indent - length(str)))
        printstyled(io, "# ", underscorise(paramlength), " parameters"; color=:light_black)
        nonparam = statelength(layer)
        if nonparam > 0
            printstyled(io, ", plus ", underscorise(nonparam),
                indent == 0 ? " non-trainable" : ""; color=:light_black)
        end
    end
end

function _big_finale(io::IO, m, len=8)
    printstyled(io, " "^len, "# Total: "; color=:light_black)
    println(io, underscorise(parameterlength(m)), " parameters,")
    printstyled(io, " "^10, "#        plus "; color=:light_black)
    print(io, underscorise(statelength(m)), " states.")
    return
end

function underscorise(n::Integer)
    return join(reverse(join.(reverse.(Iterators.partition(digits(n), 3)))), '_')
end

function _print_wrapper_model(io::IO, desc::String, model::AbstractExplicitLayer)
    if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        print(io, desc, "(\n")
        _big_show(io, model, 4)
        print(io, ") ")
        _big_finale(io, model)
        return
    end
    print(io, desc, "(")
    if !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
        _layer_show(io, model)
    else
        show(io, model)
    end
    print(io, ")")
end
