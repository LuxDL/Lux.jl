module PrettyPrinting

using Functors: Functors

using LuxCore: LuxCore, AbstractLuxContainerLayer, AbstractLuxLayer, display_name

printable_children(x) = Functors.children(x)
function printable_children(m::AbstractLuxContainerLayer{layers}) where {layers}
    children = Functors.children(m)
    length(layers) ≥ 2 && return children
    field = first(layers)
    hasfield(typeof(children), field) || return children
    nt = getfield(children, field)
    nt isa NamedTuple || (nt = NamedTuple{(field,)}((nt,)))
    return merge(Base.structdiff(children, NamedTuple{(field,)}), nt)
end

show_leaflike(x) = Functors.isleaf(x)  # mostly follow Functors, except for:
show_leaflike(x::AbstractLuxLayer) = false

function underscorise(n::Integer)
    return join(reverse(join.(reverse.(Iterators.partition(digits(n), 3)))), '_')
end

function big_show(io::IO, obj, indent::Int=0, name=nothing)
    if obj isa Function || obj isa Nothing
        print(io, " "^indent, isnothing(name) ? "" : "$name = ", obj)
        indent != 0 && println(io, ",")
        return
    end
    children = printable_children(obj)
    if all(show_leaflike, values(children))
        layer_show(io, obj, indent, name)
    else
        println(io, " "^indent, isnothing(name) ? "" : "$name = ", display_name(obj), "(")
        for (k, c) in pairs(children)
            big_show(io, c, indent + 4, k)
        end
        if indent == 0  # i.e. this is the outermost container
            print(io, rpad(")", 2))
            big_finale(io, obj)
        else
            println(io, " "^indent, ")", ",")
        end
    end
end

function big_finale(io::IO, m, len=8)
    printstyled(io, " "^len, "# Total: "; color=:light_black)
    println(io, underscorise(LuxCore.parameterlength(m)), " parameters,")
    printstyled(io, " "^10, "#        plus "; color=:light_black)
    print(io, underscorise(LuxCore.statelength(m)), " states.")
    return
end

function layer_show(io::IO, layer, indent::Int=0, name=nothing)
    _str = isnothing(name) ? "" : "$name = "
    str = _str * sprint(show, layer; context=io)
    print(io, " "^indent, str, indent == 0 ? "" : ",")
    show_parameters_count(io, layer, indent, str)
    indent == 0 || println(io)
    return
end

function show_parameters_count(io::IO, layer, indent, str::String)
    paramlength = LuxCore.parameterlength(layer)
    if paramlength > 0
        print(io, " "^max(2, (indent == 0 ? 20 : 39) - indent - length(str)))
        printstyled(io, "# ", underscorise(paramlength), " parameters"; color=:light_black)
        nonparam = LuxCore.statelength(layer)
        if nonparam > 0
            printstyled(io, ", plus ", underscorise(nonparam),
                indent == 0 ? " non-trainable" : ""; color=:light_black)
        end
    end
    return
end

function print_wrapper_model(io::IO, desc::String, model::AbstractLuxLayer)
    if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        print(io, desc, "(\n")
        big_show(io, model, 4)
        print(io, ") ")
        big_finale(io, model)
        return
    end
    print(io, desc, "(")
    if !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
        layer_show(io, model)
    else
        show(io, model)
    end
    print(io, ")")
end

tuple_string(pad::Tuple) = all(==(pad[1]), pad) ? string(pad[1]) : string(pad)

end

function Base.show(io::IO, ::MIME"text/plain",
        x::Union{AbstractLuxContainerLayer, AbstractLuxWrapperLayer})
    if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        PrettyPrinting.big_show(io, x)
    elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
        PrettyPrinting.layer_show(io, x)
    else
        show(io, x)
    end
end

function Base.show(io::IO, ::MIME"text/plain", x::AbstractLuxLayer)
    !get(io, :compact, false) && return PrettyPrinting.layer_show(io, x)
    show(io, x)
end
