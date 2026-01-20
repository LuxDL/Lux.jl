module PrettyPrinting

using Functors: Functors

using LuxCore: LuxCore, AbstractLuxWrapperLayer, AbstractLuxLayer, display_name

printable_children(x) = Functors.children(x)
function printable_children(m::AbstractLuxWrapperLayer{field}) where {field}
    children = Functors.children(m)
    hasfield(typeof(children), field) || return children
    nt = getfield(children, field)
    nt isa NamedTuple || (nt = NamedTuple{(field,)}((nt,)))
    return merge(Base.structdiff(children, NamedTuple{(field,)}), nt)
end

show_leaflike(x) = Functors.isleaf(x)  # mostly follow Functors, except for:
show_leaflike(::AbstractLuxLayer) = false

isa_printable_leaf(_) = false

function underscorise(n::Integer)
    return join(reverse(join.(reverse.(Iterators.partition(digits(n), 3)))), '_')
end

parse_name(name) = parse_name(string(name))
function parse_name(name::String)
    # Match names ending in `_` and digits, e.g. `layer_1`
    m = match(r"^(.*)_(\d+)$", name)
    if m !== nothing
        @assert m.captures[2] isa AbstractString "Invalid name: $name"
        return m.captures[1], true, parse(Int, m.captures[2])
    end
    # Match names ending in digits, e.g. `layer1`
    m = match(r"^(.*?)(\d+)$", name)
    if m !== nothing
        @assert m.captures[2] isa AbstractString "Invalid name: $name"
        return m.captures[1], false, parse(Int, m.captures[2])
    end
    return name, false, nothing
end

function big_show(io::IO, obj, indent::Int=0, name=nothing, multiplier=1)
    if obj isa Function || obj isa Nothing
        print(io, " "^indent, isnothing(name) ? "" : "$name = ", obj)
        indent != 0 && println(io, ",")
        return nothing
    end
    children = printable_children(obj)
    if all(show_leaflike, values(children)) || isa_printable_leaf(obj)
        layer_show(io, obj, indent, name, multiplier)
        return nothing
    end
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", display_name(obj), "(")

    # Group consecutive identical layers
    child_keys = collect(keys(children))
    child_vals = collect(values(children))
    n = length(child_vals)
    i = 1
    while i <= n
        curr = child_vals[i]
        curr_key = child_keys[i]
        basename, has_underscore, curr_suffix = parse_name(curr_key)
        first_suffix = curr_suffix
        # Find run of identical layers with consecutive names
        j = i
        while (
            j < n && typeof(child_vals[j + 1]) == typeof(curr) && child_vals[j + 1] == curr
        )
            next_key = child_keys[j + 1]
            next_basename, next_has_underscore, next_suffix = parse_name(next_key)
            if (
                next_basename == basename &&
                next_has_underscore == has_underscore &&
                curr_suffix !== nothing &&
                next_suffix !== nothing &&
                next_suffix == curr_suffix + 1
            )
                j += 1
                curr_suffix = next_suffix
            else
                break
            end
        end
        if j > i # Print as a range
            updated_name = "$(basename)$(has_underscore ? "_" : "")($(first_suffix)-$(curr_suffix))"
            big_show(io, curr, indent + 4, updated_name, (j - i + 1) * multiplier)
            i = j + 1
        else
            big_show(io, curr, indent + 4, curr_key, multiplier)
            i += 1
        end
    end

    if indent == 0  # i.e. this is the outermost container
        print(io, rpad(")", 2))
        big_finale(io, obj)
    else
        println(io, " "^indent, ")", ",")
    end
    return nothing
end

function big_finale(io::IO, m, len=8)
    printstyled(io, " "^len, "# Total: "; color=:light_black)
    println(io, underscorise(LuxCore.parameterlength(m)), " parameters,")
    printstyled(io, " "^10, "#        plus "; color=:light_black)
    print(io, underscorise(LuxCore.statelength(m)), " states.")
    return nothing
end

function layer_show(io::IO, layer, indent::Int=0, name=nothing, multiplier=1)
    _str = isnothing(name) ? "" : "$name = "
    str = _str * sprint(show, layer; context=io)
    print(io, " "^indent, str, indent == 0 ? "" : ",")
    show_parameters_count(io, layer, indent, length(str), multiplier)
    indent == 0 || println(io)
    return nothing
end

function show_parameters_count(io::IO, layer, indent, str_len, multiplier=1)
    paramlength = LuxCore.parameterlength(layer)
    if paramlength > 0
        print(io, " "^max(2, (indent == 0 ? 29 : 49) - indent - str_len))
        printstyled(
            io,
            "# $(multiply_count(paramlength, multiplier)) parameters";
            color=:light_black,
        )
        nonparam = LuxCore.statelength(layer)
        if nonparam > 0
            printstyled(
                io,
                ", plus $(multiply_count(nonparam, multiplier)) non-trainable";
                color=:light_black,
            )
        end
    end
    return nothing
end

function multiply_count(count, multiplier)
    multiplier == 1 && return "$(underscorise(count))"
    return "$(underscorise(count * multiplier)) ($(underscorise(count)) x $multiplier)"
end

function print_wrapper_model(io::IO, desc::String, model::AbstractLuxLayer)
    if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        print(io, desc, "(\n")
        big_show(io, model, 4)
        print(io, ") ")
        big_finale(io, model)
        return nothing
    end
    print(io, desc, "(")
    if !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
        layer_show(io, model)
    else
        show(io, model)
    end
    return print(io, ")")
end

tuple_string(pad::Tuple) = all(==(pad[1]), pad) ? string(pad[1]) : string(pad)

end

function Base.show(
    io::IO, ::MIME"text/plain", x::Union{AbstractLuxContainerLayer,AbstractLuxWrapperLayer}
)
    return if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        PrettyPrinting.big_show(io, x)
    elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
        PrettyPrinting.layer_show(io, x)
    else
        show(io, x)
    end
end

function Base.show(io::IO, ::MIME"text/plain", x::AbstractLuxLayer)
    !get(io, :compact, false) && return PrettyPrinting.layer_show(io, x)
    return show(io, x)
end
