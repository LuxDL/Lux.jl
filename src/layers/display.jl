function Base.show(io::IO, ::MIME"text/plain", x::AbstractExplicitContainerLayer)
    if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        _big_show(io, x)
    elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
        _layer_show(io, x)
    else
        show(io, x)
    end
end

function _big_show(io::IO, obj, indent::Int=0, name=nothing)
    pre, post = "(", ")"
    children = _get_children(obj)
    if obj isa Function
        println(io, " "^indent, isnothing(name) ? "" : "$name = ", obj)
    elseif all(_show_leaflike, children)
        _layer_show(io, obj, indent, name)
    else
        println(io, " "^indent, isnothing(name) ? "" : "$name = ", display_name(obj), pre)
        if obj isa Chain{<:NamedTuple}
            for k in Base.keys(obj)
                _big_show(io, obj.layers[k], indent + 4, k)
            end
        elseif obj isa Parallel{<:Any, <:NamedTuple}
            if obj.connection !== nothing
                _big_show(io, obj.connection, indent + 4)
            end
            for k in Base.keys(obj)
                _big_show(io, obj.layers[k], indent + 4, k)
            end
        elseif obj isa PairwiseFusion
            _big_show(io, obj.connection, indent + 4)
            for k in Base.keys(obj)
                _big_show(io, obj.layers[k], indent + 4, k)
            end
        elseif obj isa BranchLayer
            for k in Base.keys(obj)
                _big_show(io, obj.layers[k], indent + 4, k)
            end
        elseif obj isa Maxout
            for k in Base.keys(obj)
                _big_show(io, obj.layers[k], indent + 4, k)
            end
        elseif children isa NamedTuple
            for (k, c) in pairs(children)
                _big_show(io, c, indent + 4, k)
            end
        else
            for c in children
                _big_show(io, c, indent + 4)
            end
        end
        if indent == 0  # i.e. this is the outermost container
            print(io, rpad(post, 2))
            _big_finale(io, obj)
        else
            println(io, " "^indent, post, ",")
        end
    end
end

function _big_show(io::IO, obj::CompactLuxLayer, indent::Int=0, name=nothing)
    setup_strings = obj.setup_strings
    local_name = obj.name
    layer, input, block = obj.strings
    if local_name !== nothing && local_name != ""
        Lux._layer_show(io, obj, indent, name)
        return
    end
    pre, post = ("(", ")")
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", layer, pre)
    for (k, v) in pairs(setup_strings)
        val = _getproperty(obj.layers, Val(k))
        if val === nothing
            println(io, " "^(indent + 4), "$k = $v,")
        else
            Lux._big_show(io, val, indent + 4, k)
        end
    end
    if indent == 0  # i.e. this is the outermost container
        print(io, rpad(post, 1))
    else
        print(io, " "^indent, post)
    end
    input != "" && print(io, " do ", input)
    if block != ""
        block_to_print = block[6:end]
        # Increase indentation of block according to `indent`:
        block_to_print = replace(block_to_print, r"\n" => "\n" * " "^(indent))
        print(io, " ", block_to_print)
    end
    if indent == 0
        Lux._big_finale(io, obj, 7)
    else
        println(io, ",")
    end
    return
end

_show_leaflike(x) = Functors.isleaf(x)  # mostly follow Functors, except for:
_show_leaflike(::Tuple{}) = false       # Prevents method ambiguity
_show_leaflike(x::AbstractExplicitLayer) = false
_show_leaflike(::Tuple{Vararg{Number}}) = true         # e.g. stride of Conv
_show_leaflike(::Tuple{Vararg{AbstractArray}}) = true  # e.g. parameters of LSTMcell

function _get_children(l::AbstractExplicitContainerLayer{names}) where {names}
    return NamedTuple{names}(getfield.((l,), names))
end
function _get_children(p::Parallel)
    return p.connection === nothing ? p.layers : (p.connection, p.layers...)
end
_get_children(s::SkipConnection) = (s.layers, s.connection)
_get_children(s::WeightNorm) = (s.layer,)
_get_children(nt::NamedTuple) = nt
_get_children(::Any) = ()

function Base.show(io::IO, ::MIME"text/plain", x::AbstractExplicitLayer)
    if !get(io, :compact, false)
        _layer_show(io, x)
    else
        show(io, x)
    end
end

function _layer_show(io::IO, layer, indent::Int=0, name=nothing)
    _str = isnothing(name) ? "" : "$name = "
    str = _str * sprint(show, layer; context=io)
    print(io, " "^indent, str, indent == 0 ? "" : ",")
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
    indent == 0 || println(io)
    return
end

function _big_finale(io::IO, m, len=8)
    paramlength = parameterlength(m)
    nonparamlength = statelength(m)
    pars = underscorise(paramlength)
    nonparam = underscorise(nonparamlength)
    printstyled(io, " "^len, "# Total: "; color=:light_black)
    println(io, pars, " parameters,")
    printstyled(io, " "^10, "#        plus "; color=:light_black)
    print(io, nonparam, " states.")
    return
end

# utility functions

function underscorise(n::Integer)
    return join(reverse(join.(reverse.(Iterators.partition(digits(n), 3)))), '_')
end
