for T in [:Chain, :Parallel, :SkipConnection]  # container types
    @eval function Base.show(io::IO, ::MIME"text/plain", x::$T)
        if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
            _big_show(io, x)
        elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
            _layer_show(io, x)
        else
            show(io, x)
        end
    end
end

function _big_show(io::IO, obj, indent::Int=0, name=nothing)
    pre, post = "(", ")"
    children = _get_children(obj)
    if obj isa Function
        println(io, " "^indent, obj)
    elseif all(_show_leaflike, children)
        _layer_show(io, obj, indent, name)
    else
        println(io, " "^indent, isnothing(name) ? "" : "$name = ", nameof(typeof(obj)), pre)
        if obj isa Chain{<:NamedTuple} && children == getfield(obj, :layers)
            # then we insert names -- can this be done more generically? 
            for k in Base.keys(obj)
                _big_show(io, obj[k], indent + 4, k)
            end
        elseif obj isa Parallel{<:Any,<:NamedTuple}
            _big_show(io, obj.connection, indent + 4)
            for k in Base.keys(obj)
                _big_show(io, obj[k], indent + 4, k)
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

_show_leaflike(x) = Flux.Functors.isleaf(x)  # mostly follow Functors, except for:
_show_leaflike(x::ExplicitLayer) = false
_show_leaflike(::Tuple{Vararg{<:Number}}) = true         # e.g. stride of Conv
_show_leaflike(::Tuple{Vararg{<:AbstractArray}}) = true  # e.g. parameters of LSTMcell

_get_children(p::Parallel) = (p.connection, p.layers...)
_get_children(c::Chain) = c.layers
_get_children(s::SkipConnection) = (s.layers, s.connection)
_get_children(::Any) = ()
function _get_children(e::T) where {T<:ExplicitLayer}
    children = []
    for f ∈ fieldnames(T)
        x = getfield(e, f)
        if supertype(typeof(x)) == ExplicitLayer
            append!(children, x)
        end
    end
    return Tuple(children)
end

for T in [:Conv, :Dense, :BatchNorm]
    @eval function Base.show(io::IO, ::MIME"text/plain", x::$T)
        if !get(io, :compact, false)
            _layer_show(io, x)
        else
            show(io, x)
        end
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
            printstyled(io, ", plus ", underscorise(nonparam), indent == 0 ? " non-trainable" : ""; color=:light_black)
        end
    end
    return indent == 0 || println(io)
end

function _big_finale(io::IO, m)
    ps, st = setup(m)
    paramlength = parameterlength(m)
    nonparamlength = statelength(m)
    cnt = _childarray_sum(_ -> 1, ps)
    noncnt = _childarray_sum(_ -> 1, st)
    if cnt > 2
        pars = underscorise(paramlength)
        bytes = Base.format_bytes(Base.summarysize(m))
        if noncnt > 0
            nonparam = underscorise(nonparamlength)
            printstyled(io, " "^08, "# Total: ", cnt, " trainable arrays, "; color=:light_black)
            println(io, pars, " parameters,")
            printstyled(
                io,
                " "^10,
                "# plus ",
                noncnt,
                " non-trainable, ",
                nonparam,
                " states, summarysize ";
                color=:light_black,
            )
            print(io, bytes, ".")
        else
            printstyled(io, " "^18, "# Total: ", cnt, " arrays, "; color=:light_black)
            print(io, pars, " parameters, ", bytes, ".")
        end
    end
end

_childarray_sum(f, x::AbstractArray{<:Number}) = f(x)
function _childarray_sum(f, x)
    if Flux.Functors.isleaf(x)
        return 0
    else
        c = Flux.Functors.children(x)
        if length(c) == 0
            return 0
        else
            return sum(y -> _childarray_sum(f, y), c)
        end
    end
end

# utility functions

underscorise(n::Integer) = join(reverse(join.(reverse.(Iterators.partition(digits(n), 3)))), '_')

function _nan_show(io::IO, x)
    if !isempty(x) && _all(iszero, x)
        printstyled(io, "  (all zero)"; color=:cyan)
    elseif _any(isnan, x)
        printstyled(io, "  (some NaN)"; color=:red)
    elseif _any(isinf, x)
        printstyled(io, "  (some Inf)"; color=:red)
    end
end

_any(f, xs::AbstractArray{<:Number}) = any(f, xs)
_any(f, xs) = any(x -> _any(f, x), xs)
_any(f, x::Number) = f(x)

_all(f, xs) = !_any(!f, xs)
