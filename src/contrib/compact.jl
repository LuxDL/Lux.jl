using MacroTools

# This functionality is based off of the implementation in Fluxperimental.jl
# https://github.com/FluxML/Fluxperimental.jl/blob/cc0e36fdd542cc6028bc69449645dc0390dd980b/src/compact.jl
struct LuxCompactModelParsingException <: Exception
    msg::String
end

function Base.showerror(io::IO, e::LuxCompactModelParsingException)
    print(io, "LuxCompactModelParsingException(", e.msg, ")")
    if !TruncatedStacktraces.VERBOSE[]
        println(io, TruncatedStacktraces.VERBOSE_MSG)
    end
end

"""
    @compact(kw...) do x
        ...
    end

## Example

```julia
using Lux

Lux.Experimental.@compact(; d₁=Dense(2 => 5, relu), d₂=Dense(5 => 2)) do x
    return d₂(d₁(x))
end
```
"""
macro compact(_exs...)
    # check inputs, extracting function expression fex and unprocessed keyword arguments _kwexs
    if isempty(_exs)
        msg = "expects at least two expressions: a function and at least one keyword"
        throw(LuxCompactModelParsingException(msg))
    end
    if Meta.isexpr(first(_exs), :parameters)
        if length(_exs) < 2
            throw(LuxCompactModelParsingException("expects an anonymous function"))
        end
        fex = _exs[2]
        _kwexs = (_exs[1], _exs[3:end]...)
    else
        fex = first(_exs)
        _kwexs = _exs[2:end]
    end
    if !Meta.isexpr(fex, :(->))
        throw(LuxCompactModelParsingException("expects an anonymous function"))
    end
    isempty(_kwexs) && throw(LuxCompactModelParsingException("expects keyword arguments"))
    if any(ex -> !Meta.isexpr(ex, (:kw, :(=), :parameters)), _kwexs)
        throw(LuxCompactModelParsingException("expects only keyword arguments"))
    end

    # process keyword arguments
    if Meta.isexpr(first(_kwexs), :parameters) # handle keyword arguments provided after semicolon
        kwexs1 = map(ex -> ex isa Symbol ? Expr(:kw, ex, ex) : ex, first(_kwexs).args)
        _kwexs = _kwexs[2:end]
    else
        kwexs1 = ()
    end
    kwexs2 = map(ex -> Expr(:kw, ex.args...), _kwexs) # handle keyword arguments provided before semicolon
    kwexs = (kwexs1..., kwexs2...)

    # check if user has named layer
    name_idx = findfirst(ex -> ex.args[1] == :name, kwexs)
    name = nothing
    if name_idx !== nothing && kwexs[name_idx].args[2] !== nothing
        if length(kwexs) == 1
            throw(LuxCompactModelParsingException("expects keyword arguments"))
        end
        name = kwexs[name_idx].args[2]
        # remove name from kwexs (a tuple)
        kwexs = (kwexs[1:(name_idx - 1)]..., kwexs[(name_idx + 1):end]...)
    end

    # make strings
    layer = "@compact"
    setup = NamedTuple(map(ex -> Symbol(string(ex.args[1])) => string(ex.args[2]), kwexs))
    input = try
        fex_args = fex.args[1]
        isa(fex_args, Symbol) ? string(fex_args) : join(fex_args.args, ", ")
    catch e
        @warn "Function stringifying does not yet handle all cases. Falling back to empty string for input arguments"
    end
    block = string(Base.remove_linenums!(fex).args[2])

    # edit expressions
    vars = map(ex -> ex.args[1], kwexs)
    fex = supportself(fex, vars)

    # assemble
    return esc(:($CompactLuxLayer($fex, $name, ($layer, $input, $block), $setup;
        $(kwexs...))))
end

function supportself(fex::Expr, vars)
    @gensym self ps st curried_f res
    # To avoid having to manipulate fex's arguments and body explicitly, we split the input
    # function body and add the required arguments to the function definition.
    sdef = splitdef(fex)
    if length(sdef[:args]) != 1
        throw(LuxCompactModelParsingException("expects exactly 1 argument"))
    end
    args = [self, sdef[:args]..., ps, st]
    calls = []
    for var in vars
        push!(calls,
            :($var = Lux.Experimental.__maybe_stateful_layer(Lux._getproperty($self,
                    $(Val(var))), Lux._getproperty($ps, $(Val(var))),
                Lux._getproperty($st, $(Val(var))))))
    end
    body = Expr(:block, calls..., sdef[:body])
    sdef[:body] = body
    sdef[:args] = args
    return combinedef(sdef)
end

__maybe_stateful_layer(layer, ps, st) = StatefulLuxLayer(layer, ps, st)
__maybe_stateful_layer(::Nothing, ps, st) = ps === nothing ? st : ps

@concrete struct ValueStorage <: AbstractExplicitLayer
    ps_init_fns
    st_init_fns
end

function ValueStorage(; kwargs...)
    ps_init_fns, st_init_fns = [], []
    for (key, val) in pairs(kwargs)
        push!(val isa AbstractArray ? ps_init_fns : st_init_fns, key => () -> val)
    end
    return ValueStorage(NamedTuple(ps_init_fns), NamedTuple(st_init_fns))
end

function (v::ValueStorage)(x, ps, st)
    throw(ArgumentError("ValueStorage isn't meant to be used as a layer!!!"))
end

function initialparameters(::AbstractRNG, v::ValueStorage)
    return NamedTuple([n => fn() for (n, fn) in pairs(v.ps_init_fns)])
end

function initialstates(::AbstractRNG, v::ValueStorage)
    return NamedTuple([n => fn() for (n, fn) in pairs(v.st_init_fns)])
end

@concrete struct CompactLuxLayer <:
                 AbstractExplicitContainerLayer{(:layers, :value_storage)}
    f
    name::NAME_TYPE
    strings::NTuple{3, String}
    setup_strings
    layers
    value_storage
end

function initialparameters(rng::AbstractRNG, m::CompactLuxLayer)
    return (; initialparameters(rng, m.layers)...,
        initialparameters(rng, m.value_storage)...)
end

function initialstates(rng::AbstractRNG, m::CompactLuxLayer)
    return (; initialstates(rng, m.layers)..., initialstates(rng, m.value_storage)...)
end

function CompactLuxLayer(f::Function, name::NAME_TYPE, str::Tuple, setup_str::NamedTuple;
    kws...)
    layers, others = [], []
    for (name, val) in pairs(kws)
        push!(val isa AbstractExplicitLayer ? layers : others, name => val)
    end
    return CompactLuxLayer(f, name, str, setup_str, NamedTuple((; layers...)),
        ValueStorage(; others...))
end

function (m::CompactLuxLayer)(x, ps, st::NamedTuple{fields}) where {fields}
    y = m.f(m.layers, x, ps, st)
    st_ = NamedTuple{fields}((getfield.((st,), fields)...,))
    return y, st_
end

# TODO: Pretty printing the layer code