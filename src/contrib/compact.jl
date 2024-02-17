using MacroTools
import ConstructionBase: constructorof

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
    @compact(forward::Function; name=nothing, dispatch=nothing, parameters...)

Creates a layer by specifying some `parameters`, in the form of keywords, and (usually as a
`do` block) a function for the forward pass. You may think of `@compact` as a specialized
`let` block creating local variables that are trainable in Lux. Declared variable names may
be used within the body of the `forward` function. Note that unlike typical Lux models, the
forward function doesn't need to explicitly manage states.

## Reserved Kwargs:

 1. `name`: The name of the layer.
 2. `dispatch`: The constructed layer has the type
    `Lux.Experimental.CompactLuxLayer{dispatch}` which can be used for custom dispatches.

## Examples

Here is a linear model:

```julia
using Lux, Random
import Lux.Experimental: @compact

r = @compact(w=rand(3)) do x
    return w .* x
end
ps, st = Lux.setup(Xoshiro(0), r)
r([1, 1, 1], ps, st)  # x is set to [1, 1, 1].
```

Here is a linear model with bias and activation:

```julia
d_in = 5
d_out = 7
d = @compact(W=randn(d_out, d_in), b=zeros(d_out), act=relu) do x
    y = W * x
    return act.(y .+ b)
end
ps, st = Lux.setup(Xoshiro(0), d)
d(ones(5, 10), ps, st) # 7×10 Matrix as output.

ps_dense = (; weight=ps.W, bias=ps.b)
first(d([1, 2, 3, 4, 5], ps, st)) ≈
first(Dense(d_in => d_out, relu)([1, 2, 3, 4, 5], ps_dense, NamedTuple())) # Equivalent to a dense layer
```

Finally, here is a simple MLP:

```julia
n_in = 1
n_out = 1
nlayers = 3

model = @compact(w1=Dense(n_in, 128),
    w2=[Dense(128, 128) for i in 1:nlayers],
    w3=Dense(128, n_out),
    act=relu) do x
    embed = act(w1(x))
    for w in w2
        embed = act(w(embed))
    end
    out = w3(embed)
    return out
end

ps, st = Lux.setup(Xoshiro(0), model)

model(randn(n_in, 32), ps, st)  # 1×32 Matrix as output.
```

We can train this model just like any Lux model:

```julia
using Optimisers, Zygote

x_data = collect(-2.0f0:0.1f0:2.0f0)'
y_data = 2 .* x_data .- x_data .^ 3
optim = Optimisers.setup(Adam(), ps)

for epoch in 1:1000
    loss, gs = Zygote.withgradient(ps -> sum(abs2, first(model(x_data, ps, st)) .- y_data),
        ps)
    @show epoch, loss
    Optimisers.update!(optim, ps, gs[1])
end
```

You may also specify a `name` for the model, which will be used instead of the default
printout, which gives a verbatim representation of the code used to construct the model:

```julia
model = @compact(w=rand(3), name="Linear(3 => 1)") do x
    return sum(w .* x)
end

println(model)  # "Linear(3 => 1)()"
```

This can be useful when using `@compact` to hierarchically construct complex models to be
used inside a `Chain`.

:::tip Type Stability

If your input function `f` is type-stable but the generated model is not type stable, it
should be treated as a bug. We will appreciate issues if you find such cases.

:::

:::warning Parameter Count

Array Parameter don't print the number of parameters on the side. However, they do account
for the total number of parameters printed at the bottom.

:::
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

    # check if user has provided a custom dispatch
    dispatch_idx = findfirst(ex -> ex.args[1] == :dispatch, kwexs)
    dispatch = nothing
    if dispatch_idx !== nothing && kwexs[dispatch_idx].args[2] !== nothing
        if length(kwexs) == 1
            throw(LuxCompactModelParsingException("expects keyword arguments"))
        end
        dispatch = kwexs[dispatch_idx].args[2]
        # remove dispatch from kwexs (a tuple)
        kwexs = (kwexs[1:(dispatch_idx - 1)]..., kwexs[(dispatch_idx + 1):end]...)
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
    return esc(:($CompactLuxLayer{$dispatch}($fex, $name, ($layer, $input, $block), $setup;
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
            :($var = Lux.Experimental.__maybe_make_stateful(
                Lux._getproperty($self,
                    $(Val(var))), Lux._getproperty($ps, $(Val(var))),
                Lux._getproperty($st, $(Val(var))))))
    end
    body = Expr(:let, Expr(:block, calls...), sdef[:body])
    sdef[:body] = body
    sdef[:args] = args
    return combinedef(sdef)
end

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

@concrete struct CompactLuxLayer{dispatch} <:
                 AbstractExplicitContainerLayer{(:layers, :value_storage)}
    f
    name::NAME_TYPE
    strings::NTuple{3, String}
    setup_strings
    layers
    value_storage
end

function constructorof(::Type{<:CompactLuxLayer{dispatch}}) where {dispatch}
    return CompactLuxLayer{dispatch}
end

function initialparameters(rng::AbstractRNG, m::CompactLuxLayer)
    return (; initialparameters(rng, m.layers)...,
        initialparameters(rng, m.value_storage)...)
end

function initialstates(rng::AbstractRNG, m::CompactLuxLayer)
    return (; initialstates(rng, m.layers)..., initialstates(rng, m.value_storage)...)
end

function __try_make_lux_layer(x::Union{AbstractVector, Tuple})
    return __try_make_lux_layer(NamedTuple{Tuple(Symbol.(1:length(x)))}(x))
end
function __try_make_lux_layer(x)
    function __maybe_convert_layer(l)
        l isa AbstractExplicitLayer && return l
        l isa Function && return WrappedFunction(l)
        return l
    end
    return fmap(__maybe_convert_layer, x)
end

function CompactLuxLayer{dispatch}(f::Function, name::NAME_TYPE, str::Tuple,
        setup_str::NamedTuple; kws...) where {dispatch}
    layers, others = [], []
    for (name, val) in pairs(kws)
        if val isa AbstractExplicitLayer
            push!(layers, name => val)
        elseif LuxCore.contains_lux_layer(val)
            # TODO: Rearrange Tuple and Vectors to NamedTuples for proper CA.jl support
            # FIXME: This might lead to incorrect constructions? If the function is a closure over the provided keyword arguments?
            val = __try_make_lux_layer(val)
            if LuxCore.check_fmap_condition(!Base.Fix2(isa, AbstractExplicitLayer),
                nothing, val)
                throw(LuxCompactModelParsingException("A container `$(name) = $(val)` is found which combines Lux layers with non-Lux layers. This is not supported."))
            end
            push!(layers, name => val)
        else
            push!(others, name => val)
        end
    end
    return CompactLuxLayer{dispatch}(f, name, str, setup_str, NamedTuple((; layers...)),
        ValueStorage(; others...))
end

function (m::CompactLuxLayer)(x, ps, st::NamedTuple{fields}) where {fields}
    y = m.f(m.layers, x, ps, st)
    st_ = NamedTuple{fields}((getfield.((st,), fields)...,))
    return y, st_
end

# Shortcut for potential chain rules bug?
function (m::CompactLuxLayer)(x, ps, st::NamedTuple{()})
    y = m.f(m.layers, x, ps, st)
    return y, st
end

# Pretty printing the layer code
function Lux._big_show(io::IO, obj::CompactLuxLayer, indent::Int=0, name=nothing)
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
