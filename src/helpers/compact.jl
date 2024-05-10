# This functionality is based off of the implementation in Fluxperimental.jl
# https://github.com/FluxML/Fluxperimental.jl/blob/cc0e36fdd542cc6028bc69449645dc0390dd980b/src/compact.jl
struct LuxCompactModelParsingException <: Exception
    msg::String
end

function Base.showerror(io::IO, e::LuxCompactModelParsingException)
    print(io, "LuxCompactModelParsingException(", e.msg, ")")
    return
end

"""
    @compact(kw...) do x
        ...
    end
    @compact(kw...) do x, p
        ...
    end
    @compact(forward::Function; name=nothing, dispatch=nothing, parameters...)

Creates a layer by specifying some `parameters`, in the form of keywords, and (usually as a
`do` block) a function for the forward pass. You may think of `@compact` as a specialized
`let` block creating local variables that are trainable in Lux. Declared variable names may
be used within the body of the `forward` function. Note that unlike typical Lux models, the
forward function doesn't need to explicitly manage states.

Defining the version with `p` allows you to access the parameters in the forward pass. This
is useful when using it with SciML tools which require passing in the parameters explicitly.

## Reserved Kwargs:

 1. `name`: The name of the layer.
 2. `dispatch`: The constructed layer has the type
    `Lux.Experimental.CompactLuxLayer{dispatch}` which can be used for custom dispatches.

!!! tip

    Check the Lux tutorials for more examples of using `@compact`.

If you are passing in kwargs by splatting them, they will be passed as is to the function
body. This means if your splatted kwargs contain a lux layer that won't be registered
in the CompactLuxLayer.

## Examples

Here is a linear model:

```jldoctest
julia> using Lux, Random

julia> r = @compact(w=ones(3)) do x
           return w .* x
       end
@compact(
    w = 3-element Vector{Float64},
) do x
    return w .* x
end       # Total: 3 parameters,
          #        plus 0 states.

julia> ps, st = Lux.setup(Xoshiro(0), r);

julia> r([1, 2, 3], ps, st)  # x is set to [1, 1, 1].
([1.0, 2.0, 3.0], NamedTuple())
```

Here is a linear model with bias and activation:

```jldoctest
julia> d_in = 5
5

julia> d_out = 3
3

julia> d = @compact(W=ones(d_out, d_in), b=zeros(d_out), act=relu) do x
           y = W * x
           return act.(y .+ b)
       end
@compact(
    W = 3×5 Matrix{Float64},
    b = 3-element Vector{Float64},
    act = relu,
) do x
    y = W * x
    return act.(y .+ b)
end       # Total: 18 parameters,
          #        plus 1 states.

julia> ps, st = Lux.setup(Xoshiro(0), d);

julia> d(ones(5, 2), ps, st)[1] # 3×2 Matrix as output.
3×2 Matrix{Float64}:
 5.0  5.0
 5.0  5.0
 5.0  5.0

julia> ps_dense = (; weight=ps.W, bias=ps.b);

julia> first(d([1, 2, 3, 4, 5], ps, st)) ≈
       first(Dense(d_in => d_out, relu)([1, 2, 3, 4, 5], ps_dense, NamedTuple())) # Equivalent to a dense layer
true
```

Finally, here is a simple MLP. We can train this model just like any Lux model:

```jldoctest
julia> n_in = 1;

julia> n_out = 1;

julia> nlayers = 3;

julia> model = @compact(w1=Dense(n_in, 128),
           w2=[Dense(128, 128) for i in 1:nlayers], w3=Dense(128, n_out), act=relu) do x
           embed = act.(w1(x))
           for w in w2
               embed = act.(w(embed))
           end
           out = w3(embed)
           return out
       end
@compact(
    w1 = Dense(1 => 128),               # 256 parameters
    w2 = NamedTuple(
        1 = Dense(128 => 128),          # 16_512 parameters
        2 = Dense(128 => 128),          # 16_512 parameters
        3 = Dense(128 => 128),          # 16_512 parameters
    ),
    w3 = Dense(128 => 1),               # 129 parameters
    act = relu,
) do x
    embed = act.(w1(x))
    for w = w2
        embed = act.(w(embed))
    end
    out = w3(embed)
    return out
end       # Total: 49_921 parameters,
          #        plus 1 states.

julia> ps, st = Lux.setup(Xoshiro(0), model);

julia> size(first(model(randn(n_in, 32), ps, st)))  # 1×32 Matrix as output.
(1, 32)

julia> using Optimisers, Zygote

julia> x_data = collect(-2.0f0:0.1f0:2.0f0)';

julia> y_data = 2 .* x_data .- x_data .^ 3;

julia> optim = Optimisers.setup(Adam(), ps);

julia> loss_initial = sum(abs2, first(model(x_data, ps, st)) .- y_data);

julia> for epoch in 1:1000
           loss, gs = Zygote.withgradient(
               ps -> sum(abs2, first(model(x_data, ps, st)) .- y_data), ps)
           Optimisers.update!(optim, ps, gs[1])
       end;

julia> loss_final = sum(abs2, first(model(x_data, ps, st)) .- y_data);

julia> loss_initial > loss_final
true
```

You may also specify a `name` for the model, which will be used instead of the default
printout, which gives a verbatim representation of the code used to construct the model:

```jldoctest
julia> model = @compact(w=rand(3), name="Linear(3 => 1)") do x
           return sum(w .* x)
       end
Linear(3 => 1)()    # 3 parameters

julia> println(model)
Linear(3 => 1)()
```

This can be useful when using `@compact` to hierarchically construct complex models to be
used inside a `Chain`.

!!! tip "Type Stability"

    If your input function `f` is type-stable but the generated model is not type stable, it
    should be treated as a bug. We will appreciate issues if you find such cases.

!!! warning "Parameter Count"

    Array Parameter don't print the number of parameters on the side. However, they do
    account for the total number of parameters printed at the bottom.
"""
macro compact(_exs...)
    return __compact_macro_impl(_exs...)
end

# Needed for the deprecation path
function __compact_macro_impl(_exs...)
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
    name, kwexs = __extract_reserved_kwarg(kwexs, :name)

    # check if user has provided a custom dispatch
    dispatch, kwexs = __extract_reserved_kwarg(kwexs, :dispatch)

    # Extract splatted kwargs
    splat_idxs = findall(ex -> ex.head == :..., kwexs)
    splatted_kwargs = map(first ∘ Base.Fix2(getproperty, :args), kwexs[splat_idxs])
    kwexs = filter(ex -> ex.head != :..., kwexs)

    # make strings
    layer = "@compact"
    input = try
        fex_args = fex.args[1]
        isa(fex_args, Symbol) ? string(fex_args) : join(fex_args.args, ", ")
    catch e
        @warn "Function stringifying does not yet handle all cases. Falling back to empty \
               string for input arguments"
    end
    block = string(Base.remove_linenums!(fex).args[2])

    # edit expressions
    vars = map(first ∘ Base.Fix2(getproperty, :args), kwexs)
    fex = supportself(fex, vars, splatted_kwargs)

    # assemble
    return esc(:($CompactLuxLayer{$dispatch}($fex, $name, ($layer, $input, $block),
        (($(Meta.quot.(splatted_kwargs)...),), ($(splatted_kwargs...),)); $(kwexs...))))
end

function __extract_reserved_kwarg(kwexs, sym::Symbol)
    idx = findfirst(ex -> ex.args[1] == sym, kwexs)
    val = nothing
    if idx !== nothing && kwexs[idx].args[2] !== nothing
        length(kwexs) == 1 &&
            throw(LuxCompactModelParsingException("expects keyword arguments"))
        val = kwexs[idx].args[2]
        kwexs = (kwexs[1:(idx - 1)]..., kwexs[(idx + 1):end]...)
    end
    return val, kwexs
end

function supportself(fex::Expr, vars, splatted_kwargs)
    @gensym self ps st curried_f res
    # To avoid having to manipulate fex's arguments and body explicitly, we split the input
    # function body and add the required arguments to the function definition.
    sdef = splitdef(fex)
    custom_param = length(sdef[:args]) == 2
    length(sdef[:args]) > 2 &&
        throw(LuxCompactModelParsingException("expects at most 2 arguments"))
    args = [self, sdef[:args][1], ps, st]
    calls = []
    for var in vars
        push!(calls,
            :($var = $(__maybe_make_stateful)($(_getproperty)($self, $(Val(var))),
                $(_getproperty)($ps, $(Val(var))), $(_getproperty)($st, $(Val(var))))))
    end
    for var in splatted_kwargs
        push!(
            calls, :($var = $(_getproperty)(getproperty($st, :₋₋₋kwargs₋₋₋), $(Val(var)))))
    end
    custom_param && push!(calls, :($(sdef[:args][2]) = $ps))
    body = Expr(:let, Expr(:block, calls...), sdef[:body])
    sdef[:body] = body
    sdef[:args] = args
    return combinedef(sdef)
end

@inline function __maybe_make_stateful(layer::AbstractExplicitLayer, ps, st)
    return StatefulLuxLayer(layer, ps, st)
end
@inline __maybe_make_stateful(::Nothing, ps, st) = ifelse(ps === nothing, st, ps)
@inline function __maybe_make_stateful(model::Union{AbstractVector, Tuple}, ps, st)
    return map(i -> __maybe_make_stateful(model[i], ps[i], st[i]), eachindex(model))
end
@inline function __maybe_make_stateful(model::NamedTuple{fields}, ps, st) where {fields}
    return NamedTuple{fields}(map(
        f -> __maybe_make_stateful(
            getproperty(model, f), getproperty(ps, f), getproperty(st, f)),
        fields))
end

@concrete struct ValueStorage <: AbstractExplicitLayer
    ps_init_fns
    st_init_fns
end

function ValueStorage(; kwargs...)
    ps_init_fns, st_init_fns = [], []
    for (key, val) in pairs(kwargs)
        push!(val isa AbstractArray ? ps_init_fns : st_init_fns, key => Returns(val))
    end
    return ValueStorage(NamedTuple(ps_init_fns), NamedTuple(st_init_fns))
end

function (v::ValueStorage)(x, ps, st)
    throw(ArgumentError("`ValueStorage` isn't meant to be used as a layer!!!"))
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
    stored_kwargs
end

function ConstructionBase.constructorof(::Type{<:CompactLuxLayer{dispatch}}) where {dispatch}
    return CompactLuxLayer{dispatch}
end

function initialparameters(rng::AbstractRNG, m::CompactLuxLayer)
    return (;
        initialparameters(rng, m.layers)..., initialparameters(rng, m.value_storage)...)
end

function initialstates(rng::AbstractRNG, m::CompactLuxLayer)
    base_states = (;
        initialstates(rng, m.layers)..., initialstates(rng, m.value_storage)...)
    length(first(m.stored_kwargs)) == 0 && return base_states
    return merge(
        base_states, (; ₋₋₋kwargs₋₋₋=NamedTuple{m.stored_kwargs[1]}(m.stored_kwargs[2])))
end

function __try_make_lux_layer(x::Union{AbstractVector, Tuple})
    return __try_make_lux_layer(NamedTuple{Tuple(Symbol.(1:length(x)))}(x))
end
function __try_make_lux_layer(x)
    __maybe_convert_layer = @closure l -> begin
        l isa AbstractExplicitLayer && return l
        l isa Function && return WrappedFunction(l)
        return l
    end
    return fmap(__maybe_convert_layer, x)
end

function CompactLuxLayer{dispatch}(
        f::F, name::NAME_TYPE, str::Tuple, splatted_kwargs; kws...) where {F, dispatch}
    layers, others = [], []
    setup_strings = NamedTuple()
    for (name, val) in pairs(kws)
        is_lux_layer = false
        if val isa AbstractExplicitLayer
            is_lux_layer = true
            push!(layers, name => val)
        elseif LuxCore.contains_lux_layer(val)
            # TODO: Rearrange Tuple and Vectors to NamedTuples for proper CA.jl support
            # FIXME: This might lead to incorrect constructions? If the function is a
            #        closure over the provided keyword arguments?
            val = __try_make_lux_layer(val)
            if LuxCore.check_fmap_condition(
                !Base.Fix2(isa, AbstractExplicitLayer), nothing, val)
                throw(LuxCompactModelParsingException("A container `$(name) = $(val)` is \
                                                       found which combines Lux layers \
                                                       with non-Lux layers. This is not \
                                                       supported."))
            end
            is_lux_layer = true
            push!(layers, name => val)
        else
            push!(others, name => val)
        end

        if is_lux_layer
            setup_strings = merge(setup_strings, NamedTuple((name => val,)))
        else
            setup_strings = merge(
                setup_strings, NamedTuple((name => __kwarg_descriptor(val),)))
        end
    end

    return CompactLuxLayer{dispatch}(f, name, str, setup_strings, NamedTuple((; layers...)),
        ValueStorage(; others...), splatted_kwargs)
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

function __kwarg_descriptor(val)
    val isa Number && return string(val)
    val isa AbstractArray && return sprint(Base.array_summary, val, axes(val))
    val isa Tuple && return "(" * join(map(__kwarg_descriptor, val), ", ") * ")"
    val isa Nothing && return "nothing"
    if val isa NamedTuple
        fields = fieldnames(typeof(val))
        strs = []
        for fname in fields[1:min(length(fields), 3)]
            internal_val = getfield(val, fname)
            push!(strs, "$fname = $(__kwarg_descriptor(internal_val))")
        end
        return "@NamedTuple{$(join(strs, ", "))" * (length(fields) > 3 ? ", ..." : "") * "}"
    end
    val isa Function && return sprint(show, val; context=(:compact => true, :limit => true))
    return lazy"$(nameof(typeof(val)))(...)"
end
