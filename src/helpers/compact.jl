# This functionality is based off of the implementation in Fluxperimental.jl
# https://github.com/FluxML/Fluxperimental.jl/blob/cc0e36fdd542cc6028bc69449645dc0390dd980b/src/compact.jl
"""
    @compact(kw...) do x
        ...
        @return y # optional (but recommended for best performance)
    end
    @compact(kw...) do x, p
        ...
        @return y # optional (but recommended for best performance)
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
 2. `dispatch`: The constructed layer has the type `Lux.CompactLuxLayer{dispatch}` which can
    be used for custom dispatches.

!!! tip

    Check the Lux tutorials for more examples of using `@compact`.

If you are passing in kwargs by splatting them, they will be passed as is to the function
body. This means if your splatted kwargs contain a lux layer that won't be registered
in the CompactLuxLayer. Additionally all of the device functions treat these kwargs as
leaves.

## Special Syntax

  - `@return`: This macro doesn't really exist, but is used to return a value from the
    `@compact` block. Without the presence of this macro, we need to rely on closures which
    can lead to performance penalties in the reverse pass.

      + Having statements after the last `@return` macro might lead to incorrect code.
      + Don't do things like `@return return x`. This will generate non-sensical code like
        `<new var> = return x`. Essentially, `@return <expr>` supports any expression, that
        can be assigned to a variable.
      + Since this macro doesn't "exist", it cannot be imported as `using Lux: @return`.
        Simply use it in code, and `@compact` will understand it.

  - `@init_fn`: Provide a function that will be used to initialize the layer's parameters or
    state. See the docs of [`@init_fn`](@ref) for more details.
  - `@non_trainable`: Mark a value as non-trainable. This bypasses the regular checks and
    places the value into the state of the layer. See the docs of [`@non_trainable`](@ref)
    for more details.

# Extended Help

## Examples

Here is a linear model:

```jldoctest
julia> using Lux, Random

julia> r = @compact(w=ones(Float32, 3)) do x
           @return w .* x
       end
@compact(
    w = 3-element Vector{Float32},
) do x
    return w .* x
end       # Total: 3 parameters,
          #        plus 0 states.

julia> ps, st = Lux.setup(Xoshiro(0), r);

julia> r(Float32[1, 2, 3], ps, st)  # x is set to [1, 1, 1].
(Float32[1.0, 2.0, 3.0], NamedTuple())
```

Here is a linear model with bias and activation:

```jldoctest
julia> d_in = 5
5

julia> d_out = 3
3

julia> d = @compact(W=ones(Float32, d_out, d_in), b=zeros(Float32, d_out), act=relu) do x
           y = W * x
           @return act.(y .+ b)
       end
@compact(
    W = 3×5 Matrix{Float32},
    b = 3-element Vector{Float32},
    act = relu,
) do x
    y = W * x
    return act.(y .+ b)
end       # Total: 18 parameters,
          #        plus 1 states.

julia> ps, st = Lux.setup(Xoshiro(0), d);

julia> d(ones(Float32, 5, 2), ps, st)[1] # 3×2 Matrix as output.
3×2 Matrix{Float32}:
 5.0  5.0
 5.0  5.0
 5.0  5.0

julia> ps_dense = (; weight=ps.W, bias=ps.b);

julia> first(d(Float32[1, 2, 3, 4, 5], ps, st)) ≈
       first(Dense(d_in => d_out, relu)(Float32[1, 2, 3, 4, 5], ps_dense, NamedTuple())) # Equivalent to a dense layer
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
           @return out
       end
@compact(
    w1 = Dense(1 => 128),                         # 256 parameters
    w2 = NamedTuple(
        (1-3) = Dense(128 => 128),                # 49_536 (16_512 x 3) parameters
    ),
    w3 = Dense(128 => 1),                         # 129 parameters
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

julia> size(first(model(randn(Float32, n_in, 32), ps, st)))  # 1×32 Matrix as output.
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
julia> model = @compact(w=rand(Float32, 3), name="Linear(3 => 1)") do x
           @return sum(w .* x)
       end
Linear(3 => 1)               # 3 parameters
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
    return CompactMacroImpl.compact_macro_impl(__source__, __module__, _exs...)
end

"""
    @init_fn(fn, kind::Symbol = :parameter)

Create an initializer function for a parameter or state to be used for in a Compact Lux
Layer created using [`@compact`](@ref).

## Arguments

  - `fn`: The function to be used for initializing the parameter or state. This only takes
    a single argument `rng`.
  - `kind`: If set to `:parameter`, the initializer function will be used to initialize the
    parameters of the layer. If set to `:state`, the initializer function will be used to
    initialize the states of the layer.

## Examples

```jldoctest
julia> using Lux, Random

julia> r = @compact(w=@init_fn(rng->randn32(rng, 3, 2)),
           b=@init_fn(rng->randn32(rng, 3), :state)) do x
           @return w * x .+ b
       end;

julia> ps, st = Lux.setup(Xoshiro(0), r);

julia> size(ps.w)
(3, 2)

julia> size(st.b)
(3,)

julia> size(r([1, 2], ps, st)[1])
(3,)
```
"""
macro init_fn(args...)
    @assert 1 ≤ length(args) ≤ 2
    kind = length(args) == 1 ? :parameter : args[2]
    kind isa QuoteNode && (kind = kind.value)
    fn = args[1]
    @assert kind in (:parameter, :state)
    return esc(:($(CompactMacroImpl.InitFn){$(Meta.quot(kind))}($(fn))))
end

"""
    @non_trainable(x)

Mark a value as non-trainable. This bypasses the regular checks and places the value into
the state of the layer.

## Arguments

  - `x`: The value to be marked as non-trainable.

## Examples

```jldoctest
julia> using Lux, Random

julia> r = @compact(w=ones(3), w_fixed=@non_trainable(rand(3))) do x
           @return sum(w .* x .+ w_fixed)
       end;

julia> ps, st = Lux.setup(Xoshiro(0), r);

julia> size(ps.w)
(3,)

julia> size(st.w_fixed)
(3,)

julia> res, st_ = r([1, 2, 3], ps, st);

julia> st_.w_fixed == st.w_fixed
true

julia> res isa Number
true
```
"""
macro non_trainable(x)
    return esc(:($(CompactMacroImpl.NonTrainable)($(x))))
end

struct CompactLuxLayer{dispatch,F,N,L,V,SK} <:
       AbstractLuxContainerLayer{(:layers, :value_storage)}
    d::StaticSymbol{dispatch}
    f::F
    name::N
    strings::NTuple{3,String}
    setup_strings::Any
    layers::L
    value_storage::V
    stored_kwargs::SK
end

function initialparameters(rng::AbstractRNG, m::CompactLuxLayer)
    return (;
        initialparameters(rng, m.layers)..., initialparameters(rng, m.value_storage)...
    )
end

function initialstates(rng::AbstractRNG, m::CompactLuxLayer)
    base_states = (;
        initialstates(rng, m.layers)..., initialstates(rng, m.value_storage)...
    )
    length(first(m.stored_kwargs)) == 0 && return base_states
    return merge(
        base_states,
        (;
            ₋₋₋kwargs₋₋₋=CompactMacroImpl.KwargsStorage(
                NamedTuple{m.stored_kwargs[1]}(m.stored_kwargs[2])
            ),
        ),
    )
end

function CompactLuxLayer(
    dispatch::StaticSymbol, f::F, name::NAME_TYPE, str::Tuple, splatted_kwargs; kws...
) where {F}
    layers, others = [], []
    setup_strings = NamedTuple()
    for (name, val) in pairs(kws)
        is_lux_layer = false
        if val isa AbstractLuxLayer
            is_lux_layer = true
            push!(layers, name => val)
        elseif LuxCore.contains_lux_layer(val)
            # FIXME: This might lead to incorrect constructions? If the function is a
            #        closure over the provided keyword arguments?
            val = CompactMacroImpl.try_make_lux_layer(val)
            if LuxCore.check_fmap_condition(!Base.Fix2(isa, AbstractLuxLayer), nothing, val)
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
                setup_strings, NamedTuple((name => CompactMacroImpl.kwarg_descriptor(val),))
            )
        end
    end
    return CompactLuxLayer(
        dispatch,
        f,
        name,
        str,
        setup_strings,
        NamedTuple((; layers...)),
        CompactMacroImpl.ValueStorage(; others...),
        splatted_kwargs,
    )
end

@generated function (m::CompactLuxLayer)(x, ps, st::NamedTuple{fields}) where {fields}
    st_expr = [
        :($(LuxCore.StatefulLuxLayerImpl.get_state)(st_new.$(field))) for
        field in fields if field != :₋₋₋kwargs₋₋₋
    ]
    st_expr = :(NamedTuple{$(filter(f -> f != :₋₋₋kwargs₋₋₋, fields))}(($(st_expr...),)))
    st_expr = :(merge(st, $st_expr))
    return quote
        y, st_new = m.f(m.layers, x, ps, CRC.ignore_derivatives(st))
        return y, $(st_expr)
    end
end

# Shortcut for potential chain rules bug?
function (m::CompactLuxLayer)(x, ps, st::NamedTuple{()})
    y, _ = m.f(m.layers, x, ps, st)
    return y, st
end

# Pretty printing the layer code
function PrettyPrinting.big_show(io::IO, obj::CompactLuxLayer, indent::Int=0, name=nothing)
    setup_strings = obj.setup_strings
    local_name = obj.name
    if local_name !== nothing && local_name != ""
        _str = name === nothing ? "" : "$name = "
        str = _str * local_name
        print(io, " "^indent, str, indent == 0 ? "" : ",")
        PrettyPrinting.show_parameters_count(io, obj, indent, length(str))
        indent == 0 || println(io)
        return nothing
    end
    layer, input, block = obj.strings
    pre, post = ("(", ")")
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", layer, pre)
    for (k, v) in pairs(setup_strings)
        val = safe_getproperty(obj.layers, Val(k))
        if val === nothing
            println(io, " "^(indent + 4), "$k = $v,")
        else
            PrettyPrinting.big_show(io, val, indent + 4, k)
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
        PrettyPrinting.big_finale(io, obj, 7)
    else
        println(io, ",")
    end
    return nothing
end

module CompactMacroImpl

using ConcreteStructs: @concrete
using MacroTools: MacroTools, combinedef, splitdef
using Functors: Functors
using Random: AbstractRNG
using Static: static

using LuxCore: LuxCore, AbstractLuxLayer
using ..Lux:
    Lux,
    CompactLuxLayer,
    LuxCompactModelParsingException,
    StatefulLuxLayer,
    safe_getproperty

function compact_macro_impl(__source__, __module__, _exs...)
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
    name, kwexs = extract_reserved_kwarg(kwexs, :name)

    # check if user has provided a custom dispatch
    dispatch, kwexs = extract_reserved_kwarg(kwexs, :dispatch)
    dispatch === nothing && (dispatch = QuoteNode(:₋₋₋no_special_dispatch₋₋₋))

    # Extract splatted kwargs
    splat_idxs = findall(ex -> ex.head == :..., kwexs)
    splatted_kwargs = map(first ∘ Base.Fix2(getproperty, :args), kwexs[splat_idxs])
    kwexs = filter(ex -> ex.head != :..., kwexs)

    # make strings
    layer = "@compact"
    input = try
        fex_args = fex.args[1]
        isa(fex_args, Symbol) ? string(fex_args) : join(fex_args.args, ", ")
    catch _
        @warn "Function stringifying does not yet handle all cases. Falling back to empty \
               string for input arguments"
    end

    # Remove compact specific macros
    fex_clean =
        MacroTools.postwalk(MacroTools.striplines(Base.remove_linenums!(fex).args[2])) do x
            MacroTools.@capture(x, @return val_) && return :(return $val)
            return x
        end
    block = string(fex_clean)

    # edit expressions
    vars = map(first ∘ Base.Fix2(getproperty, :args), kwexs)
    fex = supportself(fex, vars, splatted_kwargs, __source__)

    # assemble
    return esc(
        :($CompactLuxLayer(
            $(static)($(dispatch)),
            $(fex),
            $(name),
            ($layer, $input, $block),
            (($(Meta.quot.(splatted_kwargs)...),), ($(splatted_kwargs...),));
            $(kwexs...),
        )),
    )
end

function supportself(fex::Expr, vars, splatted_kwargs, __source__)
    @gensym self ps st res
    # To avoid having to manipulate fex's arguments and body explicitly, we split the input
    # function body and add the required arguments to the function definition.
    sdef = splitdef(fex)
    custom_param = length(sdef[:args]) == 2
    length(sdef[:args]) > 2 &&
        throw(LuxCompactModelParsingException("expects at most 2 arguments"))
    args = [self, sdef[:args][1], ps, st]
    calls = []
    for var in vars
        push!(
            calls,
            :(
                $var = $(maybe_make_stateful)(
                    $(safe_getproperty)($self, $(Val(var))),
                    $(safe_getproperty)($ps, $(Val(var))),
                    $(safe_getproperty)($st, $(Val(var))),
                )
            ),
        )
    end
    for var in splatted_kwargs
        push!(
            calls,
            :(
                $var = $(safe_getproperty)(
                    getproperty(getproperty($st, :₋₋₋kwargs₋₋₋), :kws), $(Val(var))
                )
            ),
        )
    end
    custom_param && push!(calls, :($(sdef[:args][2]) = $ps))

    # Try to generate efficient code for the function body
    has_return_macro = false
    flattened_expr = MacroTools.postwalk(sdef[:body]) do x
        if MacroTools.@capture(x, @return val_)
            @gensym result
            has_return_macro = true
            return quote
                $(result) = $(val)
                return $(result), (; $(vars...))
            end
        end
        if has_return_macro && MacroTools.@capture(x, return val_)
            throw(LuxCompactModelParsingException("Encountered a return statement \
                                                   after the last @return statement. \
                                                   This is not supported."))
        end
        return x
    end

    if !has_return_macro
        @gensym fname
        @warn "No @return macro found in the function body. This will lead to the \
               generation of inefficient code."
        modified_body = quote
            $fname = () -> $(sdef[:body])
            $res = $(fname)()
            return $(res), (; $(vars...))
        end
    else
        modified_body = flattened_expr
    end

    modified_body = MacroTools.to_line(__source__, MacroTools.to_flag(modified_body))

    sdef[:body] = Expr(:let, Expr(:block, calls...), modified_body)
    sdef[:args] = args
    return combinedef(sdef)
end

function extract_reserved_kwarg(kwexs, sym::Symbol)
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

@concrete struct NonTrainable
    value
end

@concrete struct InitFn{kind} <: Function
    f <: Function
end

(f::InitFn)(args...) = f.f(args...)

@concrete struct ValueStorage <: AbstractLuxLayer
    ps_init_fns
    st_init_fns
end

function ValueStorage(; kwargs...)
    ps_init_fns, st_init_fns = [], []
    for (key, val) in pairs(kwargs)
        list, store_val = if val isa NonTrainable
            st_init_fns, Returns(val.value)
        elseif val isa AbstractVector{<:AbstractArray{<:Number}}
            ps_init_fns, Returns(val)
        elseif val isa AbstractArray
            if (isbitstype(eltype(val)) || eltype(val) <: Number)
                ps_init_fns, Returns(val)
            else
                st_init_fns, Returns(val)
            end
        elseif val isa NTuple{N,<:AbstractArray{<:Number}} where {N}
            ps_init_fns, Returns(val)
        elseif val isa InitFn{:state}
            st_init_fns, val
        elseif val isa InitFn{:parameter}
            ps_init_fns, val
        else
            st_init_fns, Returns(val)
        end
        push!(list, key => store_val)
    end
    return ValueStorage(NamedTuple(ps_init_fns), NamedTuple(st_init_fns))
end

function (::ValueStorage)(_, _, _)
    throw(ArgumentError("`ValueStorage` isn't meant to be used as a layer!!!"))
end

function LuxCore.initialparameters(rng::AbstractRNG, v::ValueStorage)
    return NamedTuple([
        n => (fn isa InitFn ? fn(rng) : fn()) for (n, fn) in pairs(v.ps_init_fns)
    ])
end

function LuxCore.initialstates(rng::AbstractRNG, v::ValueStorage)
    return NamedTuple([
        n => (fn isa InitFn ? fn(rng) : fn()) for (n, fn) in pairs(v.st_init_fns)
    ])
end

@concrete struct KwargsStorage
    kws <: NamedTuple
end

Functors.@leaf KwargsStorage

function kwarg_descriptor(val)
    val isa NonTrainable && return "@non_trainable($(kwarg_descriptor(val.value)))"
    val isa Number && return string(val)
    val isa AbstractArray && return sprint(Base.array_summary, val, axes(val))
    val isa Tuple && return "(" * join(map(kwarg_descriptor, val), ", ") * ")"
    val isa InitFn{:parameter} && return "@init_fn($(kwarg_descriptor(val.f)), parameter)"
    val isa InitFn{:state} && return "@init_fn($(kwarg_descriptor(val.f)), state)"
    val isa Nothing && return "nothing"
    if val isa NamedTuple
        fields = fieldnames(typeof(val))
        strs = []
        for fname in fields[1:min(length(fields), 3)]
            internal_val = getfield(val, fname)
            push!(strs, "$fname = $(kwarg_descriptor(internal_val))")
        end
        return "@NamedTuple{$(join(strs, ", "))" * (length(fields) > 3 ? ", ..." : "") * "}"
    end
    val isa Function && return sprint(show, val; context=(:compact => true, :limit => true))
    return "$(nameof(typeof(val)))(...)"
end

function try_make_lux_layer(x::Union{AbstractVector,Tuple})
    return try_make_lux_layer(NamedTuple{Tuple(Symbol.(1:length(x)))}(x))
end
try_make_lux_layer(x) = x

maybe_make_stateful(layer::AbstractLuxLayer, ps, st) = StatefulLuxLayer(layer, ps, st)
maybe_make_stateful(::Nothing, ::Nothing, st) = st
maybe_make_stateful(::Nothing, ps, _) = ps
function maybe_make_stateful(model::Union{AbstractVector,Tuple}, ps, st)
    return map(i -> maybe_make_stateful(model[i], ps[i], st[i]), eachindex(model))
end
function maybe_make_stateful(model::NamedTuple{fields}, ps, st) where {fields}
    return NamedTuple{fields}(
        map(
            f -> maybe_make_stateful(
                getproperty(model, f), getproperty(ps, f), getproperty(st, f)
            ),
            fields,
        ),
    )
end

end
