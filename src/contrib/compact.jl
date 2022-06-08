using MacroTools

"""
An Experimental compact API to make it easier to write Lux models.

## Example

```julia
using Lux, Random, Zygote, NNlib

Lux.@compact function MyCustomModel((x, y)::T) where {T}
    @assert size(x, 1) == 2 && size(y, 1) == 2
    a = Dense(3, 3, relu)(Dense(2, 3)(x) .+ Dense(2, 3, relu)(x)) .- Dense(2, 3, gelu)(x)
    b = Dense(3, 2, sigmoid)(a)
    c = Dense(2, 2, relu)(y)
    return b .+ c
end

model = MyCustomModel()

ps, st = Lux.setup(Random.default_rng(), model)
x, y = randn(Float32, 2, 1), randn(Float32, 2, 1)

model((x, y), ps, st)
```

## Verifying Generated Code

!!! tip
    Since this is an highly experimental feature, we recommend always manually verifying the generated model. This can be done simply using `@macroexpand Lux.@compact ...`

```julia-repl
julia> @macroexpand Lux.@compact function MyCustomModel((x, y)::T) where {T}
           @assert size(x, 1) == 2 && size(y, 1) == 2
           a = Dense(3, 3, relu)(Dense(2, 3)(x) .+ Dense(2, 3, relu)(x)) .- Dense(2, 3, gelu)(x)
           b = Dense(3, 2, sigmoid)(a)
           c = Dense(2, 2, relu)(y)
           return b .+ c
       end
quote
    struct MyCustomModel{__dense_1_type, __dense_2_type, __dense_3_type, __dense_4_type, __dense_5_type, __dense_6_type} <: Lux.AbstractExplicitContainerLayer{(:dense_1, :dense_2, :dense_3, :dense_4, :dense_5, :dense_6)}
        dense_1::__dense_1_type
        dense_2::__dense_2_type
        dense_3::__dense_3_type
        dense_4::__dense_4_type
        dense_5::__dense_5_type
        dense_6::__dense_6_type
    end
    function MyCustomModel(; )
        return MyCustomModel(Dense(3, 3, relu), Dense(2, 3), Dense(2, 3, relu), Dense(2, 3, gelu), Dense(3, 2, sigmoid), Dense(2, 2, relu))
    end
    function (mandrill::MyCustomModel)((x, y)::T, ps::Union{Lux.ComponentArray, NamedTuple}, st::NamedTuple; ) where T
        if size(x, 1) == 2 && size(y, 1) == 2
            nothing
        else
            Base.throw(Base.AssertionError("size(x, 1) == 2 && size(y, 1) == 2"))
        end
        (wildebeest, camel) = mandrill.dense_4(x, ps.dense_4, st.dense_4)
        (antelope, gaur) = mandrill.dense_3(x, ps.dense_3, st.dense_3)
        (rhinoceros, ibis) = mandrill.dense_2(x, ps.dense_2, st.dense_2)
        (mallard, jellyfish) = mandrill.dense_1(rhinoceros .+ antelope, ps.dense_1, st.dense_1)
        a = mallard .- wildebeest
        (hamster, monkey) = mandrill.dense_5(a, ps.dense_5, st.dense_5)
        b = hamster
        (wasp, sanddollar) = mandrill.dense_6(y, ps.dense_6, st.dense_6)
        c = wasp
        return (b .+ c, NamedTuple{(:dense_1, :dense_2, :dense_3, :dense_4, :dense_5, :dense_6)}((jellyfish, ibis, gaur, camel, monkey, sanddollar)))
    end
end
```

## What not to do while writing `compact` models?

Here are some common cases where `@compact` will silently fail to generate the correct code. (Encountered some additional cases, open an issue so that we can solve it or expand this list)

1. **Imcomplete Layer Calls**: All layer calls must be complete, i.e. the following is not allowed:

```julia
Lux.@compact function ThisModelDoesNotWork(x)
    @assert size(x, 1) == 2 && size(y, 1) == 2
    shared = Dense(2, 3, gelu)
    a = Dense(3, 3, relu)(shared(x) .+ Dense(2, 3, relu)(x)) .- shared(x)
    b = Dense(3, 2, sigmoid)(a)
    return b
end
```

Instead write it as:

```julia
Lux.@compact function ThisModelWorks(x)
    @assert size(x, 1) == 2 && size(y, 1) == 2
    shared = Dense(2, 3, gelu)(x)
    a = Dense(3, 3, relu)(shared .+ Dense(2, 3, relu)(x)) .- shared
    b = Dense(3, 2, sigmoid)(a)
    return b
end
```

Unfortunately, this means currently we cannot invoke the same model twice.

2. **Creating layers without `=`**: All layer definitions must be preceeded by `=`. Anything else be it `return`, ".+=". ".=", etc. is not allowed.

```julia
# Invalid
Lux.@compact function WrongReturn(x)
    return Dense(2, 2)(x)
end

Lux.@compact WrongOneLiner(x) = Dense(2, 2)(x)

# Valid
Lux.@compact function CorrectReturn(x)
    x = Dense(2, 2)(x)
    return x
end
```

!!! note
    This is heavily inspired by [flax.linen.compact](https://flax.readthedocs.io/en/latest/design_notes/module_lifecycle.html?highlight=compact#compact-modules). Unlike Flax, in Lux.@compact we still need to use the complete API to define the model. We don't perform automatic shape inference at this point. (This is a planned feature for future releases). Additionally, we don't allow specifying names for each layer.

!!! warning
    Currently `whichparams` are completely ignored. (This is a planned feature for future releases)

!!! warning
    This is an experimental API. It is likely to have bugs and untested edge cases. If you find the generated code to be incorrect, please file an issue.
"""
macro compact(expr)
    return esc(compact(__module__, __source__, expr))
end

function compact(mod, linenumbernode, expr)
    modeldef = splitdef(expr)
    layer_name, args, body = check_valid_compact_expr(modeldef)

    lname = gensym("layer")
    layers = Dict{Symbol, Any}(Symbol("___counters") => Dict())
    rewritten_body = generate_mainbody!(mod, lname, layers, body)

    which_layers = Tuple(sort(collect(filter(x -> x != Symbol("___counters") &&
                                                 !contains(string(x), "___state"),
                                             keys(layers)))))
    tnames = [Symbol("__$(w)_type") for w in which_layers]
    struct_fields = [:($w::$t) for (w, t) in zip(which_layers, tnames)]

    structdef = :(struct $layer_name{$(tnames...)} <:
                         Lux.AbstractExplicitContainerLayer{$(which_layers)}
                      $(struct_fields...)
                  end)

    layerdefs = [layers[w] for w in which_layers]

    structconstructor = combinedef(Dict(:name => layer_name, :args => [], :kwargs => [],
                                        :whereparams => [],
                                        :body => :(return $layer_name($(layerdefs...)))))

    fcall = combinedef(Dict(:name => :($lname::$layer_name), :body => rewritten_body,
                            :kwargs => [],
                            :args => append!(modeldef[:args],
                                             [
                                                 :(ps::Union{Lux.ComponentArray,
                                                             NamedTuple}),
                                                 :(st::NamedTuple),
                                             ]),
                            :whereparams => modeldef[:whereparams]))

    return prettify(Expr(:block, structdef, structconstructor, fcall))
end

function check_valid_compact_expr(modeldef)
    if !haskey(modeldef, :name)
        throw(ArgumentError("Anonymous Functions without name are not supported"))
    end
    layer_name = modeldef[:name]

    args = modeldef[:args]
    if length(args) != 1
        throw(ArgumentError("Model must take exactly 1 input!!!"))
    elseif :ps in args || :st in args
        throw(ArgumentError("Model must not take `ps` or `st` as inputs!!!"))
    end
    # Extract the names of the arguments.
    args_syms = map(args) do arg
        MacroTools.@match arg begin
            (name_::_) => name
            x_ => x
        end
    end

    kwargs = modeldef[:kwargs]
    if length(kwargs) != 0
        throw(ArgumentError("Model cannot take kwargs!!!"))
    end

    body = modeldef[:body]
    if all(l -> isa(l, LineNumberNode), body.args)
        @warn("Model definition seems empty, still continue.")
    end

    return (layer_name, args_syms, body)
end

generate_mainbody!(mod, layer_name, layers, x) = x

function generate_mainbody!(mod, layer_name, layers, expr::Expr)
    # Do not touch interpolated expressions
    expr.head === :$ && return expr.args[1]

    # Do we don't want escaped expressions because we unfortunately
    # escape the entire body afterwards.
    Meta.isexpr(expr, :escape) &&
        return generate_mainbody!(mod, layer_name, layers, expr.args[1])

    # If it's a macro, we expand it
    if Meta.isexpr(expr, :macrocall)
        return generate_mainbody!(mod, layer_name, layers,
                                  macroexpand(mod, expr; recursive=true))
    end

    if expr.head == :(=)
        rewritten_expr, _, lifted_exprs = rewrite_model_expr!(mod, layer_name, layers,
                                                              expr.args[2], expr.args[1])
        expr.args[2] = rewritten_expr
        return Expr(:block, lifted_exprs..., expr)
    end

    if expr.head == :return
        state_names = Tuple(sort(collect(filter(x -> contains(string(x), "___state"),
                                                keys(layers)))))
        state_vars = [layers[n] for n in state_names]
        field_names = Tuple([Symbol(rsplit(string(n), "___state")[1]) for n in state_names])
        state_expr = :(NamedTuple{$field_names}(tuple($(state_vars...))))
        return Expr(:return, :($(expr.args...), $state_expr))
    end

    return Expr(expr.head,
                map(x -> generate_mainbody!(mod, layer_name, layers, x), expr.args)...)
end

rewrite_model_expr!(mod, layer_name, layers, x, store_in) = (x, nothing, [])

function rewrite_model_expr!(mod, layer_name, layers, expr::Expr, store_in)
    try
        if expr.head === :call
            if expr.args[1] isa Symbol
                if eval(expr.args[1]) <: AbstractExplicitLayer
                    lname = lowercase((string(expr.args[1])))
                    c = get(layers[Symbol("___counters")], lname, 0)
                    lname_final = Symbol(lname * "_" * string((c + 1)))
                    layers[Symbol("___counters")][lname] = c + 1
                    layers[lname_final] = expr
                    layers[Symbol(string(lname_final) * "___state")] = gensym(string(lname_final) *
                                                                              "___state")
                    updated_expr = :($layer_name.$lname_final)
                    return updated_expr, lname_final, []
                end
            end
        end
    catch
    end
    expr_calls = []
    lifted_exprs = []
    i = 1
    while i <= length(expr.args)
        arg = expr.args[i]
        updated_expr, lname, lifted_priors = rewrite_model_expr!(mod, layer_name, layers,
                                                                 arg, store_in)
        lifted_exprs = append!(lifted_priors, lifted_exprs)
        if lname !== nothing
            store_in_var = gensym(store_in)

            func_args = []
            for arg in expr.args[(i + 1):end]
                arg_expr, _, lifted_priors = rewrite_model_expr!(mod, layer_name, layers,
                                                                 arg, store_in_var)
                lifted_exprs = append!(lifted_priors, lifted_exprs)
                push!(func_args, arg_expr)
            end

            if length(func_args) > 1
                push!(lifted_exprs,
                      :(($(store_in_var),
                        $(layers[Symbol(string(lname) * "___state")])) = $updated_expr(tuple($(func_args...)),
                                                                                       ps.$lname,
                                                                                       st.$lname)))
            else
                push!(lifted_exprs,
                      :(($(store_in_var),
                        $(layers[Symbol(string(lname) * "___state")])) = $updated_expr($(func_args[1]),
                                                                                       ps.$lname,
                                                                                       st.$lname)))
            end
            push!(expr_calls, :__do_collapse__)
            push!(expr_calls, :($store_in_var))
            break
        else
            push!(expr_calls, updated_expr)
        end
        i += 1
    end
    if :__do_collapse__ in expr_calls
        return expr_calls[2], nothing, lifted_exprs
    else
        return Expr(expr.head, expr_calls...), nothing, lifted_exprs
    end
end
