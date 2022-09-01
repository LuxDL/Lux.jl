using Functors: functor

"""
    @layer_map func layer ps st

See the documentation of [`Lux.layer_map`](@ref) for more details. This macro eliminates
the need to the set the layer name, and uses the variable name as the starting point.

## Example

```julia
using Lux, Random, Setfield

c = Parallel(+; chain=Chain(; dense_1=Dense(2 => 3), bn=BatchNorm(3),
                              dense_2=Dense(3 => 5)),
             dense_3=Dense(5 => 1))

rng = Random.default_rng()
ps, st = Lux.setup(rng, c)

# Makes parameters of Dense Layers inside Chain zero
function zero_dense_params(l, ps, st, name)
    if l isa Dense
        println("zeroing params of $name")
        @set! ps.weight = zero.(ps.weight)
        @set! ps.bias = zero.(ps.bias)
    end
    return l, ps, st
end

Lux.@layer_map zero_dense_params c ps st
```
"""
macro layer_map(f, l, ps, st)
    quote
        layer_map($(esc(f)), $(esc(l)), $(esc(ps)), $(esc(st)), $(string(l)))
    end
end

"""
    layer_map(f::Function, l::AbstractExplicitLayer, ps, st::NamedTuple,
              name::String="model")

Map the function `f` over the model `l`, with the parameters `ps` and states `st`. This is
different from `Functors.fmap` since it zips the layers, parameters, and states and invokes
the function on all of them together.

## Call Signature for `f`

  - Must take 4 inputs -- `AbstractExplicitLayer`, Corresponding Parameters, Corresponding
    States, and the name of the layer.
  - Must return a tuple of 3 elements -- `AbstractExplicitLayer`, new parameters and the new
    states.

!!! tip

    We recommend using the macro `Lux.@layer_map` instead of this function. It automatically
    sets the `name` of the layer to be the variable name.

## Example

```julia
using Lux, Random, Setfield

c = Parallel(+; chain=Chain(; dense_1=Dense(2 => 3), bn=BatchNorm(3),
                              dense_2=Dense(3 => 5)),
             dense_3=Dense(5 => 1))

rng = Random.default_rng()
ps, st = Lux.setup(rng, c)

# Makes parameters of Dense Layers inside Chain zero
function zero_dense_params(l, ps, st, name)
    if l isa Dense
        println("zeroing params of $name")
        @set! ps.weight = zero.(ps.weight)
        @set! ps.bias = zero.(ps.bias)
    end
    return l, ps, st
end

Lux.layer_map(zero_dense_params, c, ps, st)
```
"""
function layer_map(f::Function, l::AbstractExplicitLayer, ps, st::NamedTuple,
                   name::String="model")
    l_c, l_re = Functors.functor(l)
    ps_c, ps_re = Functors.functor(ps)
    st_c, st_re = Functors.functor(st)

    length(l_c) == 0 && return f(l, ps, st, name)

    l_c_ = l_c isa Tuple ? l_c[1] : l_c
    ks = keys(l_c_)

    l_c_new, ps_c_new, st_c_new = [], [], []
    for k in ks
        l_c_new_, ps_c_new_, st_c_new_ = layer_map(f, getproperty(l_c_, k),
                                                   getproperty(ps_c, k),
                                                   getproperty(st_c, k),
                                                   join((name, k), "."))
        push!(l_c_new, k => l_c_new_)
        push!(ps_c_new, k => ps_c_new_)
        push!(st_c_new, k => st_c_new_)
    end
    l_c_new = (; l_c_new...)
    l_c_new = l_c isa Tuple ? (l_c_new,) : l_c_new

    l_new = l_re(l_c_new)
    ps_new = ps_re((; ps_c_new...))
    st_new = st_re((; st_c_new...))

    return l_new, ps_new, st_new
end
