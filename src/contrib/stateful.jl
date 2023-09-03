"""
    StatefulLuxLayer(model, ps, st)

::: warning

This is not a Lux.AbstractExplicitLayer

:::

A convenience wrapper over Lux layers which stores the parameters and states internally.
Most users should not be using this version. This comes handy when Lux internally uses
the `@compact` to construct models and in SciML codebases where propagating state might
involving [`Box`ing](https://github.com/JuliaLang/julia/issues/15276).

For a motivating example, see the Neural ODE tutorial.

::: warning

State is mutated in place. An additional caveat is that the updated state from `model` must
have the same type as `st`.

:::

## Arguments

  - `model`: A Lux layer
  - `ps`: The parameters of the layer. This can be set to `nothing`, if the user provides
    the parameters on function call
  - `st`: The state of the layer

## Inputs

  - `x`: The input to the layer
  - `ps`: The parameters of the layer. Optional, defaults to `s.ps`

## Outputs

  - `y`: The output of the layer
"""
@concrete mutable struct StatefulLuxLayer{M <: AbstractExplicitLayer}
    model::M
    ps
    st
end

function (s::StatefulLuxLayer)(x, p=s.ps)
    y, st = apply(s.model, x, p, s.st)
    s.st = st
    return y
end

function __maybe_make_stateful(layer::AbstractExplicitLayer, ps, st)
    return StatefulLuxLayer(layer, ps, st)
end
__maybe_make_stateful(::Nothing, ps, st) = ps === nothing ? st : ps
function __maybe_make_stateful(model::Union{AbstractVector, Tuple}, ps, st)
    return [__maybe_make_stateful(m, p, s) for (m, p, s) in zip(model, ps, st)]
end
function __maybe_make_stateful(model, ps, st)
    __f(l, p, s, _) = (__maybe_make_stateful(l, p, s), p, s)
    return first(layer_map(__f, model, ps, st))
end
