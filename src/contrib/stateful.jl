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

@inline function __maybe_make_stateful(layer::AbstractExplicitLayer, ps, st)
    return StatefulLuxLayer(layer, ps, st)
end
@inline __maybe_make_stateful(::Nothing, ps, st) = ps === nothing ? st : ps
@inline function __maybe_make_stateful(model::Union{AbstractVector, Tuple}, ps, st)
    return map(i -> __maybe_make_stateful(model[i], ps[i], st[i]), eachindex(model))
end
@inline function __maybe_make_stateful(model::NamedTuple{fields}, ps, st) where {fields}
    return NamedTuple{fields}(map(f -> __maybe_make_stateful(getproperty(model, f),
            getproperty(ps, f), getproperty(st, f)), fields))
end
