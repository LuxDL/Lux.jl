"""
    StatefulLuxLayer(model, ps, st; st_fixed_type = Val(true))

!!! warning

    This is not a Lux.AbstractExplicitLayer

A convenience wrapper over Lux layers which stores the parameters and states internally.
This is meant to be used in internal implementation of layers.

## Usecases

  - Internal implementation of [`@compact`](@ref) heavily uses this layer.
  - In SciML codebases where propagating state might involving
    [`Box`ing](https://github.com/JuliaLang/julia/issues/15276). For a motivating example,
    see the Neural ODE tutorial.
  - Facilitates Nested AD support in Lux. For more details on this feature, see the
    [Nested AD Manual Page](@ref nested_autodiff).

## Arguments

  - `model`: A Lux layer
  - `ps`: The parameters of the layer. This can be set to `nothing`, if the user provides
    the parameters on function call
  - `st`: The state of the layer

## Keyword Arguments

  - `st_fixed_type`: If `Val(true)`, then the type of the `state` is fixed, i.e.,
    `typeof(last(model(x, ps, st))) == st`. If this is not the case, then `st_fixed_type`
    must be set to `Val(false)`. If `st_fixed_type` is set to `Val(false)`, then type
    stability is not guaranteed.

## Inputs

  - `x`: The input to the layer
  - `ps`: The parameters of the layer. Optional, defaults to `s.ps`

## Outputs

  - `y`: The output of the layer
"""
mutable struct StatefulLuxLayer{ST, M <: AbstractExplicitLayer, psType, stType}
    const model::M
    ps::psType
    st::stType
    st_any::Any

    function StatefulLuxLayer{ST}(model, ps, st, st_any) where {ST}
        return new{ST, typeof(model), typeof(ps), typeof(st)}(model, ps, st, st_any)
    end
end

@inline LuxCore.parameterlength(m::StatefulLuxLayer) = LuxCore.parameterlength(m.model)
@inline LuxCore.statelength(m::StatefulLuxLayer) = LuxCore.statelength(m.model)
@inline LuxCore.apply(m::StatefulLuxLayer, x, p) = m(x, p)

function ConstructionBase.constructorof(::Type{<:StatefulLuxLayer{FT}}) where {FT}
    return StatefulLuxLayer{FT}
end

StatefulLuxLayer(model, st; kwargs...) = StatefulLuxLayer(model, nothing, st; kwargs...)
function StatefulLuxLayer(model, ps, st; st_fixed_type::Val{ST}=Val(true)) where {ST}
    ST && return StatefulLuxLayer{ST}(model, ps, st, nothing)
    return StatefulLuxLayer{ST}(model, ps, nothing, st)
end

function (s::StatefulLuxLayer{true})(x, p=s.ps)
    y, st = apply(s.model, x, p, s.st)
    s.st = st
    return y
end

function (s::StatefulLuxLayer{false})(x, p=s.ps)
    y, st = apply(s.model, x, p, s.st_any)
    s.st_any = st
    return y
end

## Only needed when the parameters are `nothing`
function CRC.rrule(::Type{<:StatefulLuxLayer{FT}}, model::AbstractExplicitLayer,
        ::Nothing, st, st_any) where {FT}
    slayer = StatefulLuxLayer{FT}(model, nothing, st, st_any)
    function ∇StatefulLuxLayer(::Union{ZeroTangent, NoTangent})
        return ntuple(Returns(NoTangent()), 5)
    end
    return slayer, ∇StatefulLuxLayer
end

function CRC.rrule(::Type{<:StatefulLuxLayer{FT}},
        model::AbstractExplicitLayer, ps, st, st_any) where {FT}
    slayer = StatefulLuxLayer{FT}(model, ps, st, st_any)
    ∇StatefulLuxLayer(Δ) = (CRC.NoTangent(), Δ.model, Δ.ps, Δ.st, Δ.st_any)
    return slayer, ∇StatefulLuxLayer
end
