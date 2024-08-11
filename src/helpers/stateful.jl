"""
    StatefulLuxLayer(model, ps, st; st_fixed_type = Val(true))  # deprecated
    StatefulLuxLayer{ST}(model, ps, st)

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
end

function StatefulLuxLayer{ST}(model, ps, st, st_any) where {ST}
    return StatefulLuxLayer{ST, typeof(model), typeof(ps), typeof(st)}(
        model, ps, st, st_any)
end

function Base.show(io::IO, ::MIME"text/plain", s::StatefulLuxLayer{ST}) where {ST}
    PrettyPrinting.print_wrapper_model(io, "StatefulLuxLayer{$ST}", s.model)
end

function Functors.functor(::Type{<:StatefulLuxLayer{FT}}, x) where {FT}
    return ((; x.model, x.ps, x.st, x.st_any),
        nt -> StatefulLuxLayer{FT}(nt.model, nt.ps, nt.st, nt.st_any))
end

function LuxCore.parameterlength(m::StatefulLuxLayer)
    m.ps === nothing && return LuxCore.parameterlength(m.model)
    return LuxCore.parameterlength(m.ps)
end
function LuxCore.statelength(m::StatefulLuxLayer{FT}) where {FT}
    FT && return LuxCore.statelength(m.st)
    return LuxCore.statelength(m.st_any)
end
LuxCore.apply(m::StatefulLuxLayer, x, p) = m(x, p)

function ConstructionBase.constructorof(::Type{<:StatefulLuxLayer{FT}}) where {FT}
    return StatefulLuxLayer{FT}
end

function StatefulLuxLayer(model::AbstractExplicitLayer, st::NamedTuple; kwargs...)
    return StatefulLuxLayer(model, nothing, st; kwargs...)
end
function StatefulLuxLayer(model::AbstractExplicitLayer, ps, st::NamedTuple;
        st_fixed_type::Val{ST}=Val(true)) where {ST}
    Base.depwarn("`st_fixed_type` is deprecated. Use `StatefulLuxLayer{ST}` instead.",
        :StatefulLuxLayer)
    return StatefulLuxLayer{ST}(model, ps, st)
end
function StatefulLuxLayer{true}(model::AbstractExplicitLayer, ps, st::NamedTuple)
    return StatefulLuxLayer{true}(model, ps, st, nothing)
end
function StatefulLuxLayer{false}(model::AbstractExplicitLayer, ps, st::NamedTuple)
    return StatefulLuxLayer{false}(model, ps, nothing, st)
end

get_state(s::StatefulLuxLayer{true}) = s.st
get_state(s::StatefulLuxLayer{false}) = s.st_any

CRC.@non_differentiable get_state(::Any)

function set_state!(
        s::StatefulLuxLayer{true, M, psType, stType}, st::stType) where {M, psType, stType}
    s.st = st
end
function set_state!(::StatefulLuxLayer{true, M, psType, stType},
        ::stType2) where {M, psType, stType, stType2}
    throw(ArgumentError("Output state from the model has type `$(stType2)`, but expected \
                         `$(stType)`. Construct the Stateful layer as \
                         `StatefulLuxLayer{false}` instead of `StatefulLuxLayer{true}`."))
end
set_state!(s::StatefulLuxLayer{false}, st) = (s.st_any = st)

CRC.@non_differentiable set_state!(::Any...)

function (s::StatefulLuxLayer)(x, p=s.ps)
    y, st = apply(s.model, x, p, get_state(s))
    set_state!(s, st)
    return y
end

function CRC.rrule(::Type{<:StatefulLuxLayer{FT}},
        model::AbstractExplicitLayer, ps, st, st_any) where {FT}
    slayer = StatefulLuxLayer{FT}(model, ps, st, st_any)
    ∇StatefulLuxLayer(Δ) = NoTangent(), NoTangent(), Δ.ps, NoTangent(), NoTangent()
    return slayer, ∇StatefulLuxLayer
end

function CRC.rrule(::typeof(getproperty), s::StatefulLuxLayer, name::Symbol)
    y = getproperty(s, name)
    ∇getproperty = @closure Δ -> begin
        name === :ps && return NoTangent(), CRC.Tangent{typeof(s)}(; ps=Δ), NoTangent()
        return NoTangent(), NoTangent(), NoTangent()
    end
    return y, ∇getproperty
end
