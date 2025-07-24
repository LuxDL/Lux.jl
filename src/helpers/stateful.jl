"""
    StatefulLuxLayer{FT}(model, ps, st)
    StatefulLuxLayer(model, ps, st)

!!! warning

    This is not a Lux.AbstractLuxLayer

A convenience wrapper over Lux layers which stores the parameters and states internally.
This is meant to be used in internal implementation of layers.

When using the definition of `StatefulLuxLayer` without `FT` specified, make sure that all
of the layers in the model define [`LuxCore.preserves_state_type`](@ref). Else we implicitly
assume that the state type is preserved.

## Usecases

  - Internal implementation of [`@compact`](@ref) heavily uses this layer.
  - In SciML codebases where propagating state might involving
    [`Box`ing](https://github.com/JuliaLang/julia/issues/15276). For a motivating example,
    see the Neural ODE tutorial.
  - Facilitates Nested AD support in Lux. For more details on this feature, see the
    [Nested AD Manual Page](@ref nested_autodiff).

## Static Parameters

  - If `FT = true` then the type of the `state` is fixed, i.e.,
    `typeof(last(model(x, ps, st))) == st`.
  - If `FT = false` then type of the state might change. Note that while this works in all
    cases, it will introduce type instability.

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
mutable struct StatefulLuxLayer{ST,M<:AbstractLuxLayer,psType,stType}
    const model::M
    ps::psType
    st::stType
    st_any::Any
    fixed_state_type::ST

    function StatefulLuxLayer(
        model::AbstractLuxLayer, ps, st, st_any, fixed_state_type::StaticBool
    )
        return new{typeof(fixed_state_type),typeof(model),typeof(ps),typeof(st)}(
            model, ps, st, st_any, fixed_state_type
        )
    end
end

function StatefulLuxLayer{ST}(model, ps, st, st_any) where {ST}
    return StatefulLuxLayer(model, ps, st, st_any, static(ST))
end

function StatefulLuxLayer{true}(model::AbstractLuxLayer, ps, st::NamedTuple)
    return StatefulLuxLayer{true}(model, ps, st, nothing)
end
function StatefulLuxLayer{false}(model::AbstractLuxLayer, ps, st::NamedTuple)
    return StatefulLuxLayer{false}(model, ps, nothing, st)
end

function StatefulLuxLayer(model::AbstractLuxLayer, ps, st)
    return StatefulLuxLayer{LuxCore.preserves_state_type(model)}(model, ps, st)
end

for op in (:trainmode, :testmode)
    @eval function LuxCore.$(op)(s::StatefulLuxLayer)
        return StatefulLuxLayer{dynamic(s.fixed_state_type)}(
            s.model, s.ps, LuxCore.$(op)(get_state(s))
        )
    end
end

function LuxCore.update_state(s::StatefulLuxLayer, key::Symbol, value; kwargs...)
    return StatefulLuxLayer{dynamic(s.fixed_state_type)}(
        s.model, s.ps, LuxCore.update_state(get_state(s), key, value; kwargs...)
    )
end

preserves_state_type(::StatefulLuxLayer) = dynamic(s.fixed_state_type)

function Base.show(io::IO, ::MIME"text/plain", s::StatefulLuxLayer)
    return PrettyPrinting.print_wrapper_model(
        io, "StatefulLuxLayer{$(dynamic(s.fixed_state_type))}", s.model
    )
end

function Functors.functor(::Type{<:StatefulLuxLayer}, x)
    recon = let ft = x.fixed_state_type
        nt -> StatefulLuxLayer(nt.model, nt.ps, nt.st, nt.st_any, ft)
    end
    return (; x.model, x.ps, x.st, x.st_any), recon
end

function parameterlength(m::StatefulLuxLayer)
    m.ps === nothing && return parameterlength(m.model)
    return parameterlength(m.ps)
end
statelength(m::StatefulLuxLayer) = statelength(get_state(m))
apply(m::StatefulLuxLayer, x, p) = m(x, p)

get_state(s::StatefulLuxLayer{True}) = s.st
get_state(s::StatefulLuxLayer{False}) = s.st_any

CRC.@non_differentiable get_state(::StatefulLuxLayer)

function set_state!(s::StatefulLuxLayer{True,<:Any,<:Any,stType}, st::stType) where {stType}
    return s.st = st
end
function set_state!(
    ::StatefulLuxLayer{True,<:Any,<:Any,stType}, ::stType2
) where {stType,stType2}
    throw(
        ArgumentError("Output state from the model has type `$(stType2)`, but expected \
                       `$(stType)`. Construct the Stateful layer as \
                       `StatefulLuxLayer{false}` instead of `StatefulLuxLayer{true}`.\n \
                       Additionally ensure `LuxCore.preserves_state_type` is properly \
                       defined for all the layers in the model.")
    )
end
set_state!(s::StatefulLuxLayer{False}, st) = (s.st_any = st)

CRC.@non_differentiable set_state!(::Any...)

function (s::StatefulLuxLayer)(x, p=s.ps)
    y, st = apply(s.model, x, p, get_state(s))
    set_state!(s, st)
    return y
end

function CRC.rrule(
    ::Type{<:StatefulLuxLayer}, model::AbstractLuxLayer, ps, st, st_any, fixed_state_type
)
    slayer = StatefulLuxLayer(model, ps, st, st_any, fixed_state_type)
    function ∇StatefulLuxLayer(Δ)
        return NoTangent(), NoTangent(), Δ.ps, NoTangent(), NoTangent(), NoTangent()
    end
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
