module StatefulLuxLayerImpl

using ..LuxCore:
    AbstractLuxLayer,
    AbstractLuxContainerLayer,
    AbstractLuxWrapperLayer,
    preserves_state_type

const StaticBool = Union{Val{true},Val{false}}

static(v::StaticBool) = v
static(v::Bool) = Val(v)

dynamic(::Val{true}) = true
dynamic(::Val{false}) = false
dynamic(v::Bool) = v

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

function Base.getproperty(l::StatefulLuxLayer, s::Symbol)
    s === :st && return get_state(l)
    s === :st_any && throw(
        ArgumentError(
            "No property `st_any` for `StatefulLuxLayer`. To access the `st_any` field use \
             `getfield(l, :st_any)` instead.",
        ),
    )
    return getfield(l, s)
end

function Base.setproperty!(l::StatefulLuxLayer, s::Symbol, v)
    s === :st && return set_state!(l, v)
    s === :st_any && throw(
        ArgumentError(
            "No property `st_any` for `StatefulLuxLayer`. To set the `st_any` field use \
             `setfield!(l, :st_any, v)` instead.",
        ),
    )
    return setfield!(l, s, v)
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
    return StatefulLuxLayer{preserves_state_type(model)}(model, ps, st)
end

get_state(l::StatefulLuxLayer{Val{true}}) = getfield(l, :st)
get_state(l::StatefulLuxLayer{Val{false}}) = getfield(l, :st_any)
# Needed for compact macro implementation
get_state(st::AbstractArray{<:Number}) = st
get_state(st::Union{AbstractArray,Tuple,NamedTuple}) = map(get_state, st)
get_state(st) = st

function set_state!(
    s::StatefulLuxLayer{Val{true},<:Any,<:Any,stType}, st::stType
) where {stType}
    return setfield!(s, :st, st)
end
function set_state!(
    ::StatefulLuxLayer{Val{true},<:Any,<:Any,stType}, ::stType2
) where {stType,stType2}
    throw(
        ArgumentError("Output state from the model has type `$(stType2)`, but expected \
                       `$(stType)`. Construct the Stateful layer as \
                       `StatefulLuxLayer{false}` instead of `StatefulLuxLayer{true}`.\n \
                       Additionally ensure `LuxCore.preserves_state_type` is properly \
                       defined for all the layers in the model.")
    )
end
set_state!(s::StatefulLuxLayer{Val{false}}, st) = setfield!(s, :st_any, st)

# This is an internal implementation detail for bypassing manual state management
struct DualFieldNamedTuple{F1,F2,DM,D<:Tuple}
    is_data_mutable::DM
    data::D
end

function get_backing_data(l::DualFieldNamedTuple{F1,F2,Val{false}}) where {F1,F2}
    return getfield(l, :data)
end
function get_backing_data(l::DualFieldNamedTuple{F1,F2,Val{true}}) where {F1,F2}
    return getindex.(getfield(l, :data))
end

@generated function prefixed_fieldnames(
    ::NamedTuple{fields}, ::Val{prefix}
) where {fields,prefix}
    new_fields = Tuple(QuoteNode.(Symbol.(Ref(prefix), fields)))
    return :($(Tuple(new_fields)...),)
end

function DualFieldNamedTuple(
    data::NamedTuple{fields},
    ::Val{prefix},
    is_data_mutable::StaticBool=Val(false),
    is_data_type_fixed::StaticBool=Val(true),
) where {fields,prefix}
    @assert length(fields) == length(data)

    if dynamic(is_data_mutable)
        if dynamic(is_data_type_fixed)
            backing = Ref.(values(data))
        else
            backing = Ref{Any}.(values(data))
        end
        BT = typeof(backing)
    else
        @assert dynamic(is_data_type_fixed)
        backing = values(data)
        BT = typeof(backing)
    end

    return DualFieldNamedTuple{
        fields,prefixed_fieldnames(data, Val(prefix)),typeof(is_data_mutable),BT
    }(
        is_data_mutable, backing
    )
end

function DualFieldNamedTuple(data, args...)
    fnames = fieldnames(typeof(data))
    return DualFieldNamedTuple(NamedTuple{fnames}(getfield.(Ref(data), fnames)), args...)
end

first_property_names(::DualFieldNamedTuple{F1}) where {F1} = F1

Base.propertynames(::DualFieldNamedTuple{F1,F2}) where {F1,F2} = (F1..., F2...)

function findfieldidx(::DualFieldNamedTuple{F1,F2}, s::Symbol) where {F1,F2}
    idx = findfirst(==(s), F1)
    idx !== nothing && return idx
    return findfirst(==(s), F2)
end

maybe_unwrap_ref(::Val{false}, x) = x
maybe_unwrap_ref(::Val{true}, x::Ref) = x[]

function Base.getproperty(l::DualFieldNamedTuple{F1,F2}, s::Symbol) where {F1,F2}
    idx = findfieldidx(l, s)
    idx !== nothing && return maybe_unwrap_ref(
        getfield(l, :is_data_mutable), getfield(getfield(l, :data), idx)
    )
    throw(ArgumentError("No property $s for `DualFieldNamedTuple`"))
end

function Base.setproperty!(
    ::DualFieldNamedTuple{F1,F2,Val{false}}, s::Symbol, v
) where {F1,F2}
    throw(ArgumentError("Cannot set property $s for `DualFieldNamedTuple` since the data \
                         was constructed as immutable."))
end

function Base.setproperty!(
    l::DualFieldNamedTuple{F1,F2,Val{true}}, s::Symbol, v
) where {F1,F2}
    idx = findfieldidx(l, s)
    idx === nothing && throw(ArgumentError("No property $s for `DualFieldNamedTuple`"))
    getfield(l, :data)[idx][] = v
    return l
end

struct NamedTupleStatefulLuxLayer{layers,NT<:NamedTuple,M,NTPS,NTST}
    smodels::NT
    model::M
    ps_extra::NTPS
    st_extra::NTST
end

function get_states_as_namedtuple(l::NamedTupleStatefulLuxLayer{layers}) where {layers}
    smodels = getfield(l, :smodels)
    return NamedTuple{(layers..., first_property_names(getfield(l, :st_extra))...)}((
        (getfield(smodels, layer).st for layer in layers)...,
        get_backing_data(getfield(l, :st_extra))...,
    ),)
end

function NamedTupleStatefulLuxLayer{fields}(
    smodels, model::AbstractLuxLayer, ps::NamedTuple, st::NamedTuple
) where {fields}
    model_dfnt = DualFieldNamedTuple(model, Val(:model_))
    ps_extra = DualFieldNamedTuple(ps, Val(:ps_))
    st_extra = DualFieldNamedTuple(
        st, Val(:st_), Val(true), Val(preserves_state_type(model))
    )
    return NamedTupleStatefulLuxLayer{
        fields,typeof(smodels),typeof(model_dfnt),typeof(ps_extra),typeof(st_extra)
    }(
        smodels, model_dfnt, ps_extra, st_extra
    )
end

function NamedTupleStatefulLuxLayer(model::AbstractLuxLayer, ps, st)
    return NamedTupleStatefulLuxLayer{()}((;), model, ps, st)
end

@generated function NamedTupleStatefulLuxLayer(
    model::AbstractLuxWrapperLayer{layer}, ps, st
) where {layer}
    layers = (layer,)
    smodel_expr = [:(StatefulLuxLayer(model.$(layer), ps, st))]
    return quote
        return NamedTupleStatefulLuxLayer{$(layers)}(
            NamedTuple{$(layers)}(($(Tuple(smodel_expr)...),)),
            model,
            NamedTuple(),
            NamedTuple(),
        )
    end
end

@generated function NamedTupleStatefulLuxLayer(
    model::AbstractLuxContainerLayer{layers}, ps::PS, st::ST
) where {layers,PS,ST}
    ps_extra_fields = Tuple(setdiff(fieldnames(PS), layers))
    st_extra_fields = Tuple(setdiff(fieldnames(ST), layers))

    smodel_expr = [
        :(StatefulLuxLayer(model.$(layer), ps.$(layer), st.$(layer))) for layer in layers
    ]

    ps_get_expr = [:(getfield(ps, $(QuoteNode(f)))) for f in ps_extra_fields]
    st_get_expr = [:(getfield(st, $(QuoteNode(f)))) for f in st_extra_fields]

    return quote
        smodels = NamedTuple{$(layers)}(($(Tuple(smodel_expr)...),))
        ps_extra = NamedTuple{$(ps_extra_fields)}(($(Tuple(ps_get_expr)...),))
        st_extra = NamedTuple{$(st_extra_fields)}(($(Tuple(st_get_expr)...),))
        return NamedTupleStatefulLuxLayer{$(layers)}(smodels, model, ps_extra, st_extra)
    end
end

function Base.propertynames(l::NamedTupleStatefulLuxLayer{layers}) where {layers}
    return (
        layers...,
        propertynames(getfield(l, :ps_extra))...,
        propertynames(getfield(l, :st_extra))...,
        propertynames(getfield(l, :model))...,
    )
end

function Base.getproperty(l::NamedTupleStatefulLuxLayer{layers}, s::Symbol) where {layers}
    s in layers && return getfield(getfield(l, :smodels), s)
    hasproperty(getfield(l, :model), s) && return getproperty(getfield(l, :model), s)
    hasproperty(getfield(l, :ps_extra), s) && return getproperty(getfield(l, :ps_extra), s)
    hasproperty(getfield(l, :st_extra), s) && return getproperty(getfield(l, :st_extra), s)
    throw(ArgumentError("No property $(s) for `NamedTupleStatefulLuxLayer`"))
end

function Base.setproperty!(
    l::NamedTupleStatefulLuxLayer{layers}, s::Symbol, v
) where {layers}
    s in layers && return throw(ArgumentError("Cannot set property $(s) for \
                                              `NamedTupleStatefulLuxLayer` since it is a \
                                              container layer."))
    hasproperty(getfield(l, :model), s) &&
        throw(ArgumentError("Cannot set property $(s) for `NamedTupleStatefulLuxLayer` \
                             since it is a model field."))
    hasproperty(getfield(l, :ps_extra), s) &&
        throw(ArgumentError("Cannot set property $(s) for `NamedTupleStatefulLuxLayer` \
                            since it is a parameter."))
    hasproperty(getfield(l, :st_extra), s) &&
        return setproperty!(getfield(l, :st_extra), s, v)
    throw(ArgumentError("No property $(s) for NamedTupleStatefulLuxLayer"))
end

export StatefulLuxLayer, NamedTupleStatefulLuxLayer

end

using .StatefulLuxLayerImpl: StatefulLuxLayer

@doc """
    StatefulLuxLayer{FT}(model, ps, st)
    StatefulLuxLayer(model, ps, st)

!!! warning

    This is not a LuxCore.AbstractLuxLayer

!!! tip

    This layer can be used as a drop-in replacement for `Flux.jl` layers.

A convenience wrapper over Lux layers which stores the parameters and states internally.
This is meant to be used in internal implementation of layers.

When using the definition of `StatefulLuxLayer` without `FT` specified, make sure that all
of the layers in the model define [`LuxCore.preserves_state_type`](@ref). Else we implicitly
assume that the state type is preserved.

## Usecases

  - Internal implementation of `@compact` heavily uses this layer.
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
""" StatefulLuxLayer

# LuxCore interface
for op in (:trainmode, :testmode)
    @eval function $(op)(m::StatefulLuxLayer)
        return StatefulLuxLayer{StatefulLuxLayerImpl.dynamic(m.fixed_state_type)}(
            m.model, m.ps, LuxCore.$(op)(StatefulLuxLayerImpl.get_state(m))
        )
    end
end

function update_state(s::StatefulLuxLayer, key::Symbol, value; kwargs...)
    return StatefulLuxLayer{StatefulLuxLayerImpl.dynamic(s.fixed_state_type)}(
        s.model,
        s.ps,
        update_state(StatefulLuxLayerImpl.get_state(s), key, value; kwargs...),
    )
end

function parameterlength(m::StatefulLuxLayer)
    m.ps === nothing && return parameterlength(m.model)
    return parameterlength(m.ps)
end

statelength(m::StatefulLuxLayer) = statelength(StatefulLuxLayerImpl.get_state(m))

preserves_state_type(m::StatefulLuxLayer) = StatefulLuxLayerImpl.dynamic(m.fixed_state_type)

function (m::StatefulLuxLayer)(x, p=m.ps)
    @assert p !== nothing "Model parameters are not set in constructor. Pass in `ps` \
                           explicitly."
    y, st = apply(m.model, x, p, m.st)
    StatefulLuxLayerImpl.set_state!(m, st)
    return y
end

apply(m::StatefulLuxLayer, x, p=m.ps) = m(x, p)
