module LuxCore

using Functors, Random, Setfield

function _default_rng()
    rng = Random.default_rng()
    Random.seed!(rng, 1234)
    return rng
end

"""
    abstract type AbstractExplicitLayer

Abstract Type for all Lux Layers

Users implementing their custom layer, **must** implement

  - `initialparameters(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` -- This
    returns a `NamedTuple` containing the trainable parameters for the layer.
  - `initialstates(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` -- This returns a
    NamedTuple containing the current state for the layer. For most layers this is typically
    empty. Layers that would potentially contain this include `BatchNorm`, `LSTM`, `GRU` etc.

Optionally:

  - `parameterlength(layer::CustomAbstractExplicitLayer)` -- These can be automatically
    calculated, but it is recommended that the user defines these.
  - `statelength(layer::CustomAbstractExplicitLayer)` -- These can be automatically
    calculated, but it is recommended that the user defines these.

See also [`AbstractExplicitContainerLayer`](@ref)
"""
abstract type AbstractExplicitLayer end

"""
    initialparameters(rng::AbstractRNG, layer)

Generate the initial parameters of the layer `l`.
"""
initialparameters(::AbstractRNG, ::AbstractExplicitLayer) = NamedTuple()
function initialparameters(rng::AbstractRNG, l::NamedTuple)
    return map(Base.Fix1(initialparameters, rng), l)
end
initialparameters(::AbstractRNG, ::Nothing) = NamedTuple()
function initialparameters(rng::AbstractRNG, l::Union{Tuple, AbstractArray})
    any(Base.Fix2(isa, AbstractExplicitLayer), l) &&
        return map(Base.Fix1(initialparameters, rng), l)
    throw(MethodError(initialparameters, (rng, l)))
end
function initialparameters(rng::AbstractRNG, l)
    contains_lux_layer(l) && return fmap(Base.Fix1(initialparameters, rng), l)
    throw(MethodError(initialparameters, (rng, l)))
end

"""
    initialstates(rng::AbstractRNG, layer)

Generate the initial states of the layer `l`.
"""
initialstates(::AbstractRNG, ::AbstractExplicitLayer) = NamedTuple()
initialstates(rng::AbstractRNG, l::NamedTuple) = map(Base.Fix1(initialstates, rng), l)
initialstates(::AbstractRNG, ::Nothing) = NamedTuple()
function initialstates(rng::AbstractRNG, l::Union{Tuple, AbstractArray})
    any(Base.Fix2(isa, AbstractExplicitLayer), l) &&
        return map(Base.Fix1(initialstates, rng), l)
    throw(MethodError(initialstates, (rng, l)))
end
function initialstates(rng::AbstractRNG, l)
    contains_lux_layer(l) && return fmap(Base.Fix1(initialstates, rng), l)
    throw(MethodError(initialstates, (rng, l)))
end

_getemptystate(::AbstractExplicitLayer) = NamedTuple()
function _getemptystate(l::NamedTuple{fields}) where {fields}
    return NamedTuple{fields}(map(_getemptystate, values(l)))
end

"""
    parameterlength(layer)

Return the total number of parameters of the layer `l`.
"""
function parameterlength(l::AbstractExplicitLayer)
    return parameterlength(initialparameters(_default_rng(), l))
end
function parameterlength(nt::Union{NamedTuple, Tuple})
    return length(nt) == 0 ? 0 : sum(parameterlength, nt)
end
parameterlength(a::AbstractArray) = length(a)

"""
    statelength(layer)

Return the total number of states of the layer `l`.
"""
statelength(l::AbstractExplicitLayer) = statelength(initialstates(_default_rng(), l))
statelength(nt::Union{NamedTuple, Tuple}) = length(nt) == 0 ? 0 : sum(statelength, nt)
statelength(a::AbstractArray) = length(a)
statelength(::Any) = 1

"""
    inputsize(layer)

Return the input size of the layer.
"""
function inputsize end

__size(x::AbstractVector{T}) where {T} = isbitstype(T) ? size(x) : __size.(x)
function __size(x::AbstractArray{T, N}) where {T, N}
    return isbitstype(T) ? size(x)[1:(N - 1)] : __size.(x)
end
__size(x::Tuple) = __size.(x)
__size(x::NamedTuple{fields}) where {fields} = NamedTuple{fields}(__size.(values(x)))
__size(x) = fmap(__size, x)

"""
    outputsize(layer, x, rng)

Return the output size of the layer. If `outputsize(layer)` is defined, that method
takes precedence, else we compute the layer output to determine the final size.
"""
function outputsize(layer, x, rng)
    has_static_outputsize = hasmethod(outputsize, Tuple{typeof(layer)})
    return outputsize(Val(has_static_outputsize), layer, x, rng)
end

function outputsize(::Val{true}, layer, x, rng)
    return outputsize(layer)
end

function outputsize(::Val{false}, layer, x, rng)
    ps, st = LuxCore.setup(rng, layer)
    y = first(apply(layer, x, ps, st))
    return __size(y)
end

"""
    setup(rng::AbstractRNG, layer)

Shorthand for getting the parameters and states of the layer `l`. Is equivalent to
`(initialparameters(rng, l), initialstates(rng, l))`.

!!! warning

    This function is not pure, it mutates `rng`.
"""
setup(rng::AbstractRNG, l) = (initialparameters(rng, l), initialstates(rng, l))

"""
    apply(model, x, ps, st)

Simply calls `model(x, ps, st)`
"""
apply(model::AbstractExplicitLayer, x, ps, st) = model(x, ps, st)

"""
    stateless_apply(model, x, ps)

Calls `apply` and only returns the first argument.
"""
function stateless_apply(model::AbstractExplicitLayer, x, ps)
    return first(apply(model, x, ps, _getemptystate(model)))
end

"""
    display_name(layer::AbstractExplicitLayer)

Printed Name of the `layer`. If the `layer` has a field `name` that is used, else the type
name is used.
"""
@generated function display_name(l::L) where {L <: AbstractExplicitLayer}
    hasfield(L, :name) &&
        return :(ifelse(l.name === nothing, $(string(nameof(L))), string(l.name)))
    return :($(string(nameof(L))))
end
display_name(::T) where {T} = string(nameof(T))

Base.show(io::IO, x::AbstractExplicitLayer) = print(io, "$(display_name(x))()")

# Abstract Container Layers
"""
    abstract type AbstractExplicitContainerLayer{layers} <: AbstractExplicitLayer

Abstract Container Type for certain Lux Layers. `layers` is a tuple containing fieldnames
for the layer, and constructs the parameters and states using those.

Users implementing their custom layer can extend the same functions as in
[`AbstractExplicitLayer`](@ref).

!!! tip

    Advanced structure manipulation of these layers post construction is possible via
    `Functors.fmap`. For a more flexible interface, we recommend using
    `Lux.Experimental.@layer_map`.
"""
abstract type AbstractExplicitContainerLayer{layers} <: AbstractExplicitLayer end

function initialparameters(rng::AbstractRNG,
        l::AbstractExplicitContainerLayer{layers}) where {layers}
    length(layers) == 1 && return initialparameters(rng, getfield(l, layers[1]))
    return NamedTuple{layers}(initialparameters.(rng, getfield.((l,), layers)))
end

function initialstates(rng::AbstractRNG,
        l::AbstractExplicitContainerLayer{layers}) where {layers}
    length(layers) == 1 && return initialstates(rng, getfield(l, layers[1]))
    return NamedTuple{layers}(initialstates.(rng, getfield.((l,), layers)))
end

function parameterlength(l::AbstractExplicitContainerLayer{layers}) where {layers}
    return sum(parameterlength, getfield.((l,), layers))
end

function statelength(l::AbstractExplicitContainerLayer{layers}) where {layers}
    return sum(statelength, getfield.((l,), layers))
end

function _getemptystate(l::AbstractExplicitContainerLayer{layers}) where {layers}
    length(layers) == 1 && return _getemptystate(getfield(l, first(layers)))
    return NamedTuple{layers}(_getemptystate.(getfield.((l,), layers)))
end

# Make AbstractExplicit Layers Functor Compatible
function Functors.functor(::Type{<:AbstractExplicitContainerLayer{layers}},
        x) where {layers}
    _children = NamedTuple{layers}(getproperty.((x,), layers))
    function layer_reconstructor(z)
        return reduce((l, (c, n)) -> set(l, Setfield.PropertyLens{n}(), c), zip(z, layers);
            init=x)
    end
    return _children, layer_reconstructor
end

# Test Mode
"""
    testmode(st::NamedTuple)

Make all occurances of `training` in state `st` -- `Val(false)`.
"""
testmode(st::NamedTuple) = update_state(st, :training, Val(false))

"""
    trainmode(st::NamedTuple)

Make all occurances of `training` in state `st` -- `Val(true)`.
"""
trainmode(st::NamedTuple) = update_state(st, :training, Val(true))

"""
    update_state(st::NamedTuple, key::Symbol, value;
        layer_check=_default_layer_check(key))

Recursively update all occurances of the `key` in the state `st` with the `value`.
"""
function update_state(st::NamedTuple, key::Symbol, value;
        layer_check=_default_layer_check(key))
    function _update_state(st, key::Symbol, value)
        return Setfield.set(st, Setfield.PropertyLens{key}(), value)
    end
    return fmap(_st -> _update_state(_st, key, value), st; exclude=layer_check)
end

function _default_layer_check(key)
    _default_layer_check_closure(x) = hasmethod(keys, (typeof(x),)) ? key ∈ keys(x) : false
    return _default_layer_check_closure
end

"""
    contains_lux_layer(l) -> Bool

Check if the structure `l` is a Lux AbstractExplicitLayer or a container of such a layer.
"""
function contains_lux_layer(l)
    return check_fmap_condition(Base.Fix2(isa, AbstractExplicitLayer),
        AbstractExplicitLayer, l)
end

"""
    check_fmap_condition(cond, tmatch, x) -> Bool

`fmap`s into the structure `x` and see if `cond` is statisfied for any of the leaf
elements.

## Arguments

    * `cond` - A function that takes a single argument and returns a `Bool`.
    * `tmatch` - A shortcut to check if `x` is of type `tmatch`. Can be disabled by passing
      `nothing`.
    * `x` - The structure to check.

## Returns

A Boolean Value
"""
function check_fmap_condition(cond, tmatch, x)
    tmatch !== nothing && x isa tmatch && return true
    matched = Ref(false)
    function __check(l)
        cond(l) && (matched[] = true)
        return l
    end
    fmap(__check, x)
    return matched[]
end

end
