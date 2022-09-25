"""
    AbstractExplicitLayer

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
    initialparameters(rng::AbstractRNG, l)

Generate the initial parameters of the layer `l`.
"""
initialparameters(::AbstractRNG, ::AbstractExplicitLayer) = NamedTuple()
function initialparameters(rng::AbstractRNG, l::NamedTuple)
    return map(Base.Fix1(initialparameters, rng), l)
end

"""
    initialstates(rng::AbstractRNG, l)

Generate the initial states of the layer `l`.
"""
initialstates(::AbstractRNG, ::AbstractExplicitLayer) = NamedTuple()
initialstates(rng::AbstractRNG, l::NamedTuple) = map(Base.Fix1(initialstates, rng), l)

"""
    parameterlength(l)

Return the total number of parameters of the layer `l`.
"""
function parameterlength(l::AbstractExplicitLayer)
    return parameterlength(initialparameters(Random.default_rng(), l))
end
function parameterlength(nt::Union{NamedTuple, Tuple})
    return length(nt) == 0 ? 0 : sum(parameterlength, nt)
end
parameterlength(a::AbstractArray) = length(a)

"""
    statelength(l)

Return the total number of states of the layer `l`.
"""
statelength(l::AbstractExplicitLayer) = statelength(initialstates(Random.default_rng(), l))
statelength(nt::Union{NamedTuple, Tuple}) = length(nt) == 0 ? 0 : sum(statelength, nt)
statelength(a::AbstractArray) = length(a)
statelength(x::Union{Number, Symbol, Val}) = 1

"""
    setup(rng::AbstractRNG, l::AbstractExplicitLayer)

Shorthand for getting the parameters and states of the layer `l`. Is equivalent to
`(initialparameters(rng, l), initialstates(rng, l))`.

!!! warning
    
    This function is not pure, it mutates `rng`.
"""
function setup(rng::AbstractRNG, l::AbstractExplicitLayer)
    return (initialparameters(rng, l), initialstates(rng, l))
end

"""
    apply(model::AbstractExplicitLayer, x, ps, st::NamedTuple)

Simply calls `model(x, ps, st)`
"""
function apply(model::AbstractExplicitLayer, x, ps, st::NamedTuple)
    return model(x, ps, st)
end

function Base.show(io::IO, x::AbstractExplicitLayer)
    __t = rsplit(string(get_typename(x)), "."; limit=2)
    T = length(__t) == 2 ? __t[2] : __t[1]
    return print(io, "$T()")
end

# Abstract Container Layers
"""
    AbstractExplicitContainerLayer{layers} <: AbstractExplicitLayer

Abstract Container Type for certain Lux Layers. `layers` is a tuple containing fieldnames
for the layer, and constructs the parameters and states using those.

Users implementing their custom layer can extend the same functions as in
[`AbstractExplicitLayer`](@ref).

!!! tip
    
    Advanced structure manipulation of these layers post construction is possible via
    `Functors.fmap`. For a more flexible interface, we recommend using the experimental
    feature [`Lux.@layer_map`](@ref).
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

# Make AbstractExplicit Layers Functor Compatible
function Functors.functor(::Type{<:AbstractExplicitContainerLayer},
                          x::AbstractExplicitContainerLayer{layers}) where {layers}
    _children = getproperty.((x,), layers)
    function layer_reconstructor(z)
        l = x
        for (child, name) in zip(z, layers)
            l = Setfield.set(l, Setfield.PropertyLens{name}(), child)
        end
        return l
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
    update_state(st::NamedTuple, key::Symbol, value; layer_check=_default_layer_check(key))

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
