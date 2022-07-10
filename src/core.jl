"""
    AbstractExplicitLayer{P, S}

Abstract Type for all Lux Layers

Type parameters:

  - `P::Bool`: indicates that the layer requires parameters
  - `S::Bool`: indicates that the layer requires a state

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
abstract type AbstractExplicitLayer{P, S} end

hasparams(l) = false
hasparams(::AbstractExplicitLayer{P}) where {P} = P

hasstate(l) = false
hasstate(::AbstractExplicitLayer{P, S}) where {P, S} = S

"""
    initialparameters(rng::AbstractRNG, l)

Generate the initial parameters of the layer `l`.
"""
initialparameters(::AbstractRNG, ::AbstractExplicitLayer) = NamedTuple()
initialparameters(::AbstractRNG, ::AbstractExplicitLayer{false}) = NamedTuple()
function initialparameters(rng::AbstractRNG, l::NamedTuple)
    return map(Base.Fix1(initialparameters, rng), l)
end

"""
    initialstates(rng::AbstractRNG, l)

Generate the initial states of the layer `l`.
"""
initialstates(::AbstractRNG, ::AbstractExplicitLayer) = NamedTuple()
initialstates(::AbstractRNG, ::AbstractExplicitLayer{P, false}) where {P} = NamedTuple()
initialstates(rng::AbstractRNG, l::NamedTuple) = map(Base.Fix1(initialstates, rng), l)

"""
    parameterlength(l)

Return the total number of parameters of the layer `l`.
"""
function parameterlength(l::AbstractExplicitLayer)
    return parameterlength(initialparameters(Random.default_rng(), l))
end
parameterlength(::AbstractExplicitLayer{false}) = 0
function parameterlength(nt::Union{NamedTuple, Tuple})
    return length(nt) == 0 ? 0 : sum(parameterlength, nt)
end
parameterlength(a::AbstractArray) = length(a)

"""
    statelength(l)

Return the total number of states of the layer `l`.
"""
statelength(l::AbstractExplicitLayer) = statelength(initialstates(Random.default_rng(), l))
statelength(l::AbstractExplicitLayer{P, false}) where {P} = 0
statelength(nt::Union{NamedTuple, Tuple}) = length(nt) == 0 ? 0 : sum(statelength, nt)
statelength(a::AbstractArray) = length(a)
statelength(x::Union{Number, Symbol, Val}) = 1

"""
    setup(rng::AbstractRNG, l::AbstractExplicitLayer)

Shorthand for getting the parameters and states of the layer `l`. Is equivalent to
`(initialparameters(rng, l), initialstates(rng, l))`.
"""
function setup(rng::AbstractRNG, l::AbstractExplicitLayer)
    return (initialparameters(rng, l), initialstates(rng, l))
end

"""
    apply(model::AbstractExplicitLayer, x, ps::Union{ComponentArray,NamedTuple},
          st::NamedTuple)

Simply calls `model(x, ps, st)`
"""
apply(model::AbstractExplicitLayer, x, ps, st::NamedTuple) = model(x, ps, st)
apply(model::AbstractExplicitLayer{true, false}, x, ps) = model(x, ps)
apply(model::AbstractExplicitLayer{false, true}, x, st::NamedTuple) = model(x, st)
apply(model::AbstractExplicitLayer{false, false}, x) = model(x)

(model::AbstractExplicitLayer{true, false})(x, ps) = model(x, ps, NamedTuple())
(model::AbstractExplicitLayer{false, true})(x, st::NamedTuple) = model(x, NamedTuple(), st)
(model::AbstractExplicitLayer{false, false})(x) = model(x, NamedTuple(), NamedTuple())

function Base.show(io::IO, x::AbstractExplicitLayer)
    __t = rsplit(string(get_typename(x)), "."; limit=2)
    T = length(__t) == 2 ? __t[2] : __t[1]
    return print(io, "$T()")
end

# Abstract Container Layers
"""
    AbstractExplicitContainerLayer{layers,P,S} <: AbstractExplicitLayer{P,S}

Abstract Container Type for certain Lux Layers. `layers` is a tuple containing fieldnames
for the layer, and constructs the parameters and states using those.

Users implementing their custom layer can extend the same functions as in
initialstates(::AbstractRNG, ::AbstractExplicitLayer) = NamedTuple()
[`AbstractExplicitLayer`](@ref)
"""
abstract type AbstractExplicitContainerLayer{layers, P, S} <: AbstractExplicitLayer{P, S} end

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

function initialparameters(::AbstractRNG,
                           ::AbstractExplicitContainerLayer{layers, false}) where {layers}
    return NamedTuple()
end

function initialstates(::AbstractRNG,
                       ::AbstractExplicitContainerLayer{layers, P, false}) where {layers, P}
    return NamedTuple()
end

function parameterlength(l::AbstractExplicitContainerLayer{layers}) where {layers}
    return sum(parameterlength, getfield.((l,), layers))
end

function statelength(l::AbstractExplicitContainerLayer{layers}) where {layers}
    return sum(statelength, getfield.((l,), layers))
end

parameterlength(::AbstractExplicitContainerLayer{layers, false}) where {layers} = 0
statelength(::AbstractExplicitContainerLayer{layers, P, false}) where {layers, P} = 0

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
