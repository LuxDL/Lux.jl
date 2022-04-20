# Base Type
## API:
### initialparameters(rng, l) --> Must return a NamedTuple/ComponentArray
### initialstates(rng, l)     --> Must return a NamedTuple
### parameterlength(l)        --> Integer
### statelength(l)            --> Integer
### l(x, ps, st)

abstract type AbstractExplicitLayer end

initialparameters(::AbstractRNG, ::Any) = ComponentArray{Float32}()
initialparameters(l) = initialparameters(Random.GLOBAL_RNG, l)
initialparameters(rng::AbstractRNG, l::NamedTuple{fields}) where {fields} =
    ComponentArray(NamedTuple{fields}(initialparameters.(rng, values(l))))

initialstates(::AbstractRNG, ::Any) = NamedTuple()
initialstates(l) = initialstates(Random.GLOBAL_RNG, l)
initialstates(rng::AbstractRNG, l::NamedTuple{fields}) where {fields} =
    NamedTuple{fields}(initialstates.(rng, values(l)))

parameterlength(l::AbstractExplicitLayer) = parameterlength(initialparameters(l))
parameterlength(nt::Union{NamedTuple,Tuple}) = length(nt) == 0 ? 0 : sum(parameterlength, nt)
parameterlength(a::AbstractArray) = length(a)

statelength(l::AbstractExplicitLayer) = statelength(initialstates(l))
statelength(nt::Union{NamedTuple,Tuple}) = length(nt) == 0 ? 0 : sum(statelength, nt)
statelength(a::AbstractArray) = length(a)
statelength(x) = 1

setup(rng::AbstractRNG, l::AbstractExplicitLayer) = (initialparameters(rng, l), initialstates(rng, l))
setup(l::AbstractExplicitLayer) = setup(Random.GLOBAL_RNG, l)

apply(model::AbstractExplicitLayer, x, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple) = model(x, ps, st)

# Abstract Container Layers
abstract type AbstractExplicitContainerLayer{layers} <: AbstractExplicitLayer end

function initialparameters(rng::AbstractRNG, l::AbstractExplicitContainerLayer{layers}) where {layers}
    length(layers) == 1 && return initialparameters(rng, getfield(l, layers[1]))
    return ComponentArray(NamedTuple{layers}(initialparameters.(rng, getfield.((l,), layers))))
end

function initialstates(rng::AbstractRNG, l::AbstractExplicitContainerLayer{layers}) where {layers}
    length(layers) == 1 && return initialstates(rng, getfield(l, layers[1]))
    return NamedTuple{layers}(initialstates.(rng, getfield.((l,), layers)))
end

parameterlength(l::AbstractExplicitContainerLayer{layers}) where {layers} =
    sum(parameterlength, getfield.((l,), layers))

statelength(l::AbstractExplicitContainerLayer{layers}) where {layers} =
    sum(statelength, getfield.((l,), layers))

# Test Mode
testmode(st::NamedTuple, mode::Bool=true) = update_state(st, :training, !mode)

trainmode(x::Any, mode::Bool=true) = testmode(x, !mode)

# Utilities to modify global state
function update_state(st::NamedTuple, key::Symbol, value; layer_check=_default_layer_check(key))
    function _update_state(st, key::Symbol, value)
        return Setfield.set(st, Setfield.PropertyLens{key}(), value)
    end
    return fmap(_st -> _update_state(_st, key, value), st; exclude=layer_check)
end

function _default_layer_check(key)
    _default_layer_check_closure(x) = hasmethod(keys, (typeof(x),)) ? key ∈ keys(x) : false
    return _default_layer_check_closure
end
