# Base Type
## API:
### initialparameters(rng, l)
### initialstates(rng, l)
### createcache(rng, l, x, ps, st)
### outputdescriptor(l, x, ps, st)
### parameterlength(l)
### statelength(l)
### cachesize(l, x, ps, st)
### l(x, ps, st)
### l(x, ps, st, cache)

abstract type AbstractExplicitLayer end

for f in (:initialparameters, :initialstates, :createcache)
    @eval begin
        $(f)(::AbstractRNG, ::Any, args...; kwargs...) = NamedTuple()
        $(f)(l, args...; kwargs...) = $(f)(Random.GLOBAL_RNG, l, args...; kwargs...)
        function $(f)(rng::AbstractRNG, l::NamedTuple{fields}, args...; kwargs...) where {fields}
            return NamedTuple{fields}(map(x -> $(f)(rng, x, args...; kwargs...), values(l)))
        end
    end
end

for (f, g) in ((:parameterlength, :initialparameters), (:statelength, :initialstates), (:cachesize, :createcache))
    @eval begin
        $(f)(l::AbstractExplicitLayer, args...; kwargs...) = $(f)($(g)(l), args...; kwargs...)
        $(f)(x::NamedTuple, args...; kwargs...) = nestedtupleofarrayslength(x)
    end
end

setup(rng::AbstractRNG, l::AbstractExplicitLayer) = (initialparameters(rng, l), initialstates(rng, l))
setup(l::AbstractExplicitLayer) = setup(Random.GLOBAL_RNG, l)

function setup(rng::AbstractRNG, l::AbstractExplicitLayer, input)
    ps, st = setup(rng, l)
    return (ps, st, createcache(rng, l, input, ps, st))
end
setup(l::AbstractExplicitLayer, input) = setup(Random.GLOBAL_RNG, l, input)

function outputdescriptor(l::AbstractExplicitLayer, x, ps::NamedTuple, st::NamedTuple)
    # Please implement `outputsize`, the default is quite slow
    return zero(first(l(x, ps, st)))
end

nestedtupleofarrayslength(t::Any) = 1
nestedtupleofarrayslength(t::AbstractArray) = length(t)
function nestedtupleofarrayslength(t::Union{NamedTuple,Tuple})
    length(t) == 0 && return 0
    return sum(nestedtupleofarrayslength, t)
end

function (model::AbstractExplicitLayer)(x, ps::NamedTuple, st::NamedTuple, ::NamedTuple)
    return model(x, ps, st)
end

apply(model::AbstractExplicitLayer, x, ps::NamedTuple, st::NamedTuple) = model(x, ps, st)
apply(model::AbstractExplicitLayer, x, ps::NamedTuple, st::NamedTuple, cache::NamedTuple) = model(x, ps, st, cache)

# Test Mode
testmode(st::NamedTuple, mode::Bool=true) = update_state(st, :training, !mode)

testmode(x::Any, mode::Bool=true) = x

testmode(m::AbstractExplicitLayer, mode::Bool=true) = testmode(initialstates(m), mode)

trainmode(x::Any, mode::Bool=true) = testmode(x, !mode)

# Utilities to modify global state
function update_state(st::NamedTuple, key::Symbol, value; layer_check=_default_layer_check(key))
    function _update_state(st, key::Symbol, value)
        return Setfield.set(st, Setfield.PropertyLens{key}(), value)
    end
    return fmap(_st -> _update_state(_st, key, value), st; exclude=layer_check)
end

function _default_layer_check(key)
    _default_layer_check_closure(x) = key ∈ keys(x)
    _default_layer_check_closure(x::Symbol) = false
end
