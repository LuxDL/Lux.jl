# Base Type
abstract type AbstractExplicitLayer end

initialparameters(::AbstractRNG, ::Any) = NamedTuple()
initialparameters(l) = initialparameters(Random.GLOBAL_RNG, l)
initialstates(::AbstractRNG, ::Any) = NamedTuple()
initialstates(l) = initialstates(Random.GLOBAL_RNG, l)

function initialparameters(rng::AbstractRNG, l::NamedTuple)
    return NamedTuple{Tuple(collect(keys(l)))}(initialparameters.((rng,), values(l)))
end
initialstates(rng::AbstractRNG, l::NamedTuple) = NamedTuple{Tuple(collect(keys(l)))}(initialstates.((rng,), values(l)))

setup(rng::AbstractRNG, l::AbstractExplicitLayer) = (initialparameters(rng, l), initialstates(rng, l))
setup(l::AbstractExplicitLayer) = setup(Random.GLOBAL_RNG, l)

nestedtupleofarrayslength(t::Any) = 1
nestedtupleofarrayslength(t::AbstractArray) = length(t)
function nestedtupleofarrayslength(t::Union{NamedTuple,Tuple})
    length(t) == 0 && return 0
    return sum(nestedtupleofarrayslength, t)
end

parameterlength(l::AbstractExplicitLayer) = parameterlength(initialparameters(l))
statelength(l::AbstractExplicitLayer) = statelength(initialstates(l))
parameterlength(ps::NamedTuple) = nestedtupleofarrayslength(ps)
statelength(st::NamedTuple) = nestedtupleofarrayslength(st)

apply(model::AbstractExplicitLayer, x, ps::NamedTuple, s::NamedTuple) = model(x, ps, s)

# Test Mode
function testmode(states::NamedTuple, mode::Bool=true)
    updated_states = []
    for (k, v) in pairs(states)
        if k == :training
            push!(updated_states, k => !mode)
            continue
        end
        push!(updated_states, k => testmode(v, mode))
    end
    return (; updated_states...)
end

testmode(x::Any, mode::Bool=true) = x

testmode(m::AbstractExplicitLayer, mode::Bool=true) = testmode(initialstates(m), mode)

trainmode(x::Any, mode::Bool=true) = testmode(x, !mode)