module LuxComponentArraysExt

isdefined(Base, :get_extension) ? (using ComponentArrays) : (using ..ComponentArrays)

using Functors, Lux, Optimisers, Zygote
import ChainRulesCore as CRC

@inline function Lux._getproperty(x::ComponentArray, ::Val{prop}) where {prop}
    return prop in propertynames(x) ? getproperty(x, prop) : nothing
end

function Functors.functor(::Type{<:ComponentArray}, c)
    return NamedTuple{propertynames(c)}(getproperty.((c,), propertynames(c))),
           ComponentArray
end

# Zygote Fixes
function Zygote.accum(x::ComponentArray, ys::ComponentArray...)
    return ComponentArray(Zygote.accum(getdata(x), getdata.(ys)...), getaxes(x))
end

# Optimisers Fixes
Optimisers.setup(opt::AbstractRule, ps::ComponentArray) = Optimisers.setup(opt, getdata(ps))

function Optimisers.update(tree, ps::ComponentArray, gs::ComponentArray)
    tree, ps_new = Optimisers.update(tree, getdata(ps), getdata(gs))
    return tree, ComponentArray(ps_new, getaxes(ps))
end

function Optimisers.update!(tree::Optimisers.Leaf, ps::ComponentArray, gs::ComponentArray)
    tree, ps_new = Optimisers.update!(tree, getdata(ps), getdata(gs))
    return tree, ComponentArray(ps_new, getaxes(ps))
end

# Freezing
Lux._merge(nt1::ComponentArray, nt2::NamedTuple) = merge(NamedTuple(nt1), nt2)
Lux._merge(nt1::NamedTuple, nt2::ComponentArray) = merge(nt1, NamedTuple(nt2))

# Parameter Sharing
Lux._parameter_structure(ps::ComponentArray) = Lux._parameter_structure(NamedTuple(ps))

# CRC + CA Temporary Patch -- Needs to be upstreamed
function CRC.rrule(::Type{ComponentArray}, nt::NamedTuple)
    return ComponentArray(nt), Δ -> (CRC.NoTangent(), NamedTuple(Δ))
end

end
