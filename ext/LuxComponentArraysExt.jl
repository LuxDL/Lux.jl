module LuxComponentArraysExt

isdefined(Base, :get_extension) ? (using ComponentArrays) : (using ..ComponentArrays)

using Functors, Lux, Optimisers
import TruncatedStacktraces: @truncate_stacktrace
import ChainRulesCore as CRC

@inline function Lux._getproperty(x::ComponentArray, ::Val{prop}) where {prop}
    return prop in propertynames(x) ? getproperty(x, prop) : nothing
end

function Functors.functor(::Type{<:ComponentArray}, c)
    return NamedTuple{propertynames(c)}(getproperty.((c,), propertynames(c))),
           ComponentArray
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
    res = ComponentArray(nt)
    function CA_NT_pullback(Δ::AbstractArray)
        if length(Δ) == length(res)
            return (CRC.NoTangent(), NamedTuple(ComponentArray(vec(Δ), getaxes(res))))
        end
        error("Got pullback input of shape $(size(Δ)) & type $(typeof(Δ)) for output " *
              "of shape $(size(res)) & type $(typeof(res))")
        return nothing
    end
    CA_NT_pullback(Δ::ComponentArray) = (@show Δ; (CRC.NoTangent(), NamedTuple(Δ)))
    return res, CA_NT_pullback
end

# Definitely needs an upstream :P
@truncate_stacktrace ComponentArray 1

end
