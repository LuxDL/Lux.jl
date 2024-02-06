module LuxComponentArraysExt

using ComponentArrays, Functors, Lux, Optimisers
import TruncatedStacktraces: @truncate_stacktrace
import ChainRulesCore as CRC

@generated function Lux._getproperty(x::ComponentArray{T, N, A, Tuple{Ax}},
        ::Val{v}) where {v, T, N, A, Ax <: ComponentArrays.AbstractAxis}
    names = propertynames(ComponentArrays.indexmap(Ax))
    return v ∈ names ? :(x.$v) : :(nothing)
end

function Functors.functor(::Type{<:ComponentArray}, c)
    return (NamedTuple{propertynames(c)}(getproperty.((c,), propertynames(c))),
        ComponentArray)
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
function Lux._merge(p::AbstractArray, ca::ComponentArray)
    @assert length(p) == 0
    return ca
end
function Lux._merge(ca::ComponentArray, p::AbstractArray)
    @assert length(p) == 0
    return ca
end

function Lux._pairs(ca::ComponentArray)
    pnames = propertynames(ca)
    vals = NamedTuple{pnames}(getproperty.((ca,), pnames))
    return Iterators.Pairs(vals, pnames)
end

# Empty NamedTuple: Hack to avoid breaking precompilation
function ComponentArrays.ComponentArray(data::Vector{Any}, axes::Tuple{FlatAxis})
    length(data) == 0 && return ComponentArray(Float32[], axes)
    return ComponentArray{Any, 1, typeof(data), typeof(axes)}(data, axes)
end

# Parameter Sharing
function Lux.Experimental._parameter_structure(ps::ComponentArray)
    return Lux.Experimental._parameter_structure(NamedTuple(ps))
end

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
    CA_NT_pullback(Δ::ComponentArray) = CRC.NoTangent(), NamedTuple(Δ)
    return res, CA_NT_pullback
end

# Definitely needs an upstream :P
@truncate_stacktrace ComponentArray 1

end
