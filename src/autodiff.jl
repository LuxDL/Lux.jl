Base.zero(s::NamedTuple{(), Tuple{}}) = s

Base.zero(::Symbol) = Symbol()

Base.zero(nt::NamedTuple{fields}) where {fields} = NamedTuple{fields}(zero.(values(nt)))

# Layers are stateless so we can simply return that
Base.zero(l::AbstractExplicitLayer) = l

ChainRulesCore.rrule(::typeof(istraining)) = true, _ -> (NoTangent(),)

ChainRulesCore.@non_differentiable _update_stats!(::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any)

function ChainRulesCore.rrule(::typeof(Val), x)
    valx = Val(x)
    val_pullback(::Val{Δ}) where Δ = NoTangent(), Δ
    return valx, val_pullback
end

# Yota Compatibility
## TODO: Add conditional dep on Yota
import Yota: ungetindex

function ungetindex(x::NamedTuple{fields}, dy, i::Int) where {fields}
    @eval @set $x.$(fields[i]) = $dy
end

## Yota uses .+ to accumulate gradients which doesn't work well
## where broadcasting is reserved, for eg, with NamedTuple 
function Base.broadcast(::typeof(+), nt1::NamedTuple{fields}, nt2::NamedTuple{fields}) where {fields}
    accum(x::NamedTuple, y::NamedTuple) = map(accum, x, y)
    accum(x::AbstractArray, y::AbstractArray) = x .+ y
    accum(x::Tangent{<:NamedTuple{f}}, y::NamedTuple{f}) where {f} = NamedTuple{f}(ntuple(i -> accum(x[i], y[i]), length(f)))
    accum(x::NamedTuple{f}, y::Tangent{<:NamedTuple{f}}) where {f} = NamedTuple{f}(ntuple(i -> accum(x[i], y[i]), length(f)))

    return NamedTuple{fields}(map(accum, nt1, nt2))
end
