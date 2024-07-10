module LuxReverseDiffExt

using ADTypes: ADTypes, AutoReverseDiff
using ArrayInterface: ArrayInterface
using Lux: Lux, LuxCPUDevice
using Lux.Experimental: TrainingBackendCache, TrainState
using ReverseDiff: ReverseDiff, TrackedArray, TrackedReal, @grad_from_chainrules

# AoS to SoA conversion
function Lux.apply(
        m::Lux.AbstractExplicitLayer, x::AbstractArray{<:ReverseDiff.TrackedReal}, ps, st)
    @warn "Lux.apply(m::Lux.AbstractExplicitLayer, \
           x::AbstractArray{<:ReverseDiff.TrackedReal}, ps, st) input was corrected to \
           Lux.apply(m::Lux.AbstractExplicitLayer, x::ReverseDiff.TrackedArray}, ps, \
           st).\n\n\
           1. If this was not the desired behavior overload the dispatch on `m`.\n\n\
           2. This might have performance implications. Check which layer was causing this \
              problem using `Lux.Experimental.@debug_mode`." maxlog=1
    return Lux.apply(m, reshape(ArrayInterface.aos_to_soa(x), size(x)), ps, st)
end

## Prevent an infinite loop
Lux.apply(m::Lux.AbstractExplicitLayer, x::TrackedArray, ps, st) = m(x, ps, st)

@inline Lux.__eltype(::TrackedArray{T}) where {T} = T
@inline Lux.__eltype(::TrackedReal{T}) where {T} = T
@inline Lux.__eltype(::AbstractArray{<:TrackedReal{T}}) where {T} = T

@inline Lux.__reverse(x::TrackedArray; dims=:) = ArrayInterface.aos_to_soa(reverse(x; dims))
@inline function Lux.__reverse(x::AbstractArray{<:TrackedReal}; dims=:)
    return ArrayInterface.aos_to_soa(reverse(x; dims))
end

# multigate: avoid soa formation
@inline function Lux._gate(x::TrackedArray{T, R, 1}, h::Int, n::Int) where {T, R}
    return x[Lux._gate(h, n)]
end
@inline function Lux._gate(x::AbstractVector{<:TrackedReal}, h::Int, n::Int)
    return ArrayInterface.aos_to_soa(view(x, Lux._gate(h, n)))
end
@inline function Lux._gate(x::TrackedArray{T, R, 2}, h::Int, n::Int) where {T, R}
    return x[Lux._gate(h, n), :]
end
@inline function Lux._gate(x::AbstractMatrix{<:TrackedReal}, h::Int, n::Int)
    return ArrayInterface.aos_to_soa(view(x, Lux._gate(h, n), :))
end

@inline function Lux.__convert_eltype(::Type{T}, x::AbstractArray{<:TrackedReal}) where {T}
    @warn "`Lux.__convert_eltype` doesn't support converting element types of ReverseDiff \
           `TrackedReal` arrays. Currently this is a no-op." maxlog=1
    return x
end

include("rules.jl")
include("training.jl")

end
