module LuxLibTrackerExt

using ChainRulesCore: ChainRulesCore
using FastClosures: @closure
using LuxLib: LuxLib
using NNlib: NNlib
using Static: True
using Tracker: Tracker, TrackedArray, TrackedReal, TrackedVector

const CRC = ChainRulesCore

# NNlib: batched_mul
for T1 in (:AbstractArray, :TrackedArray), T2 in (:AbstractArray, :TrackedArray)
    LuxLib.__is_tracked(T1, T2) || continue

    @eval Tracker.@grad_from_chainrules NNlib.batched_mul(x::$T1, y::$T2)
end

# NNlib: gather
Tracker.@grad_from_chainrules NNlib.gather!(
    dst::AbstractArray, src::TrackedArray, idx::AbstractArray)

# Base.repeat
Tracker.@grad_from_chainrules Base.repeat(x::TrackedArray, counts...)

# Base.selectdim -- Needed for GPUArrays
Base.selectdim(x::TrackedArray, d::Integer, i) = Tracker.track(selectdim, x, d, i)

Tracker.@grad function Base.selectdim(x::AbstractArray, d::Integer, i)
    x_ = Tracker.data(x)
    y = selectdim(x_, d, i)
    ∇selectdim = @closure Δ -> begin
        ∂x = zero(x_)
        selectdim(∂x, d, i) .= Tracker.data(Δ)
        return ∂x, nothing, nothing
    end
    return y, ∇selectdim
end

# cuDNN batchnorm -- the chain rule gets defined once cuDNN is loaded
for RM in (:TrackedVector, :Nothing, :AbstractVector),
    RV in (:TrackedVector, :Nothing, :AbstractVector),
    S in (:TrackedVector, :Nothing, :AbstractVector),
    B in (:TrackedVector, :Nothing, :AbstractVector),
    XT in (:TrackedArray, :AbstractArray)

    LuxLib.__is_tracked(RM, RV, S, B, XT) || continue

    @eval Tracker.@grad_from_chainrules LuxLib.batchnorm_cudnn(
        running_mean::$RM, running_var::$RV, scale::$S, bias::$B,
        x::$XT, momentum::Real, eps::Real, training::Val)
end

LuxLib.remove_tracking(x::TrackedReal) = Tracker.data(x)
LuxLib.remove_tracking(x::TrackedArray) = Tracker.data(x)
LuxLib.remove_tracking(x::AbstractArray{<:TrackedReal}) = Tracker.data.(x)
LuxLib.remove_tracking(::Type{<:TrackedReal{T}}) where {T} = LuxLib.remove_tracking(T)

LuxLib.is_tracked(::Type{<:TrackedReal}) = True()

end
