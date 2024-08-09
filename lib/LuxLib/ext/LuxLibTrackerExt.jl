module LuxLibTrackerExt

using FastClosures: @closure
using LuxLib: LuxLib, Utils, Traits
using NNlib: NNlib
using Static: True, StaticBool
using Tracker: Tracker, TrackedArray, TrackedReal, TrackedVector

# NNlib: batched_mul
for T1 in (:AbstractArray, :TrackedArray), T2 in (:AbstractArray, :TrackedArray)
    Utils.is_tracked(T1, T2) || continue

    @eval Tracker.@grad_from_chainrules NNlib.batched_mul(
        x::$T1{<:Number, 3}, y::$T2{<:Number, 3})
    @eval Tracker.@grad_from_chainrules LuxLib.Impl.batched_matmul(
        x::$T1{<:Number, 3}, y::$T2{<:Number, 3})
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

# Impl: batchnorm_cudnn
## cuDNN batchnorm -- the chain rule gets defined once cuDNN is loaded
for RM in (:TrackedVector, :Nothing, :AbstractVector),
    RV in (:TrackedVector, :Nothing, :AbstractVector),
    S in (:TrackedVector, :Nothing, :AbstractVector),
    B in (:TrackedVector, :Nothing, :AbstractVector),
    XT in (:TrackedArray, :AbstractArray)

    Utils.is_tracked(RM, RV, S, B, XT) || continue

    @eval Tracker.@grad_from_chainrules LuxLib.Impl.batchnorm_cudnn(
        γ::$RM, β::$RV, x::$XT, rμ::$RM, rσ²::$RV, m::Real, ϵ::Real, training::StaticBool)
end

# Utils extensions
Utils.remove_tracking(x::TrackedReal) = Tracker.data(x)
Utils.remove_tracking(x::TrackedArray) = Tracker.data(x)
Utils.remove_tracking(x::AbstractArray{<:TrackedReal}) = Tracker.data.(x)
Utils.remove_tracking(::Type{<:TrackedReal{T}}) where {T} = Utils.remove_tracking(T)

# Traits extensions
Traits.is_tracked(::Type{<:TrackedReal}) = True()

end
