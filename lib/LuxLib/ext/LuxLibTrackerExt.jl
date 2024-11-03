module LuxLibTrackerExt

using FastClosures: @closure
using LuxLib: LuxLib, Utils, Traits
using NNlib: NNlib
using Static: True, StaticBool
using Tracker: Tracker, TrackedArray, TrackedReal, TrackedVector

tracker_data(x) = Tracker.data(x)
tracker_data(x::NNlib.BatchedAdjoint) = NNlib.batched_adjoint(tracker_data(parent(x)))
tracker_data(x::NNlib.BatchedTranspose) = NNlib.batched_transpose(tracker_data(parent(x)))

# batched matrix multiplication
import LuxLib.Impl: batched_matmul
import NNlib: batched_mul

## Without the rules on BatchedAdjoint and BatchedTranspose, we end up constructing
## AbstractMatrix{<:TrackedReal} which is not efficient
for T1 in (:AbstractArray, :TrackedArray), T2 in (:AbstractArray, :TrackedArray)
    Utils.is_tracked(T1, T2) || continue

    for op in (:batched_mul, :batched_matmul)
        @eval begin
            $(op)(x::$T1{<:Any, 3}, y::$T2{<:Any, 3}) = Tracker.track($(op), x, y)
            function $(op)(x::NNlib.BatchedAdjOrTrans{<:Any, <:$T1{<:Any, 3}},
                    y::$T2{<:Any, 3})
                return Tracker.track($(op), x, y)
            end
            function $(op)(
                    x::$T1{<:Any, 3}, y::NNlib.BatchedAdjOrTrans{<:Any, <:$T2{<:Any, 3}})
                return Tracker.track($(op), x, y)
            end
            function $(op)(x::NNlib.BatchedAdjOrTrans{<:Any, <:$T1{<:Any, 3}},
                    y::NNlib.BatchedAdjOrTrans{<:Any, <:$T2{<:Any, 3}})
                return Tracker.track($(op), x, y)
            end
        end
    end
end

for op in (:batched_mul, :batched_matmul)
    @eval Tracker.@grad function $(op)(x, y)
        z = $(op)(tracker_data(x), tracker_data(y))
        ∇batched_matmul = @closure Δ -> begin
            ∂x = $(op)(tracker_data(Δ), NNlib.batched_adjoint(tracker_data(y)))
            size(x, 3) == 1 && (∂x = sum(∂x; dims=3))
            ∂y = $(op)(NNlib.batched_adjoint(tracker_data(x)), tracker_data(Δ))
            size(y, 3) == 1 && (∂y = sum(∂y; dims=3))
            return Tracker.nobacksies(:batched_matmul, (∂x, ∂y))
        end
        return z, ∇batched_matmul
    end
end

# NNlib: gather
Tracker.@grad_from_chainrules NNlib.gather!(
    dst::AbstractArray, src::TrackedArray, idx::AbstractArray)

# Base.repeat
Tracker.@grad_from_chainrules Base.repeat(x::TrackedArray, counts...)

# Base.selectdim -- Needed for GPUArrays
Base.selectdim(x::TrackedArray, d::Integer, i) = Tracker.track(selectdim, x, d, i)

Tracker.@grad function Base.selectdim(x::AbstractArray, d::Integer, i)
    x′ = Tracker.data(x)
    y = selectdim(x′, d, i)
    ∇selectdim = @closure Δ -> begin
        ∂x = zero(x′)
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
        γ::$S, β::$B, x::$XT, rμ::$RM, rσ²::$RV, m::Real, ϵ::Real, training::StaticBool)
end

# Utils extensions
Utils.remove_tracking(x::TrackedReal) = Tracker.data(x)
Utils.remove_tracking(x::TrackedArray) = Tracker.data(x)
Utils.remove_tracking(x::AbstractArray{<:TrackedReal}) = Tracker.data.(x)
Utils.remove_tracking(::Type{<:TrackedReal{T}}) where {T} = Utils.remove_tracking(T)

Utils.within_autodiff(::TrackedReal) = True()
Utils.within_autodiff(::TrackedArray) = True()
Utils.within_autodiff(::AbstractArray{<:TrackedReal}) = True()

# Traits extensions
Traits.is_tracked(::Type{<:TrackedReal}) = True()

end
