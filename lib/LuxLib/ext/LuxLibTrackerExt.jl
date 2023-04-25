module LuxLibTrackerExt

if isdefined(Base, :get_extension)
    using Tracker
    import Tracker: @grad, data, nobacksies, track, TrackedArray, TrackedVector, TrackedReal
else
    using ..Tracker
    import ..Tracker: @grad, data, nobacksies, track, TrackedArray, TrackedVector,
                      TrackedReal
end
using NNlib, LuxLib
import LuxLib: AA, AV, _batchnorm_cudnn!, _get_batchnorm_statistics, FP_32_64, ∂∅,
               __is_tracked
import ChainRulesCore as CRC

# NNlib: batched_mul
for T1 in (:AbstractArray, :TrackedArray), T2 in (:AbstractArray, :TrackedArray)
    __is_tracked(T1, T2) || continue

    @eval NNlib.batched_mul(x::$T1, y::$T2) = track(batched_mul, x, y)
end

@grad function NNlib.batched_mul(A::AA{<:Any, 3}, B::AA{<:Any, 3})
    function batched_mul_pullback(Δ)
        tmp = batched_mul(Δ, batched_adjoint(data(B)))
        ΔA = size(A, 3) == 1 ? sum(tmp; dims=3) : tmp
        tmp = batched_mul(batched_adjoint(data(A)), Δ)
        ΔB = size(B, 3) == 1 ? sum(tmp; dims=3) : tmp
        return nobacksies(:batched_mul, (ΔA, ΔB))
    end
    return batched_mul(data(A), data(B)), batched_mul_pullback
end

# NNlib: gather
function NNlib.gather!(dst::AA, src::TrackedArray, idx::AA)
    return track(NNlib.gather!, dst, src, idx)
end

@grad function NNlib.gather!(dst::AA, src::AA, idx::AA)
    function gather!_pullback(Δ)
        return nobacksies(:gather, (nothing, NNlib.∇gather_src(Δ, size(src), idx), nothing))
    end
    return NNlib.gather!(dst, data(src), idx), gather!_pullback
end

# Base.repeat
Base.repeat(x::TrackedArray, counts...) = track(Base.repeat, x, counts...)

@grad function Base.repeat(x, counts...)
    y, pullback_function = CRC.rrule(Base.repeat, data(x), counts...)
    function repeat_pullback(Δ)
        _, res... = pullback_function(Δ)
        return nobacksies(:repeat, map(x -> x == ∂∅ ? nothing : CRC.unthunk(x), res))
    end
    return y, repeat_pullback
end

# utils.jl
function LuxLib._copy_autodiff_barrier(x::Union{TrackedArray, TrackedReal})
    return LuxLib._copy_autodiff_barrier(data(x))
end

LuxLib._get_backend(x::TrackedArray) = LuxLib._get_backend(data(x))

# api/dropout.jl
LuxLib._dropout_fptype(x::TrackedArray) = LuxLib._dropout_fptype(data(x))

# api/groupnorm.jl
for T1 in (:TrackedArray, :AbstractArray),
    T2 in (:TrackedVector, :AbstractVector),
    T3 in (:TrackedVector, :AbstractVector)

    __is_tracked(T1, T2, T3) || continue

    @eval function LuxLib.groupnorm(x::$T1{T, 4}, scale::$T2{T}, bias::$T3{T}; groups::Int,
                                    epsilon::Real) where {T <: FP_32_64}
        return track(LuxLib.groupnorm, x, scale, bias; groups, epsilon)
    end
end

@grad function LuxLib.groupnorm(x::AA{T, 4}, scale::AV{T}, bias::AV{T}; groups::Int,
                                epsilon::Real) where {T <: FP_32_64}
    LuxLib._assert_same_backend(data(x), data(scale), data(bias))
    if length(scale) != length(bias) != size(x, 3)
        throw(ArgumentError("Length of `scale` and `bias` must be equal to the number of channels (N - 1 dim of the input array)."))
    end
    if size(x, 3) % groups != 0
        throw(ArgumentError("Number of channels $(size(x, 3)) must be divisible by the number of groups $groups."))
    end

    y, mu, rsig = LuxLib._groupnorm(data(x), groups, data(scale), data(bias), epsilon)
    function groupnorm_pullback(dy)
        dx, dscale, dbias = LuxLib._dgroupnorm(dy, y, data(x), groups, data(scale),
                                               data(bias), mu, rsig)
        return nobacksies(:groupnorm, (dx, dscale, dbias))
    end
    return y, groupnorm_pullback
end

end
