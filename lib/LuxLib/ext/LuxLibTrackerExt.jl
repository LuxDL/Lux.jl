module LuxLibTrackerExt

if isdefined(Base, :get_extension)
    using Tracker
    import Tracker: @grad, data, nobacksies, track, TrackedArray, TrackedVector, TrackedReal
else
    using ..Tracker
    import ..Tracker: @grad, data, nobacksies, track, TrackedArray, TrackedVector,
                      TrackedReal
end
using LuxCUDA
using NNlib, LuxLib
using LuxLib: _CUDNN_BATCHNORM_FLOAT, _GROUPNORM_IMPL_FLOAT
import ChainRulesCore as CRC

# NNlib: batched_mul
for T1 in (:AbstractArray, :TrackedArray), T2 in (:AbstractArray, :TrackedArray)
    T1 == :AbstractArray && T2 == :AbstractArray && continue

    @eval NNlib.batched_mul(x::$T1, y::$T2) = track(batched_mul, x, y)
end

@grad function NNlib.batched_mul(A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3})
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
function NNlib.gather!(dst::AbstractArray, src::TrackedArray, idx::AbstractArray)
    return track(NNlib.gather!, dst, src, idx)
end

@grad function NNlib.gather!(dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
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
        return nobacksies(:repeat,
                          map(x -> x isa CRC.NoTangent ? nothing : CRC.unthunk(x), res))
    end
    return y, repeat_pullback
end

# utils.jl
function LuxLib._copy_autodiff_barrier(x::Union{TrackedArray, TrackedReal})
    return LuxLib._copy_autodiff_barrier(data(x))
end

LuxLib._get_backend(x::TrackedArray) = LuxLib._get_backend(data(x))

# api/batchnorm.jl
_TR_BN = Union{TrackedArray{<:Any, <:Any, <:CuArray{<:_CUDNN_BATCHNORM_FLOAT, 2}},
               TrackedArray{<:Any, <:Any, <:CuArray{<:_CUDNN_BATCHNORM_FLOAT, 4}},
               TrackedArray{<:Any, <:Any, <:CuArray{<:_CUDNN_BATCHNORM_FLOAT, 5}}}

_TR_BN_VEC = TrackedArray{<:Any, <:Any, <:CuVector{<:_CUDNN_BATCHNORM_FLOAT}}

function LuxLib.batchnorm(x::_TR_BN, scale::Union{_TR_BN_VEC, Nothing},
                          bias::Union{_TR_BN_VEC, Nothing},
                          running_mean::Union{_TR_BN_VEC, Nothing},
                          running_var::Union{_TR_BN_VEC, Nothing}; momentum::Real,
                          training::Val, epsilon::Real)
    rm, rv = LuxLib._get_batchnorm_statistics(x, running_mean, running_var, training)

    x_ = LuxLib._batchnorm_cudnn!(rm, rv, scale, bias, x, momentum, epsilon, training)
    return x_, (; running_mean=rm, running_var=rv)
end

for RM in (:TrackedVector, :AbstractVector),
    RV in (:TrackedVector, :AbstractVector),
    S in (:TrackedVector, :Nothing, :AbstractVector),
    B in (:TrackedVector, :Nothing, :AbstractVector),
    XT in (:TrackedArray, :AbstractArray)

    RM == :AbstractVector &&
        RV == :AbstractVector &&
        (S == :AbstractVector || S == Nothing) &&
        (B == :AbstractVector || B == Nothing) &&
        XT == :AbstractArray &&
        continue

    @eval function LuxLib._batchnorm_cudnn!(running_mean::$RM, running_var::$RV, scale::$S,
                                            bias::$B, x::$XT, momentum, eps, training::Val)
        return track(LuxLib._batchnorm_cudnn!, running_mean, running_var, scale, bias, x,
                     momentum, eps, training)
    end
end

@grad function LuxLib._batchnorm_cudnn!(running_mean, running_var, scale, bias, x, momentum,
                                        eps, training)
    y = LuxLib._batchnorm_cudnn!(data(running_mean), data(running_var), data(scale),
                                 data(bias), data(x), momentum, eps, training)
    function _batchnorm_cudnn!_pullback(dy)
        dg, db, dx = NNlibCUDA.∇batchnorm(data(scale), data(bias), data(x), dy,
                                          data(running_mean), data(running_var), momentum;
                                          eps, training)
        return (nothing, nothing, dg, db, dx, nothing, nothing, nothing)
    end
    return y, _batchnorm_cudnn!_pullback
end

# api/dropout.jl
LuxLib._dropout_fptype(x::TrackedArray) = LuxLib._dropout_fptype(data(x))

# api/groupnorm.jl
for T1 in (:TrackedArray, :AbstractArray),
    T2 in (:TrackedVector, :AbstractVector),
    T3 in (:TrackedVector, :AbstractVector)

    T1 == :AbstractArray && T2 == :AbstractVector && T3 == :AbstractVector && continue

    @eval function LuxLib.groupnorm(x::$T1{T, 4}, scale::$T2{T}, bias::$T3{T}; groups::Int,
                                    epsilon::Real) where {T <: _GROUPNORM_IMPL_FLOAT}
        return track(LuxLib.groupnorm, x, scale, bias; groups, epsilon)
    end
end

@grad function LuxLib.groupnorm(x::AbstractArray{T, 4}, scale::AbstractVector{T},
                                bias::AbstractVector{T}; groups::Int,
                                epsilon::Real) where {T <: _GROUPNORM_IMPL_FLOAT}
    LuxLib._assert_same_backend(data(x), data(scale), data(bias))
    if length(scale) != length(bias) != size(x, 3)
        throw(ArgumentError("Length of `scale` and `bias` must be equal to the number of
                             channels (N - 1 dim of the input array)."))
    end
    if size(x, 3) % groups != 0
        throw(ArgumentError("Number of channels $(size(x, 3)) must be divisible by the
                             number of groups $groups."))
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
