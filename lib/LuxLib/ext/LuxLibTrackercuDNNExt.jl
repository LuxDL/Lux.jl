module LuxLibTrackercuDNNExt

using FastClosures: @closure
# cuDNN not loaded but it is needed for the batchnorm_cudnn implementation
using CUDA: CUDA, CuArray, CuVector, CuPtr
using LuxLib: LuxLib
using Tracker: Tracker, @grad, TrackedArray, TrackedVector, TrackedReal

# api/batchnorm.jl
const TR_CUDNN_BN_ARRAY_TYPE = Union{
    TrackedArray{<:Any, <:Any, <:CuArray{<:Union{Float32, Float64}, 2}},
    TrackedArray{<:Any, <:Any, <:CuArray{<:Union{Float32, Float64}, 4}},
    TrackedArray{<:Any, <:Any, <:CuArray{<:Union{Float32, Float64}, 5}}}
const TR_BNParamType = Union{
    Nothing, TrackedArray{<:Any, <:Any, <:CuVector{<:Union{Float32, Float64}}},
    CuVector{<:Union{Float32, Float64}}}

function LuxLib.batchnorm(
        x::TR_CUDNN_BN_ARRAY_TYPE, scale::TR_BNParamType, bias::TR_BNParamType,
        running_mean::TR_BNParamType, running_var::TR_BNParamType;
        momentum::Real, training::Val, epsilon::Real)
    rm, rv = LuxLib._get_batchnorm_statistics(x, running_mean, running_var, training)
    x_ = first(LuxLib.batchnorm_cudnn(rm, rv, scale, bias, x, momentum, epsilon, training))
    return x_, (; running_mean=rm, running_var=rv)
end

for RM in (:TrackedVector, :Nothing, :AbstractVector),
    RV in (:TrackedVector, :Nothing, :AbstractVector),
    S in (:TrackedVector, :Nothing, :AbstractVector),
    B in (:TrackedVector, :Nothing, :AbstractVector),
    XT in (:TrackedArray, :AbstractArray)

    LuxLib.__is_tracked(RM, RV, S, B, XT) || continue

    @eval function LuxLib.batchnorm_cudnn(running_mean::$RM, running_var::$RV, scale::$S,
            bias::$B, x::$XT, momentum, eps, training::Val)
        return Tracker.track(LuxLib.batchnorm_cudnn, running_mean, running_var,
            scale, bias, x, momentum, eps, training)
    end
end

@inline __make_nothing(x) = x
@inline __make_nothing(::CuPtr{Nothing}) = 0

@grad function LuxLib.batchnorm_cudnn(
        running_mean, running_var, scale, bias, x, momentum, eps, training)
    y, xmean, xivar = LuxLib.batchnorm_cudnn(
        Tracker.data(running_mean), Tracker.data(running_var), Tracker.data(scale),
        Tracker.data(bias), Tracker.data(x), momentum, eps, training)
    ∇batchnorm_cudnn_internal = @closure Δ -> begin
        ∂y = first(Δ)
        ∂g, ∂b, ∂x = ∇batchnorm_cudnn(
            Tracker.data(scale), Tracker.data(bias), Tracker.data(x), ∂y,
            Tracker.data(running_mean), Tracker.data(running_var), xmean, xivar; ϵ=eps)
        return (nothing, nothing, ∂g, ∂b, ∂x, nothing, nothing, nothing)
    end
    return (y, __make_nothing(xmean), __make_nothing(xivar)), ∇batchnorm_cudnn_internal
end

end
