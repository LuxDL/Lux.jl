module LuxLibTrackercuDNNExt

# cuDNN not loaded but it is needed for the batchnorm_cudnn implementation
using CUDA: CUDA, CuArray, CuVector
using LuxLib: LuxLib
using Tracker: Tracker, TrackedVector, TrackedArray

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
    # NOTE: The following returns a tracked tuple so we can't do `first` on it
    x_ = LuxLib.batchnorm_cudnn(rm, rv, scale, bias, x, momentum, epsilon, training)[1]
    return x_, (; running_mean=rm, running_var=rv)
end

for RM in (:TrackedVector, :Nothing, :AbstractVector),
    RV in (:TrackedVector, :Nothing, :AbstractVector),
    S in (:TrackedVector, :Nothing, :AbstractVector),
    B in (:TrackedVector, :Nothing, :AbstractVector),
    XT in (:TrackedArray, :AbstractArray)

    LuxLib.__is_tracked(RM, RV, S, B, XT) || continue

    @eval LuxLib.@tracker_grad_from_chainrules LuxLib.batchnorm_cudnn(
        running_mean::$RM, running_var::$RV, scale::$S, bias::$B,
        x::$XT, momentum::Real, eps::Real, training::Val)
end

end
