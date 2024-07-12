module LuxLibTrackercuDNNExt

# cuDNN not loaded but it is needed for the batchnorm_cudnn implementation
using CUDA: CUDA, CuArray, CuVector
using LuxLib: LuxLib
using Tracker: Tracker, TrackedVector, TrackedArray

# api/batchnorm.jl
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

end
