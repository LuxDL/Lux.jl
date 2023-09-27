module LuxLibLuxCUDAForwardDiffExt

using LuxLib, LuxCUDA, ForwardDiff, Statistics
import ForwardDiff: Dual
import LuxLib: AA, FP_32_64

const CUDNN_FD_BN_ARRAY_TYPE{Tag, V, P} = Union{CuArray{<:Dual{Tag, V, P}, 2},
    CuArray{<:Dual{Tag, V, P}, 4},
    CuArray{<:Dual{Tag, V, P}, 5}}
const BNParamType = Union{Nothing, CuVector{<:FP_32_64}}

# This dispatch is exclusively for when `x` is a `Dual`. When any of the other arguments
# contains Dual elements, the slower fallback implementation will be used!
function LuxLib.batchnorm(x::CUDNN_FD_BN_ARRAY_TYPE{Tag, V, P}, scale::BNParamType,
    bias::BNParamType, running_mean::BNParamType, running_var::BNParamType; momentum::Real,
    training::Val, epsilon::Real) where {Tag, V, P}
    x_ = ForwardDiff.value.(x)
    rm, rv = LuxLib._get_batchnorm_statistics(x_, running_mean, running_var, training)

    y, xmean, xivar = LuxLib._batchnorm_cudnn!(rm, rv, scale, bias, x_, momentum, epsilon,
        training)

    # Note: There will be a slight discrepancy in the answer if CUDNN batchnorm doesn't add
    #       epsilon into the ivar
    rdims = LuxLib._get_batchnorm_reduce_dims(x_)
    dims = LuxLib._unwrap_val(rdims)
    γ = LuxLib._reshape_into_proper_shape(scale, x)
    α = ifelse(γ === nothing, 1, γ) .* sqrt.(xivar)
    dy = ntuple(_ -> similar(y), P)
    for i in 1:P
        xₚ = ForwardDiff.partials.(x, i)
        μₚ = mean(xₚ; dims=LuxLib._unwrap_val(rdims))
        sx_ = (x_ .- xmean)
        σ²ₚ = mean(2 .* (xₚ .- μₚ) .* sx_; dims)
        @. dy[i] = α * (xₚ - μₚ - (sx_ * xivar * σ²ₚ / 2))
    end

    return (map((yᵢ, dyᵢ...) -> Dual{Tag, V, P}(yᵢ, ForwardDiff.Partials(dyᵢ)), y, dy...),
        (; running_mean=rm, running_var=rv))
end

end
