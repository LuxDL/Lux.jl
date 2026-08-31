module LuxLibMooncakecuDNNExt

using LuxLib: LuxLib, Impl
using CUDA: CuArray, DenseCuArray, DenseCuVector
using Static: StaticBool
using Mooncake:
    Mooncake, MinimalCtx, ReverseMode, @is_primitive, CoDual, NoRData, arrayify, primal
import Mooncake: rrule!!

# batchnorm_cudnn wraps cudnnBatchNormalizationForwardTraining, an untraceable C call.
# We cache xμ/xσ⁻² from the forward pass for the pullback, since recomputing them
# would be non-deterministic and break Mooncake's Reuse check.

@is_primitive(
    MinimalCtx,
    ReverseMode,
    Tuple{
        typeof(Impl.batchnorm_cudnn),
        DenseCuVector{T},
        DenseCuVector{T},
        DenseCuArray{T},
        Any,
        Any,
        Any,
        Any,
        StaticBool,
    } where {T<:Union{Float32,Float64}},
)
function rrule!!(
    ::CoDual{typeof(Impl.batchnorm_cudnn)},
    γ_::CoDual{<:DenseCuVector{T}},
    β_::CoDual{<:DenseCuVector{T}},
    x_::CoDual{<:DenseCuArray{T}},
    rμ_::CoDual,
    rσ²_::CoDual,
    m_::CoDual,
    ϵ_::CoDual,
    training_::CoDual{<:StaticBool},
) where {T<:Union{Float32,Float64}}
    γ, dγ = arrayify(γ_)
    β, dβ = arrayify(β_)
    x, dx = arrayify(x_)
    prμ = primal(rμ_)
    prσ² = primal(rσ²_)
    pm = primal(m_)
    pϵ = primal(ϵ_)
    ptraining = primal(training_)

    y, xμ, xσ⁻² = Impl.batchnorm_cudnn(γ, β, x, prμ, prσ², pm, pϵ, ptraining)

    dy_out = zero(y)
    dxμ_out = zero(xμ)
    dxσ⁻²_out = zero(xσ⁻²)

    function batchnorm_cudnn_pb!!(::NoRData)
        ∂γ, ∂β, ∂x = Impl.∇batchnorm_cudnn(γ, β, x, dy_out, prμ, prσ², xμ, xσ⁻², pϵ)
        dγ .+= ∂γ
        dβ .+= ∂β
        dx .+= ∂x
        return NoRData(),
        NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), zero(pm), zero(pϵ),
        NoRData()
    end

    return CoDual((y, xμ, xσ⁻²), (dy_out, dxμ_out, dxσ⁻²_out)), batchnorm_cudnn_pb!!
end

end
