module LuxLibForwardDiffExt

using ForwardDiff: ForwardDiff
using GPUArraysCore: AnyGPUArray
using LuxLib: LuxLib
using NNlib: NNlib, ConvDims

# dropout
@inline function LuxLib._dropout_fptype(x::AbstractArray{<:ForwardDiff.Dual})
    return ForwardDiff.valtype(eltype(x))
end

# Convolutions: We might want to capture these furthur down in `conv!`
# NOTE: In principle we can concatenate all of the partials along the batch dimension
#       and cut down substantially on the time to compute jacobians.
# Here we should be broadcasting with `Tag` for safety but that breaks GPU compilation.
for op in [:conv, :depthwiseconv, :∇conv_data, :∇conv_filter]
    op! = Symbol("$(op)!")

    @eval function NNlib.$(op)(x1::AbstractArray{<:ForwardDiff.Dual{Tag, V, P}, N},
            x2::AbstractArray{<:Real, N}, cdims::NNlib.ConvDims;
            kwargs...) where {N, Tag, V, P}
        x1_data = ForwardDiff.value.(x1)

        y = NNlib.$(op)(x1_data, x2, cdims; kwargs...)
        dys = ntuple(
            i -> NNlib.$(op)(ForwardDiff.partials.(x1, i), x2, cdims; kwargs...), P)

        return map(
            (yᵢ, dyᵢ...) -> ForwardDiff.Dual{Tag, V, P}(yᵢ, ForwardDiff.Partials(dyᵢ)),
            y, dys...)
    end

    @eval function NNlib.$(op)(x1::AbstractArray{<:Real, N},
            x2::AbstractArray{<:ForwardDiff.Dual{Tag, V, P}, N},
            cdims::NNlib.ConvDims; kwargs...) where {N, Tag, V, P}
        x2_data = ForwardDiff.value.(x2)

        y = NNlib.$(op)(x1, x2_data, cdims; kwargs...)
        dys = ntuple(
            i -> NNlib.$(op)(x1, ForwardDiff.partials.(x2, i), cdims; kwargs...), P)

        return map(
            (yᵢ, dyᵢ...) -> ForwardDiff.Dual{Tag, V, P}(yᵢ, ForwardDiff.Partials(dyᵢ)),
            y, dys...)
    end

    @eval function NNlib.$(op)(x1::AbstractArray{<:ForwardDiff.Dual{Tag, Vₓ, P}, N},
            x2::AbstractArray{<:ForwardDiff.Dual{Tag, Vₚ, P}, N},
            cdims::NNlib.ConvDims; kwargs...) where {N, Tag, Vₓ, Vₚ, P}
        x1_data = ForwardDiff.value.(x1)
        x2_data = ForwardDiff.value.(x2)

        y = NNlib.$(op)(x1_data, x2_data, cdims; kwargs...)

        dys₁ = ntuple(P) do i
            dys₁ᵢ = NNlib.$(op)(ForwardDiff.partials.(x1, i), x2_data, cdims; kwargs...)
            dys₂ᵢ = NNlib.$(op)(x1_data, ForwardDiff.partials.(x2, i), cdims; kwargs...)
            dys₁ᵢ .+= dys₂ᵢ
            return dys₁ᵢ
        end

        # Technically it should `promote_type(Vₓ, Vₚ)` but this causes GPU compilation
        # failure. We will assume it matches the type of the input.
        return map(
            (yᵢ, dyᵢ...) -> ForwardDiff.Dual{Tag, Vₓ, P}(yᵢ, ForwardDiff.Partials(dyᵢ)),
            y, dys₁...)
    end
end

# TODO: We would want to use the fused versions here, but for now we will just dispatch the
#       duals to the generic implementation for GPUArrays
function LuxLib.fused_conv_bias_activation(σ::F, weight::AnyGPUArray{<:ForwardDiff.Dual, N},
        x::AnyGPUArray{xT, N}, bias::Nothing, cdims::ConvDims) where {F, N, xT}
    return LuxLib._generic_conv_bias_activation(σ, weight, x, bias, cdims)
end
function LuxLib.fused_conv_bias_activation(
        σ::F, weight::AnyGPUArray{wT, N}, x::AnyGPUArray{<:ForwardDiff.Dual, N},
        bias::Nothing, cdims::ConvDims) where {F, N, wT}
    return LuxLib._generic_conv_bias_activation(σ, weight, x, bias, cdims)
end
function LuxLib.fused_conv_bias_activation(σ::F, weight::AnyGPUArray{<:ForwardDiff.Dual, N},
        x::AnyGPUArray{<:ForwardDiff.Dual, N}, bias::Nothing, cdims::ConvDims) where {F, N}
    return LuxLib._generic_conv_bias_activation(σ, weight, x, bias, cdims)
end
function LuxLib.fused_conv_bias_activation(
        σ::F, weight::AnyGPUArray{<:ForwardDiff.Dual, N}, x::AnyGPUArray{xT, N},
        bias::AnyGPUArray{bT, N}, cdims::ConvDims) where {F, N, xT, bT}
    return LuxLib._generic_conv_bias_activation(σ, weight, x, bias, cdims)
end
function LuxLib.fused_conv_bias_activation(
        σ::F, weight::AnyGPUArray{wT, N}, x::AnyGPUArray{<:ForwardDiff.Dual, N},
        bias::AnyGPUArray{bT, N}, cdims::ConvDims) where {F, wT, bT, N}
    return LuxLib._generic_conv_bias_activation(σ, weight, x, bias, cdims)
end
function LuxLib.fused_conv_bias_activation(σ::F, weight::AnyGPUArray{<:ForwardDiff.Dual, N},
        x::AnyGPUArray{<:ForwardDiff.Dual, N},
        bias::AnyGPUArray{bT, N}, cdims::ConvDims) where {F, N, bT}
    return LuxLib._generic_conv_bias_activation(σ, weight, x, bias, cdims)
end
function LuxLib.fused_conv_bias_activation(
        σ::F, weight::AnyGPUArray{<:ForwardDiff.Dual, N}, x::AnyGPUArray{xT, N},
        bias::AnyGPUArray{<:ForwardDiff.Dual, N}, cdims::ConvDims) where {F, N, xT}
    return LuxLib._generic_conv_bias_activation(σ, weight, x, bias, cdims)
end
function LuxLib.fused_conv_bias_activation(
        σ::F, weight::AnyGPUArray{wT, N}, x::AnyGPUArray{<:ForwardDiff.Dual, N},
        bias::AnyGPUArray{<:ForwardDiff.Dual, N}, cdims::ConvDims) where {F, N, wT}
    return LuxLib._generic_conv_bias_activation(σ, weight, x, bias, cdims)
end
function LuxLib.fused_conv_bias_activation(σ::F, weight::AnyGPUArray{<:ForwardDiff.Dual, N},
        x::AnyGPUArray{<:ForwardDiff.Dual, N},
        bias::AnyGPUArray{<:ForwardDiff.Dual, N}, cdims::ConvDims) where {F, N}
    return LuxLib._generic_conv_bias_activation(σ, weight, x, bias, cdims)
end

function LuxLib._drop_forwarddiff_partials(x::AbstractArray{<:ForwardDiff.Dual})
    return ForwardDiff.value.(x)
end

end
