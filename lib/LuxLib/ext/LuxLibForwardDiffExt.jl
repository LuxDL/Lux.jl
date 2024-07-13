module LuxLibForwardDiffExt

using ForwardDiff: ForwardDiff
using LuxLib: LuxLib
using LuxDeviceUtils: AbstractLuxDevice, AbstractLuxGPUDevice
using NNlib: NNlib

LuxLib.__has_dual(::ForwardDiff.Dual) = true
LuxLib.__has_dual(::AbstractArray{<:ForwardDiff.Dual}) = true

# Convolutions: We might want to capture these further down in `conv!`
# NOTE: In principle we can concatenate all of the partials along the batch dimension
#       and cut down substantially on the time to compute jacobians.
for op in [:conv, :depthwiseconv, :∇conv_data, :∇conv_filter]
    luxlibop = Symbol("__$(op)")

    @eval function NNlib.$(op)(x1::AbstractArray{<:ForwardDiff.Dual{Tag, V, P}, N},
            x2::AbstractArray{<:Real, N}, cdims::NNlib.ConvDims;
            kwargs...) where {N, Tag, V, P}
        value_fn(x) = ForwardDiff.value(Tag, x)
        partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

        y = LuxLib.$(luxlibop)(value_fn.(x1), x2, cdims; kwargs...)
        dys = ntuple(i -> LuxLib.$(luxlibop)(partial_fn.(x1, i), x2, cdims; kwargs...), P)

        partials = ForwardDiff.Partials.(tuple.(dys...))
        return ForwardDiff.Dual{Tag, V, P}.(y, partials)
    end

    @eval function NNlib.$(op)(x1::AbstractArray{<:Real, N},
            x2::AbstractArray{<:ForwardDiff.Dual{Tag, V, P}, N},
            cdims::NNlib.ConvDims; kwargs...) where {N, Tag, V, P}
        value_fn(x) = ForwardDiff.value(Tag, x)
        partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

        y = LuxLib.$(luxlibop)(x1, value_fn.(x2), cdims; kwargs...)
        dys = ntuple(i -> LuxLib.$(luxlibop)(x1, partial_fn.(x2, i), cdims; kwargs...), P)

        partials = ForwardDiff.Partials.(tuple.(dys...))
        return ForwardDiff.Dual{Tag, V, P}.(y, partials)
    end

    @eval function NNlib.$(op)(x1::AbstractArray{<:ForwardDiff.Dual{Tag, Vₓ, P}, N},
            x2::AbstractArray{<:ForwardDiff.Dual{Tag, Vₚ, P}, N},
            cdims::NNlib.ConvDims; kwargs...) where {N, Tag, Vₓ, Vₚ, P}
        value_fn(x) = ForwardDiff.value(Tag, x)
        partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

        x1_data, x2_data = value_fn.(x1), value_fn.(x2)

        y = LuxLib.$(luxlibop)(x1_data, x2_data, cdims; kwargs...)

        dys₁ = ntuple(P) do i
            dys₁ᵢ = LuxLib.$(luxlibop)(partial_fn.(x1, i), x2_data, cdims; kwargs...)
            dys₂ᵢ = LuxLib.$(luxlibop)(x1_data, partial_fn.(x2, i), cdims; kwargs...)
            dys₁ᵢ .+= dys₂ᵢ
            return dys₁ᵢ
        end

        partials = ForwardDiff.Partials.(tuple.(dys₁...))
        return ForwardDiff.Dual{Tag, promote_type(Vₓ, Vₚ), P}.(y, partials)
    end
end

# Don't try to promote the input types
function LuxLib.__get_conv_input_weight(
        ::Type{<:AbstractLuxGPUDevice}, ::Type{<:ForwardDiff.Dual},
        ::Type{T}, x, weight) where {T}
    return LuxLib.__materialize_subarray(x), LuxLib.__materialize_subarray(weight)
end
function LuxLib.__get_conv_input_weight(::Type{<:AbstractLuxGPUDevice}, ::Type{T},
        ::Type{<:ForwardDiff.Dual}, x, weight) where {T}
    return LuxLib.__materialize_subarray(x), LuxLib.__materialize_subarray(weight)
end
function LuxLib.__get_conv_input_weight(
        ::Type{<:AbstractLuxGPUDevice}, ::Type{<:ForwardDiff.Dual},
        ::Type{<:ForwardDiff.Dual}, x, weight)
    return LuxLib.__materialize_subarray(x), LuxLib.__materialize_subarray(weight)
end

LuxLib.__value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
LuxLib.__value(x::AbstractArray{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)
LuxLib.__value(::Type{<:ForwardDiff.Dual{T}}) where {T} = LuxLib.__value(T)

end
