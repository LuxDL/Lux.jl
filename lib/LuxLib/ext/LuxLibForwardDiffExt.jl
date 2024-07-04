module LuxLibForwardDiffExt

using ForwardDiff: ForwardDiff
using LuxLib: LuxLib
using NNlib: NNlib

LuxLib.__has_dual(::ForwardDiff.Dual) = true
LuxLib.__has_dual(::AbstractArray{<:ForwardDiff.Dual}) = true

# dropout
@inline function LuxLib._dropout_fptype(x::AbstractArray{<:ForwardDiff.Dual})
    return ForwardDiff.valtype(eltype(x))
end

# Convolutions: We might want to capture these further down in `conv!`
# NOTE: In principle we can concatenate all of the partials along the batch dimension
#       and cut down substantially on the time to compute jacobians.
# Here we should be broadcasting with `Tag` for safety but that breaks GPU compilation.
for op in [:conv, :depthwiseconv, :∇conv_data, :∇conv_filter]
    luxlibop = Symbol("__$(op)")

    @eval function NNlib.$(op)(x1::AbstractArray{<:ForwardDiff.Dual{Tag, V, P}, N},
            x2::AbstractArray{<:Real, N}, cdims::NNlib.ConvDims;
            kwargs...) where {N, Tag, V, P}
        x1_data = ForwardDiff.value.(x1)

        y = LuxLib.$(luxlibop)(x1_data, x2, cdims; kwargs...)
        dys = ntuple(
            i -> LuxLib.$(luxlibop)(ForwardDiff.partials.(x1, i), x2, cdims; kwargs...), P)

        return map(
            (yᵢ, dyᵢ...) -> ForwardDiff.Dual{Tag, V, P}(yᵢ, ForwardDiff.Partials(dyᵢ)),
            y, dys...)
    end

    @eval function NNlib.$(op)(x1::AbstractArray{<:Real, N},
            x2::AbstractArray{<:ForwardDiff.Dual{Tag, V, P}, N},
            cdims::NNlib.ConvDims; kwargs...) where {N, Tag, V, P}
        x2_data = ForwardDiff.value.(x2)

        y = LuxLib.$(luxlibop)(x1, x2_data, cdims; kwargs...)
        dys = ntuple(
            i -> LuxLib.$(luxlibop)(x1, ForwardDiff.partials.(x2, i), cdims; kwargs...), P)

        return map(
            (yᵢ, dyᵢ...) -> ForwardDiff.Dual{Tag, V, P}(yᵢ, ForwardDiff.Partials(dyᵢ)),
            y, dys...)
    end

    @eval function NNlib.$(op)(x1::AbstractArray{<:ForwardDiff.Dual{Tag, Vₓ, P}, N},
            x2::AbstractArray{<:ForwardDiff.Dual{Tag, Vₚ, P}, N},
            cdims::NNlib.ConvDims; kwargs...) where {N, Tag, Vₓ, Vₚ, P}
        x1_data = ForwardDiff.value.(x1)
        x2_data = ForwardDiff.value.(x2)

        y = LuxLib.$(luxlibop)(x1_data, x2_data, cdims; kwargs...)

        dys₁ = ntuple(P) do i
            dys₁ᵢ = LuxLib.$(luxlibop)(
                ForwardDiff.partials.(x1, i), x2_data, cdims; kwargs...)
            dys₂ᵢ = LuxLib.$(luxlibop)(
                x1_data, ForwardDiff.partials.(x2, i), cdims; kwargs...)
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

# Don't try to promote the input types
@inline function LuxLib.__gpu_get_weight_input(
        ::Type{T}, ::Type{<:ForwardDiff.Dual}, weight, x) where {T}
    return LuxLib.__materialize_subarray(weight), LuxLib.__materialize_subarray(x)
end
@inline function LuxLib.__gpu_get_weight_input(
        ::Type{<:ForwardDiff.Dual}, ::Type{T}, weight, x) where {T}
    return LuxLib.__materialize_subarray(weight), LuxLib.__materialize_subarray(x)
end
@inline function LuxLib.__gpu_get_weight_input(
        ::Type{<:ForwardDiff.Dual}, ::Type{<:ForwardDiff.Dual}, weight, x)
    return LuxLib.__materialize_subarray(weight), LuxLib.__materialize_subarray(x)
end

@inline function LuxLib._drop_forwarddiff_partials(x::AbstractArray{<:ForwardDiff.Dual})
    return ForwardDiff.value.(x)
end

end
