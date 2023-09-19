module LuxLibForwardDiffExt

using ForwardDiff, LuxLib
import ForwardDiff: Dual

# dropout
function LuxLib._dropout_fptype(x::AbstractArray{<:Dual})
    return ForwardDiff.valtype(eltype(x))
end

# Convolutions: We might want to capture these furthur down in `conv!`
# NOTE: In principle we can concatenate all of the partials along the batch dimension
#       and cut down substantially on the time to compute jacobians.
for op in [:conv, :depthwiseconv]
    op! = Symbol("$(op)!")

    @eval function NNlib.$(op)(x::AbstractArray{<:Dual{Tag, V, P}, N},
        w::AbstractArray{<:Real, N}, cdims::ConvDims; kwargs...) where {N, Tag, V, P}
        x_ = ForwardDiff.value.(x)

        y = $(op)(x_, w, cdims; kwargs...)
        dys = ntuple(i -> $(op)(ForwardDiff.partials.(x, i), w, cdims; kwargs...), P)

        return map((yᵢ, dyᵢ...) -> Dual{Tag, V, P}(yᵢ, ForwardDiff.Partials(dyᵢ)), y,
            dys...)
    end

    @eval function NNlib.$(op)(x::AbstractArray{<:Real, N},
        w::AbstractArray{<:Dual{Tag, V, P}, N},
        cdims::ConvDims; kwargs...) where {N, Tag, V, P}
        w_ = ForwardDiff.value.(w)

        y = $(op)(x, w_, cdims; kwargs...)
        dys = ntuple(i -> $(op)(x, ForwardDiff.partials.(w, i), cdims; kwargs...), P)

        return map((yᵢ, dyᵢ...) -> Dual{Tag, V, P}(yᵢ, ForwardDiff.Partials(dyᵢ)), y,
            dys...)
    end

    @eval function NNlib.$(op)(x::AbstractArray{<:Dual{Tag, Vₓ, P}, N},
        w::AbstractArray{<:Dual{Tag, Vₚ, P}, N}, cdims::ConvDims;
        kwargs...) where {N, Tag, Vₓ, Vₚ, P}
        x_ = ForwardDiff.value.(x)
        w_ = ForwardDiff.value.(w)

        y = $(op)(x_, w_, cdims; kwargs...)

        dys₁ = ntuple(_ -> similar(x_, Vₓ, NNlib.output_size(cdims)...,
                NNlib.channels_out(cdims), size(x, N)), P)
        dys₂ = ntuple(_ -> similar(x_, Vₓ, NNlib.output_size(cdims)...,
                NNlib.channels_out(cdims), size(x, N)), P)
        for i in 1:P
            $(op!)(dys₁[i], ForwardDiff.partials.(x, i), w_, cdims; kwargs...)
            $(op!)(dys₂[i], x_, ForwardDiff.partials.(w, i), cdims; kwargs...)
            dys₁[i] .+= dys₂[i]
        end

        # Technically it should `promote_type(Vₓ, Vₚ)` but this causes GPU compilation
        # failure. We will assume it matches the type of the input.
        return map((yᵢ, dyᵢ...) -> Dual{Tag, Vₓ, P}(yᵢ, ForwardDiff.Partials(dyᵢ)), y,
            dys₁...)
    end
end

function LuxLib._drop_forwarddiff_partials(x::AbstractArray{<:Dual})
    return ForwardDiff.value.(x)
end

end
