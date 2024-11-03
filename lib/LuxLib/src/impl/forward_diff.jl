for op in [:conv, :depthwiseconv, :∇conv_data, :∇conv_filter]
    patched_op = op !== :depthwiseconv ? eval(op) : getfield(NNlib, op)

    @eval function NNlib.$(op)(x1::AbstractArray{<:ForwardDiff.Dual{Tag, V, P}, N},
            x2::AbstractArray{<:Real, N}, cdims::NNlib.ConvDims;
            kwargs...) where {N, Tag, V, P}
        value_fn(x) = ForwardDiff.value(Tag, x)
        partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

        y = $(patched_op)(value_fn.(x1), x2, cdims; kwargs...)
        dys = ntuple(i -> $(patched_op)(partial_fn.(x1, i), x2, cdims; kwargs...), P)

        partials = ForwardDiff.Partials.(tuple.(dys...))
        return ForwardDiff.Dual{Tag, eltype(y), P}.(y, partials)
    end

    @eval function NNlib.$(op)(x1::AbstractArray{<:Real, N},
            x2::AbstractArray{<:ForwardDiff.Dual{Tag, V, P}, N},
            cdims::NNlib.ConvDims; kwargs...) where {N, Tag, V, P}
        value_fn(x) = ForwardDiff.value(Tag, x)
        partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

        y = $(patched_op)(x1, value_fn.(x2), cdims; kwargs...)
        dys = ntuple(i -> $(patched_op)(x1, partial_fn.(x2, i), cdims; kwargs...), P)

        partials = ForwardDiff.Partials.(tuple.(dys...))
        return ForwardDiff.Dual{Tag, eltype(y), P}.(y, partials)
    end

    @eval function NNlib.$(op)(x1::AbstractArray{<:ForwardDiff.Dual{Tag, Vₓ, P}, N},
            x2::AbstractArray{<:ForwardDiff.Dual{Tag, Vₚ, P}, N},
            cdims::NNlib.ConvDims; kwargs...) where {N, Tag, Vₓ, Vₚ, P}
        value_fn(x) = ForwardDiff.value(Tag, x)
        partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

        x1_data, x2_data = value_fn.(x1), value_fn.(x2)

        y = $(patched_op)(x1_data, x2_data, cdims; kwargs...)

        dys₁ = ntuple(P) do i
            dys₁ᵢ = $(patched_op)(partial_fn.(x1, i), x2_data, cdims; kwargs...)
            dys₂ᵢ = $(patched_op)(x1_data, partial_fn.(x2, i), cdims; kwargs...)
            dys₁ᵢ .+= dys₂ᵢ
            return dys₁ᵢ
        end

        partials = ForwardDiff.Partials.(tuple.(dys₁...))
        return ForwardDiff.Dual{Tag, eltype(y), P}.(y, partials)
    end
end
