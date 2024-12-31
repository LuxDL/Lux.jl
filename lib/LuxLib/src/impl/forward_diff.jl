for op in (:conv, :depthwiseconv, :∇conv_data, :∇conv_filter)
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

for op in (:logsoftmax, :softmax)
    dual_op = Symbol(op, :_dual)
    @eval function NNlib.$(op)(
            x::AbstractArray{<:ForwardDiff.Dual{Tag, T, P}}; dims=1) where {Tag, T, P}
        return Impl.$(dual_op)(x; dims)
    end
end

function softmax_dual(
        x::AbstractArray{<:ForwardDiff.Dual{Tag, T, P}}; dims=1) where {Tag, T, P}
    value_fn(x) = ForwardDiff.value(Tag, x)
    partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

    x_data = value_fn.(x)

    y = NNlib.softmax(x_data; dims)
    dysᵢ = ntuple(P) do i
        v = partial_fn.(x, i)
        return y .* (v .- sum(y .* v; dims))
    end

    partials = ForwardDiff.Partials.(tuple.(dysᵢ...))
    return ForwardDiff.Dual{Tag, eltype(y), P}.(y, partials)
end

function logsoftmax_dual(
        x::AbstractArray{<:ForwardDiff.Dual{Tag, T, P}}; dims=1) where {Tag, T, P}
    value_fn(x) = ForwardDiff.value(Tag, x)
    partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

    x_data = value_fn.(x)

    y = NNlib.softmax(x_data; dims)
    dysᵢ = ntuple(P) do i
        v = partial_fn.(x, i)
        return v .- sum(y .* v; dims)
    end

    partials = ForwardDiff.Partials.(tuple.(dysᵢ...))
    return ForwardDiff.Dual{Tag, eltype(y), P}.(y, partials)
end

@eval function NNlib.meanpool(
        x::AbstractArray{<:ForwardDiff.Dual{Tag, T, P}}, pdims::NNlib.PoolDims;
        kwargs...) where {Tag, T, P}
    value_fn(x) = ForwardDiff.value(Tag, x)
    partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

    y = NNlib.meanpool(value_fn.(x), pdims; kwargs...)
    dysᵢ = ntuple(P) do i
        return NNlib.meanpool(partial_fn.(x, i), pdims; kwargs...)
    end

    partials = ForwardDiff.Partials.(tuple.(dysᵢ...))
    return ForwardDiff.Dual{Tag, eltype(y), P}.(y, partials)
end

function NNlib.∇meanpool(
        dy::AbstractArray{<:ForwardDiff.Dual{Tag, T1, P}},
        y::AbstractArray{<:ForwardDiff.Dual{Tag, T1, P}},
        x::AbstractArray{<:ForwardDiff.Dual{Tag, T2, P}},
        pdims::NNlib.PoolDims; kwargs...) where {Tag, T1, T2, P}
    value_fn(x) = ForwardDiff.value(Tag, x)
    partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

    dy_data, y_data, x_data = value_fn.(dy), value_fn.(y), value_fn.(x)

    dx = NNlib.∇meanpool(dy_data, y_data, x_data, pdims; kwargs...)
    dysᵢ = ntuple(P) do i
        ∇y₁ = NNlib.∇meanpool(partial_fn.(dy, i), y_data, x_data, pdims; kwargs...)
        ∇y₂ = NNlib.∇meanpool(dy_data, partial_fn.(y, i), x_data, pdims; kwargs...)
        @. ∇y₁ = (∇y₁ + ∇y₂) * partial_fn(x, i)
        return ∇y₁
    end

    partials = ForwardDiff.Partials.(tuple.(dysᵢ...))
    return ForwardDiff.Dual{Tag, eltype(dx), P}.(dx, partials)
end
