module ForwardDiffExt

using ForwardDiff: ForwardDiff
using LuxLib: LuxLib, Impl, Utils, Traits
using NNlib: NNlib
using Static: False, True

# ================================ Utils Extensions ================================
# ofeltype_array for Dual types
function Utils.ofeltype_array(
    ::Type{T}, x::AbstractArray{<:ForwardDiff.Dual{Tag,T,N}}
) where {Tag,T,N}
    return x
end
function Utils.ofeltype_array(
    ::Type{T}, x::AbstractArray{<:ForwardDiff.Dual{Tag,T2,N}}
) where {Tag,T,T2,N}
    return ForwardDiff.Dual{Tag,T,N}.(x)
end

# remove_tracking for Dual types
Utils.remove_tracking(x::ForwardDiff.Dual) = ForwardDiff.value(x)
Utils.remove_tracking(x::AbstractArray{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)
function Utils.remove_tracking(::Type{<:ForwardDiff.Dual{Tag,T}}) where {Tag,T}
    return Utils.remove_tracking(T)
end

# eltype_mismatch for Dual types
function Utils.eltype_mismatch(
    ::Type{T}, ::Type{<:ForwardDiff.Dual{Tag,T,N}}
) where {Tag,T,N}
    return False()
end
function Utils.eltype_mismatch(
    ::Type{<:ForwardDiff.Dual{Tag,T,N}}, ::Type{T}
) where {Tag,T,N}
    return False()
end

# within_autodiff for Dual types
Utils.within_autodiff(::ForwardDiff.Dual) = True()
Utils.within_autodiff(::AbstractArray{<:ForwardDiff.Dual}) = True()

# ================================ Traits Extensions ================================
Traits.has_dual(::Type{<:ForwardDiff.Dual}) = True()

# Conv operations for Dual arrays
for op in (:conv, :depthwiseconv, :∇conv_data, :∇conv_filter)
    patched_op = getfield(NNlib, op)
    cdim_types = if op === :depthwiseconv
        (NNlib.ConvDims, NNlib.DepthwiseConvDims)
    else
        (NNlib.ConvDims,)
    end

    for CDims in cdim_types
        @eval function NNlib.$(op)(
            x1::AbstractArray{<:ForwardDiff.Dual{Tag,V,P},N},
            x2::AbstractArray{<:Real,N},
            cdims::$(CDims);
            kwargs...,
        ) where {N,Tag,V,P}
            value_fn(x) = ForwardDiff.value(Tag, x)
            partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

            y = $(patched_op)(value_fn.(x1), x2, cdims; kwargs...)
            dys = ntuple(i -> $(patched_op)(partial_fn.(x1, i), x2, cdims; kwargs...), P)

            partials = ForwardDiff.Partials.(tuple.(dys...))
            return ForwardDiff.Dual{Tag,eltype(y),P}.(y, partials)
        end

        @eval function NNlib.$(op)(
            x1::AbstractArray{<:Real,N},
            x2::AbstractArray{<:ForwardDiff.Dual{Tag,V,P},N},
            cdims::$(CDims);
            kwargs...,
        ) where {N,Tag,V,P}
            value_fn(x) = ForwardDiff.value(Tag, x)
            partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

            y = $(patched_op)(x1, value_fn.(x2), cdims; kwargs...)
            dys = ntuple(i -> $(patched_op)(x1, partial_fn.(x2, i), cdims; kwargs...), P)

            partials = ForwardDiff.Partials.(tuple.(dys...))
            return ForwardDiff.Dual{Tag,eltype(y),P}.(y, partials)
        end

        @eval function NNlib.$(op)(
            x1::AbstractArray{<:ForwardDiff.Dual{Tag,Vₓ,P},N},
            x2::AbstractArray{<:ForwardDiff.Dual{Tag,Vₚ,P},N},
            cdims::$(CDims);
            kwargs...,
        ) where {N,Tag,Vₓ,Vₚ,P}
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
            return ForwardDiff.Dual{Tag,eltype(y),P}.(y, partials)
        end
    end
end

# Softmax/logsoftmax dispatches
for op in (:logsoftmax, :softmax)
    dual_op = Symbol(op, :_dual)
    @eval function NNlib.$(op)(
        x::AbstractArray{<:ForwardDiff.Dual{Tag,T,P}}; dims=1
    ) where {Tag,T,P}
        return Impl.$(dual_op)(x; dims)
    end
end

function Impl.softmax_dual(
    x::AbstractArray{<:ForwardDiff.Dual{Tag,T,P}}; dims=1
) where {Tag,T,P}
    value_fn(x) = ForwardDiff.value(Tag, x)
    partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

    x_data = value_fn.(x)

    y = NNlib.softmax(x_data; dims)
    dysᵢ = ntuple(P) do i
        v = partial_fn.(x, i)
        return y .* (v .- sum(y .* v; dims))
    end

    partials = ForwardDiff.Partials.(tuple.(dysᵢ...))
    return ForwardDiff.Dual{Tag,eltype(y),P}.(y, partials)
end

function Impl.logsoftmax_dual(
    x::AbstractArray{<:ForwardDiff.Dual{Tag,T,P}}; dims=1
) where {Tag,T,P}
    value_fn(x) = ForwardDiff.value(Tag, x)
    partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

    x_data = value_fn.(x)

    y = NNlib.softmax(x_data; dims)
    dysᵢ = ntuple(P) do i
        v = partial_fn.(x, i)
        return v .- sum(y .* v; dims)
    end

    partials = ForwardDiff.Partials.(tuple.(dysᵢ...))
    return ForwardDiff.Dual{Tag,eltype(y),P}.(y, partials)
end

# Meanpool for Dual arrays
function NNlib.meanpool(
    x::AbstractArray{<:ForwardDiff.Dual{Tag,T,P}}, pdims::NNlib.PoolDims; kwargs...
) where {Tag,T,P}
    value_fn(x) = ForwardDiff.value(Tag, x)
    partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

    y = NNlib.meanpool(value_fn.(x), pdims; kwargs...)
    dysᵢ = ntuple(P) do i
        return NNlib.meanpool(partial_fn.(x, i), pdims; kwargs...)
    end

    partials = ForwardDiff.Partials.(tuple.(dysᵢ...))
    return ForwardDiff.Dual{Tag,eltype(y),P}.(y, partials)
end

function NNlib.∇meanpool(
    dy::AbstractArray{<:ForwardDiff.Dual{Tag1,T1,P1}},
    y::AbstractArray{<:ForwardDiff.Dual{Tag2,T2,P2}},
    x::AbstractArray{<:ForwardDiff.Dual{Tag3,T3,P3}},
    pdims::NNlib.PoolDims;
    kwargs...,
) where {Tag1,Tag2,Tag3,T1,T2,T3,P1,P2,P3}
    value_fn(z) = ForwardDiff.value(Tag1, z)
    partial_fn(z, i) = ForwardDiff.partials(Tag1, z, i)

    dy_data, y_data, x_data = value_fn.(dy), ForwardDiff.value.(y), ForwardDiff.value.(x)

    dx = NNlib.∇meanpool(dy_data, y_data, x_data, pdims; kwargs...)
    dysᵢ = ntuple(
        i -> NNlib.∇meanpool(partial_fn.(dy, i), y_data, x_data, pdims; kwargs...), P1
    )

    partials = ForwardDiff.Partials.(tuple.(dysᵢ...))
    return ForwardDiff.Dual{Tag1,eltype(dx),P1}.(dx, partials)
end

function NNlib.∇meanpool(
    dy::AbstractArray{<:Real},
    y::AbstractArray{<:ForwardDiff.Dual{Tag1,T1,P1}},
    x::AbstractArray{<:ForwardDiff.Dual{Tag2,T2,P2}},
    pdims::NNlib.PoolDims;
    kwargs...,
) where {Tag1,Tag2,T1,T2,P1,P2}
    dx = NNlib.∇meanpool(dy, ForwardDiff.value.(y), ForwardDiff.value.(x), pdims; kwargs...)
    return ForwardDiff.Dual{Tag2,eltype(dx),P2}.(dx)
end

function NNlib.∇meanpool(
    dy::AbstractArray{<:ForwardDiff.Dual{Tag,T1,P}},
    y::AbstractArray{<:Real},
    x::AbstractArray{<:Real},
    pdims::NNlib.PoolDims;
    kwargs...,
) where {Tag,T1,P}
    value_fn(z) = ForwardDiff.value(Tag, z)
    partial_fn(z, i) = ForwardDiff.partials(Tag, z, i)

    dx = NNlib.∇meanpool(value_fn.(dy), y, x, pdims; kwargs...)
    dysᵢ = ntuple(i -> NNlib.∇meanpool(partial_fn.(dy, i), y, x, pdims; kwargs...), P)

    partials = ForwardDiff.Partials.(tuple.(dysᵢ...))
    return ForwardDiff.Dual{Tag,eltype(dx),P}.(dx, partials)
end

function NNlib.∇meanpool(
    dy::AbstractArray{<:ForwardDiff.Dual{Tag1,T1,P1}},
    y::AbstractArray{<:Real},
    x::AbstractArray{<:ForwardDiff.Dual{Tag2,T2,P2}},
    pdims::NNlib.PoolDims;
    kwargs...,
) where {Tag1,Tag2,T1,T2,P1,P2}
    value_fn(z) = ForwardDiff.value(Tag1, z)
    partial_fn(z, i) = ForwardDiff.partials(Tag1, z, i)

    dy_data, x_data = value_fn.(dy), ForwardDiff.value.(x)

    dx = NNlib.∇meanpool(dy_data, y, x_data, pdims; kwargs...)
    dysᵢ = ntuple(
        i -> NNlib.∇meanpool(partial_fn.(dy, i), y, x_data, pdims; kwargs...), P1
    )

    partials = ForwardDiff.Partials.(tuple.(dysᵢ...))
    return ForwardDiff.Dual{Tag1,eltype(dx),P1}.(dx, partials)
end

function NNlib.∇meanpool(
    dy::AbstractArray{<:ForwardDiff.Dual{Tag1,T1,P1}},
    y::AbstractArray{<:ForwardDiff.Dual{Tag2,T2,P2}},
    x::AbstractArray{<:Real},
    pdims::NNlib.PoolDims;
    kwargs...,
) where {Tag1,Tag2,T1,T2,P1,P2}
    value_fn(z) = ForwardDiff.value(Tag1, z)
    partial_fn(z, i) = ForwardDiff.partials(Tag1, z, i)

    dy_data, y_data = value_fn.(dy), ForwardDiff.value.(y)

    dx = NNlib.∇meanpool(dy_data, y_data, x, pdims; kwargs...)
    dysᵢ = ntuple(
        i -> NNlib.∇meanpool(partial_fn.(dy, i), y_data, x, pdims; kwargs...), P1
    )

    partials = ForwardDiff.Partials.(tuple.(dysᵢ...))
    return ForwardDiff.Dual{Tag1,eltype(dx),P1}.(dx, partials)
end

function NNlib.∇meanpool(
    dy::AbstractArray{<:Real},
    y::AbstractArray{<:Real},
    x::AbstractArray{<:ForwardDiff.Dual{Tag,T,P}},
    pdims::NNlib.PoolDims;
    kwargs...,
) where {Tag,T,P}
    value_fn(x) = ForwardDiff.value(Tag, x)

    dx = NNlib.∇meanpool(dy, y, value_fn.(x), pdims; kwargs...)
    return ForwardDiff.Dual{Tag,eltype(dx),P}.(dx)
end

function NNlib.∇meanpool(
    dy::AbstractArray{<:Real},
    y::AbstractArray{<:ForwardDiff.Dual{Tag,T,P}},
    x::AbstractArray{<:Real},
    pdims::NNlib.PoolDims;
    kwargs...,
) where {Tag,T,P}
    value_fn(x) = ForwardDiff.value(Tag, x)

    dx = NNlib.∇meanpool(dy, value_fn.(y), x, pdims; kwargs...)
    return ForwardDiff.Dual{Tag,eltype(dx),P}.(dx)
end

# Gather for Dual arrays
function NNlib.gather(
    src::AbstractArray{<:ForwardDiff.Dual{Tag,T,P}}, inds::AbstractArray
) where {Tag,T,P}
    value_fn(x) = ForwardDiff.value(Tag, x)
    partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

    inds_data = ForwardDiff.value.(inds)

    dst_data = NNlib.gather(value_fn.(src), inds_data)
    dst_partials = ntuple(i -> NNlib.gather(partial_fn.(src, i), inds_data), P)

    partials = ForwardDiff.Partials.(tuple.(dst_partials...))
    return ForwardDiff.Dual{Tag,eltype(dst_data),P}.(dst_data, partials)
end

# Scatter for Dual arrays
function NNlib.scatter(
    op::OP, src::AbstractArray{<:ForwardDiff.Dual{Tag,T,P}}, idx::AbstractArray; kwargs...
) where {OP,Tag,T,P}
    value_fn(x) = ForwardDiff.value(Tag, x)
    partial_fn(x, i) = ForwardDiff.partials(Tag, x, i)

    idx_data = ForwardDiff.value.(idx)

    dst_data = NNlib.scatter(op, value_fn.(src), idx_data; kwargs...)
    dst_partials = ntuple(
        i -> NNlib.scatter(op, partial_fn.(src, i), idx_data; kwargs...), P
    )

    partials = ForwardDiff.Partials.(tuple.(dst_partials...))
    return ForwardDiff.Dual{Tag,eltype(dst_data),P}.(dst_data, partials)
end

end
