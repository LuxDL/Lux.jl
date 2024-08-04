function __batched_matmul_impl(
        ::False, ::Type, A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3})
    return batched_mul(A, B)  # Simple fallback to NNlib version
end

function __batched_matmul_impl(::True, ::Type{AbstractGPUDevice},
        A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3})
    return batched_mul(A, B)  # GPU versions are well optimized
end

function __batched_matmul_impl(
        ::True, ::Type{<:AMDGPUDevice}, A::AbstractArray{<:Complex, 3},
        B::AbstractArray{<:Complex, 3})
    @warn "Using fallback implementation of `batched_matmul` for complex numbers on \
           AMDGPUDevice" maxlog=1
    @assert size(A, 3) == size(B, 3) || size(A, 3) == 1 || size(B, 3) == 1
    size(A, 3) == size(B, 3) && return stack(*, eachslice(A; dims=3), eachslice(B; dims=3))
    size(A, 2) == 1 && stack(map(Base.Fix1(*, view(A, :, :, 1)), eachslice(B; dims=3)))
    return stack(map(Base.Fix2(*, view(B, :, :, 1)), eachslice(A; dims=3)))
end

function __batched_matmul_impl(
        ::True, ::Type{CPUDevice}, A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3})
    @assert size(A, 3) == size(B, 3) || size(A, 3) == 1 || size(B, 3) == 1
    C = similar(A, size(A, 1), size(B, 2), max(size(A, 3), size(B, 3)))
    __batched_matmul_impl!(C, internal_operation_mode((C, A, B)), A, B)
    return C
end

function __batched_matmul_impl!(C::AbstractArray{<:Any, 3}, ::AbstractInternalArrayOpMode,
        A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3})
    batched_mul!(C, A, B)
    return
end

function __batched_matmul_impl!(C::AbstractArray{<:Any, 3}, ::LoopedArrayOp,
        A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3})
    __batched_matmul_loopvec_impl!(C, A, B)
    return
end

function __batched_matmul_loopvec_impl!(
        C::AbstractArray{<:Any, 3}, A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3})
    if size(A, 3) == size(B, 3)
        @batch for L in indices((C, A, B), 3)
            __serial_loopvec_matmul!(batchview(C, L), batchview(A, L), batchview(B, L))
        end
    elseif size(A, 3) == 1
        @batch for L in indices((C, B), 3)
            __serial_loopvec_matmul!(batchview(C, L), batchview(A, 1), batchview(B, L))
        end
    else # has to be size(B, 3) == 1
        @batch for L in indices((C, A), 3)
            __serial_loopvec_matmul!(batchview(C, L), batchview(A, L), batchview(B, 1))
        end
    end
end

function __serial_loopvec_matmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    if !LoopVectorization.check_args(C, A, B)
        Octavian.matmul_serial!(C, A, B)
        return
    end
    @turbo for K in indices((C, B), 2), J in indices((C, A), 1)
        Cⱼₖ = zero(eltype(C))
        for I in indices((A, B), (2, 1))
            Cⱼₖ += A[I, J] * B[I, K]
        end
        C[J, K] = Cⱼₖ
    end
end

function CRC.rrule(
        ::typeof(batched_matmul), A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3})
    function batched_mul_pullback(_Δ)
        Δ = CRC.unthunk(_Δ)
        ∂A = CRC.@thunk begin
            tmp = batched_matmul(Δ, batched_adjoint(B))
            size(A, 3) == 1 ? sum(tmp; dims=3) : tmp
        end
        ∂B = CRC.@thunk begin
            tmp = batched_matmul(batched_adjoint(A), Δ)
            size(B, 3) == 1 ? sum(tmp; dims=3) : tmp
        end
        return ∂∅, ∂A, ∂B
    end
    return batched_matmul(A, B), ∇batched_matmul
end
