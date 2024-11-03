module LuxLibLoopVectorizationExt

using LoopVectorization: LoopVectorization, @tturbo, @turbo, indices
using Polyester: @batch
using Static: True

using LuxLib: LuxLib, Utils

Utils.is_extension_loaded(::Val{:LoopVectorization}) = True()

Utils.can_loopvec_args_check(::True, args...) = LoopVectorization.check_args(args...)

# matmul
for serial in (true, false)
    opname = serial ? :serial_matmul_loopvec! : :matmul_loopvec!
    @eval @inline function LuxLib.Impl.$(opname)(
            C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α::Number, β::Number)
        if !iszero(β) # Secial case this because Base.FastMath.mul_fast(NaN, false) = NaN
            @turbo thread=$(!serial) for K in indices((C, B), 2), J in indices((C, A), 1)
                Cⱼₖ = zero(eltype(C))
                for I in indices((A, B), (2, 1))
                    Cⱼₖ += A[J, I] * B[I, K]
                end
                C[J, K] = α * Cⱼₖ + β * C[J, K]
            end
        else
            @turbo thread=$(!serial) for K in indices((C, B), 2), J in indices((C, A), 1)
                Cⱼₖ = zero(eltype(C))
                for I in indices((A, B), (2, 1))
                    Cⱼₖ += A[J, I] * B[I, K]
                end
                C[J, K] = α * Cⱼₖ
            end
        end
    end
end

@inline function LuxLib.Impl.matmuladd_loopvec!(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    @tturbo for K in indices((C, B), 2), J in indices((C, A), 1)
        Cⱼₖ = zero(eltype(C))
        for I in indices((A, B), (2, 1))
            Cⱼₖ += A[J, I] * B[I, K]
        end
        C[J, K] = bias[J] + Cⱼₖ
    end
    return
end

# batched matmul
function LuxLib.Impl.batched_matmul_loopvec_impl!(
        z::AbstractArray{zT, 3}, x::AbstractArray{xT, 3},
        y::AbstractArray{yT, 3}, α::Number=true, β::Number=false) where {zT, xT, yT}
    if size(x, 3) == size(y, 3)
        @batch for L in axes(z, 3)
            LuxLib.Impl.serial_matmul_loopvec!(
                Utils.batchview(z, L), Utils.batchview(x, L), Utils.batchview(y, L), α, β)
        end
    elseif size(x, 3) == 1
        @batch for L in axes(z, 3)
            LuxLib.Impl.serial_matmul_loopvec!(
                Utils.batchview(z, L), Utils.batchview(x, 1), Utils.batchview(y, L), α, β)
        end
    else # has to be size(y, 3) == 1
        @batch for L in axes(z, 3)
            LuxLib.Impl.serial_matmul_loopvec!(
                Utils.batchview(z, L), Utils.batchview(x, L), Utils.batchview(y, 1), α, β)
        end
    end
end

end
