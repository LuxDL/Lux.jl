# Wrappers over Base & LinearAlgebra implementations to use poly algs if needed
matmuladd(A, B, ::Nothing) = matmul(A, B)
function matmuladd(A::AbstractMatrix, B::AbstractVector, bias::AbstractVector)
    return matmuladd(A, reshape(B, :, 1), bias)
end
function matmuladd(A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    return matmuladd(internal_operation_mode((A, B, bias)), A, B, bias)
end

function matmuladd(
        ::GenericBroadcastOp, A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    return muladd(A, B, bias)
end
function matmuladd(opmode::AbstractInternalArrayOpMode, A::AbstractMatrix,
        B::AbstractMatrix, bias::AbstractVector)
    if size(A, 2) != size(B, 1)
        throw(DimensionMismatch(lazy"A has shape ($(size(A, 1)), $(size(A, 2))) but B has shape ($(size(B, 1)), $(size(B, 2)))"))
    end
    if length(bias) != size(A, 1)
        throw(DimensionMismatch(lazy"bias has length $(length(bias)) but A has shape ($(size(A, 1)), $(size(A, 2)))"))
    end
    C = similar(A, promote_type(eltype(A), eltype(B), eltype(bias)), size(A, 1), size(B, 2))
    matmuladd!(C, opmode, A, B, bias)
    return C
end

matmul(A::AbstractMatrix, B::AbstractVector) = vec(matmul(A, reshape(B, :, 1)))
function matmul(A::AbstractMatrix, B::AbstractMatrix)
    if size(A, 2) != size(B, 1)
        throw(DimensionMismatch(lazy"A has shape ($(size(A, 1)), $(size(A, 2))) but B has shape ($(size(B, 1)), $(size(B, 2)))"))
    end
    return matmul(internal_operation_mode((A, B)), A, B)
end

matmul(::GenericBroadcastOp, A::AbstractMatrix, B::AbstractMatrix) = A * B
function matmul(::AbstractInternalArrayOpMode, A::AbstractMatrix, B::AbstractMatrix)
    C = similar(A, promote_type(eltype(A), eltype(B)), size(A, 1), size(B, 2))
    matmul!(C, A, B)
    return C
end

# Slightly higher level. Here we make decisions about which implementation to use
function matmuladd!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, ::Nothing)
    matmul!(C, A, B)
    return
end
function matmuladd!(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    matmuladd!(C, internal_operation_mode((C, A, B, bias)), A, B, bias)
    return
end

function matmuladd!(C::AbstractMatrix, ::AbstractInternalArrayOpMode,
        A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    C .= bias
    matmul_generic!(C, A, B, true, true)
    return
end

function matmuladd!(C::AbstractMatrix, ::GPUBroadcastOp{CUDADevice},
        A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    retcode = cublasLt_fused_dense!(C, identity, A, B, bias, False())
    retcode == -1 || return
    matmuladd!(C, GenericBroadcastOp(), A, B, bias)
    return
end

function matmuladd!(C::AbstractMatrix, opmode::LoopedArrayOp,
        A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    matmuladd!(C, opmode, System.use_octavian(), A, B, bias)
    return
end

function matmuladd!(C::AbstractMatrix, ::LoopedArrayOp, ::False,
        A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    if LV.check_args(C, A, B) &&
       Utils.unrolled_all(≤(256), (size(C, 1), size(A, 2), size(B, 2)))
        matmuladd_loopvec!(C, A, B, bias)
        return
    end
    matmuladd!(C, GenericBroadcastOp(), A, B, bias)
    return
end

function matmuladd!(C::AbstractMatrix, opmode::LoopedArrayOp, ::True,
        A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    if LV.check_args(C, A, B)
        if Utils.unrolled_all(≤(256), (size(C, 1), size(A, 2), size(B, 2)))
            matmuladd_loopvec!(C, A, B, bias)
            return
        elseif Utils.unrolled_any(≤(2048), size(C), size(A), size(B)) &&
               Utils.unrolled_all(≤(10_000), size(C), size(A), size(B))
            matmuladd_octavian!(C, A, B, true, false)
            bias_add!(C, opmode, C, bias)
            return
        end
    end
    matmuladd!(C, GenericBroadcastOp(), A, B, bias)
    return
end

function matmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    matmul!(C, internal_operation_mode((C, A, B)), A, B)
    return
end

function matmul!(C::AbstractMatrix, ::AbstractInternalArrayOpMode,
        A::AbstractMatrix, B::AbstractMatrix)
    matmul_generic!(C, A, B, true, false)
    return
end

function matmul!(
        C::AbstractMatrix, opmode::LoopedArrayOp, A::AbstractMatrix, B::AbstractMatrix)
    return matmul!(C, opmode, System.use_octavian(), A, B)
end

function matmul!(
        C::AbstractMatrix, ::LoopedArrayOp, ::True, A::AbstractMatrix, B::AbstractMatrix)
    dims = (size(C, 1), size(A, 2), size(B, 2))
    if LV.check_args(C, A, B)
        if Utils.unrolled_all(≤(16), dims)
            serial_matmul_loopvec!(C, A, B, true, false)
            return
        elseif Utils.unrolled_any(≤(2048), dims) && Utils.unrolled_all(≤(10_000), dims)
            matmul_octavian!(C, A, B, true, false)
            return
        end
    end
    matmul_generic!(C, A, B, true, false)
    return
end

function matmul!(
        C::AbstractMatrix, ::LoopedArrayOp, ::False, A::AbstractMatrix, B::AbstractMatrix)
    if LV.check_args(C, A, B) &&
       Utils.unrolled_all(≤(256), (size(C, 1), size(A, 2), size(B, 2)))
        matmul_loopvec!(C, A, B, true, false)
        return
    end
    matmul_generic!(C, A, B, true, false)
    return
end

# Low-Level Matmul implementations -- Either call libraries or implement our own
function matmul_octavian!(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α::Number, β::Number)
    Octavian.matmul!(C, A, B, α, β)
    return
end

function matmul_generic!(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α::Number, β::Number)
    mul!(C, A, B, α, β)
    return
end

for serial in (true, false)
    opname = serial ? :serial_matmul_loopvec! : :matmul_loopvec!
    @eval function $opname(
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

function matmuladd_loopvec!(
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

# ChainRules
function CRC.rrule(::typeof(matmul), A::AbstractMatrix, B::AbstractMatrix)
    𝒫A = CRC.ProjectTo(A)
    𝒫B = CRC.ProjectTo(B)
    ∇matmul = @closure Δ -> begin
        Δ_ = CRC.unthunk(Δ)
        ∂A = CRC.@thunk(𝒫A(matmul(Δ_, B')))
        ∂B = CRC.@thunk(𝒫B(matmul(A', Δ_)))
        return ∂∅, ∂A, ∂B
    end
    return matmul(A, B), ∇matmul
end

function CRC.rrule(
        ::typeof(matmuladd), A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    𝒫A = CRC.ProjectTo(A)
    𝒫B = CRC.ProjectTo(B)
    𝒫bias = CRC.ProjectTo(bias)
    ∇matmuladd = @closure Δ -> begin
        Δ_ = CRC.unthunk(Δ)
        ∂A = CRC.@thunk(𝒫A(matmul(Δ_, B')))
        ∂B = CRC.@thunk(𝒫B(matmul(A', Δ_)))
        ∂bias = CRC.@thunk(𝒫bias(∇bias_add(bias, Δ_)))
        return ∂∅, ∂A, ∂B, ∂bias
    end
    return matmuladd(A, B, bias), ∇matmuladd
end

# EnzymeRules
Utils.@enzyme_reverse_alternative matmul_octavian! matmul_generic!
Utils.@enzyme_reverse_alternative serial_matmul_loopvec! matmul_generic!
Utils.@enzyme_reverse_alternative matmul_loopvec! matmul_generic!
