# Wrappers over Base & LinearAlgen implementations to use poly algs if needed
matmuladd(A, B, ::Nothing) = matmul(A, B)
function matmuladd(A::AbstractMatrix, B::AbstractVector, bias::AbstractVector)
    return vec(matmuladd(A, reshape(B, :, 1), bias))
end
function matmuladd(A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    return matmuladd(internal_operation_mode((A, B, bias)), A, B, bias)
end

function matmuladd(::AbstractInternalArrayOpMode, A::AbstractMatrix,
        B::AbstractMatrix, bias::AbstractVector)
    return muladd(A, B, bias)
end
function matmuladd(
        opmode::LoopedArrayOp, A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    C = similar(A, promote_type(eltype(A), eltype(B)), size(A, 1), size(B, 2))
    matmuladd!(C, opmode, A, B, bias)
    return C
end

function CRC.rrule(::typeof(matmuladd), opmode::LoopedArrayOp,
        A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    proj_A = CRC.ProjectTo(A)
    proj_B = CRC.ProjectTo(B)
    proj_bias = CRC.ProjectTo(bias)
    ∇matmuladd = @closure Δ -> begin
        Δ_ = CRC.unthunk(Δ)
        ∂A = CRC.@thunk(proj_A(matmul(opmode, Δ_, B')))
        ∂B = CRC.@thunk(proj_B(matmul(opmode, A', Δ_)))
        ∂bias = CRC.@thunk(proj_bias(__added_bias_gradient(bias, Δ_)))
        return ∂∅, ∂∅, ∂A, ∂B, ∂bias
    end
    return matmuladd(opmode, A, B, bias), ∇matmuladd
end

matmuladd!(C, A, B, ::Nothing) = matmul!(C, A, B)
function matmuladd!(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    matmuladd!(C, internal_operation_mode((A, B, bias)), A, B, bias)
    return nothing
end
function matmuladd!(C::AbstractMatrix, ::AbstractInternalArrayOpMode,
        A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    C .= bias
    mul!(C, A, B, true, true)
    return nothing
end
function matmuladd!(C::AbstractMatrix, ::LoopedArrayOp, A::AbstractMatrix,
        B::AbstractMatrix, bias::AbstractVector)
    if unrolled_all(≤(256), (size(C, 1), size(A, 2), size(B, 2)))
        @tturbo for n in indices((C, B), 2), m in indices((C, A), 1)
            Cmn = zero(eltype(C))
            for k in indices((A, B), (2, 1))
                Cmn += A[m, k] * B[k, n]
            end
            C[m, n] = Cmn + bias[m]
        end
        return nothing
    end
    C .= bias
    mul!(C, A, B, true, true)
    return nothing
end

function matmul(A::AbstractMatrix, B::AbstractVector)
    return vec(matmul(A, reshape(B, :, 1)))
end
function matmul(A::AbstractMatrix, B::AbstractMatrix)
    return matmul(internal_operation_mode((A, B)), A, B)
end

matmul(::AbstractInternalArrayOpMode, A::AbstractMatrix, B::AbstractMatrix) = A * B
function matmul(opmode::LoopedArrayOp, A::AbstractMatrix, B::AbstractMatrix)
    C = similar(A, promote_type(eltype(A), eltype(B)), size(A, 1), size(B, 2))
    matmul!(C, opmode, A, B)
    return C
end

function CRC.rrule(
        ::typeof(matmul), opmode::LoopedArrayOp, A::AbstractMatrix, B::AbstractMatrix)
    proj_A = CRC.ProjectTo(A)
    proj_B = CRC.ProjectTo(B)
    ∇matmul = @closure Δ -> begin
        Δ_ = CRC.unthunk(Δ)
        ∂A = CRC.@thunk(proj_A(matmul(opmode, Δ_, B')))
        ∂B = CRC.@thunk(proj_B(matmul(opmode, A', Δ_)))
        return ∂∅, ∂∅, ∂A, ∂B
    end
    return matmul(opmode, A, B), ∇matmul
end

function matmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    matmul!(C, internal_operation_mode((A, B)), A, B)
    return nothing
end
function matmul!(C::AbstractMatrix, ::AbstractInternalArrayOpMode,
        A::AbstractMatrix, B::AbstractMatrix)
    mul!(C, A, B)
    return nothing
end
function matmul!(C::AbstractMatrix, ::LoopedArrayOp, A::AbstractMatrix, B::AbstractMatrix)
    if unrolled_all(≤(256), (size(C, 1), size(A, 2), size(B, 2)))
        @tturbo for n in indices((C, B), 2), m in indices((C, A), 1)
            Cmn = zero(eltype(C))
            for k in indices((A, B), (2, 1))
                Cmn += A[m, k] * B[k, n]
            end
            C[m, n] = Cmn
        end
        return nothing
    end
    mul!(C, A, B)
    return nothing
end

# TODO: `matmul!` and `matmuladd!` need EnzymeRules
