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

matmuladd!(C, A, B, ::Nothing) = matmul!(C, A, B)
function matmuladd!(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    matmuladd!(C, internal_operation_mode((A, B, bias)), A, B, bias)
    return
end
function matmuladd!(C::AbstractMatrix, ::AbstractInternalArrayOpMode,
        A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    C .= bias
    mul!(C, A, B, true, true)
    return
end
function matmuladd!(C::AbstractMatrix, ::LoopedArrayOp, A::AbstractMatrix,
        B::AbstractMatrix, bias::AbstractVector)
    dims = (size(C, 1), size(A, 2), size(B, 2))
    if unrolled_any(≤(2048), dims) &&
       unrolled_all(≤(10_000), dims) &&
       LoopVectorization.check_args(C, A, B)
        __matmuladd_octavian!(C, A, B, bias)
        return
    end
    __matmuladd_generic!(C, A, B, bias)
    return
end

function __matmuladd_octavian!(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    # NOTE: Octavian doesn't do size checks.
    #       See https://github.com/JuliaLinearAlgebra/Octavian.jl/issues/109
    if size(A, 2) != size(B, 1)
        throw(DimensionMismatch(lazy"A has shape ($(size(A, 1)), $(size(A, 2))) but B has shape ($(size(B, 1)), $(size(B, 2)))"))
    end

    if length(bias) != size(A, 1)
        throw(DimensionMismatch(lazy"bias has length $(length(bias)) but A has shape ($(size(A, 1)), $(size(A, 2)))"))
    end

    Octavian.matmul!(C, A, B)
    __bias_add_impl!(C, internal_operation_mode((C, bias)), C, bias)
    return
end

function __matmuladd_generic!(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    C .= bias
    mul!(C, A, B, true, true)
    return
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

function matmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    matmul!(C, internal_operation_mode((A, B)), A, B)
    return
end
function matmul!(C::AbstractMatrix, ::AbstractInternalArrayOpMode,
        A::AbstractMatrix, B::AbstractMatrix)
    mul!(C, A, B)
    return
end
function matmul!(C::AbstractMatrix, ::LoopedArrayOp, A::AbstractMatrix, B::AbstractMatrix)
    dims = (size(C, 1), size(A, 2), size(B, 2))
    if unrolled_any(≤(2048), dims) &&
       unrolled_all(≤(10_000), dims) &&
       LoopVectorization.check_args(C, A, B)
        __matmul_octavian!(C, A, B)
        return
    end
    __matmul_generic!(C, A, B)
    return
end

function __matmul_octavian!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    # NOTE: Octavian doesn't do size checks.
    #       See https://github.com/JuliaLinearAlgebra/Octavian.jl/issues/109
    if size(A, 2) != size(B, 1)
        throw(DimensionMismatch(lazy"A has shape ($(size(A, 1)), $(size(A, 2))) but B has shape ($(size(B, 1)), $(size(B, 2)))"))
    end
    Octavian.matmul!(C, A, B)
    return
end

function __matmul_generic!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    mul!(C, A, B)
    return
end

# ChainRules
## `matmul`
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

## `matmuladd`
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

# EnzymeRules
@enzyme_reverse_alternative __matmul_octavian! __matmul_generic!

@enzyme_reverse_alternative __matmuladd_octavian! __matmuladd_generic!
