# Wrappers over Base & LinearAlgebra implementations to use poly algs if needed
matmuladd(A, B, ::Nothing) = matmul(A, B)
function matmuladd(A::AbstractMatrix, B::AbstractVector, bias::AbstractVector)
    return matmuladd(A, get_utils(:insert_batch_dim)(B), bias)
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

function matmul(A::AbstractMatrix, B::AbstractVector)
    return vec(matmul(A, get_utils(:insert_batch_dim)(B)))
end
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
    mul!(C, A, B, true, true)
    return
end

function matmuladd!(C::AbstractMatrix, ::GPUBroadcastOp{CUDADevice},
        A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    cublasLt_fused_dense!(C, identity, A, B, bias)
    return
end

function matmuladd!(C::AbstractMatrix, ::LoopedArrayOp, A::AbstractMatrix,
        B::AbstractMatrix, bias::AbstractVector)
    if LV.check_args(C, A, B, bias) && System.fits_in_l2cache(C, A, B, bias)
        matmuladd_loopvec!(C, A, B, bias)
        return
    end
    matmuladd_cpu_fallback!(C, A, B, bias)
    return
end

function matmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    matmul!(C, internal_operation_mode((C, A, B)), A, B)
    return
end

function matmul!(C::AbstractMatrix, ::AbstractInternalArrayOpMode,
        A::AbstractMatrix, B::AbstractMatrix)
    mul!(C, A, B)
    return
end

function matmul!(C::AbstractMatrix, ::LoopedArrayOp, A::AbstractMatrix, B::AbstractMatrix)
    return matmul_cpu!(C, System.use_octavian(), System.explicit_blas_loaded(), A, B)
end

for spl_blas in (True, False)
    @eval begin
        function matmul_cpu!( # Octavian can be used
                C::AbstractMatrix, ::True, ::$(spl_blas),
                A::AbstractMatrix, B::AbstractMatrix)
            if LV.check_args(C, A, B)
                if System.fits_in_l1cache(C, A, B)
                    matmul_loopvec!(C, A, B, true, false)
                    return
                elseif $(Utils.known(spl_blas()) ? System.fits_in_l2cache :
                         System.fits_in_l3cache)(C, A, B)
                    matmul_octavian!(C, A, B, true, false)
                    return
                end
            end
            matmul_cpu_fallback!(C, A, B, true, false)
            return
        end

        function matmul_cpu!( # Octavian cannot be used
                C::AbstractMatrix, ::False, ::$(spl_blas),
                A::AbstractMatrix, B::AbstractMatrix)
            if LV.check_args(C, A, B)
                if $(Utils.known(spl_blas()) ? System.fits_in_l1cache :
                     System.fits_in_l2cache)(C, A, B)
                    matmul_loopvec!(C, A, B, true, false)
                    return
                end
            end
            matmul_cpu_fallback!(C, A, B, true, false)
            return
        end
    end
end

# Low-Level Matmul implementations -- Either call libraries or implement our own
# We force inlining here to avoid allocations in the inner loops
@inline function matmul_octavian!(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α::Number, β::Number)
    Octavian.matmul!(C, A, B, α, β)
    return
end

# Best case fallback, we are likely going to hit BLAS
@inline function matmul_cpu_fallback!(C::AbstractMatrix{T}, A::AbstractMatrix{T},
        B::AbstractMatrix{T}, α::Number, β::Number) where {T}
    matmul_linalg_default!(C, A, B, α, β)
    return
end

@inline function matmul_cpu_fallback!(C::AbstractMatrix{T}, A::AbstractMatrix{AT},
        B::AbstractMatrix{BT}, α::Number, β::Number) where {T, AT, BT}
    if LV.check_args(C, A, B)  # Use Octavian if possible. Don't check via `use_octavian()`
        matmul_octavian!(C, A, B, α, β)
        return
    end
    # Generic fallback is actually quite good starting julia 1.11
    @static if VERSION ≥ v"1.11-"
        @warn lazy"Mixed-Precision `matmul_cpu_fallback!` detected and Octavian.jl cannot be used for this set of inputs (C [$(typeof(C))]: A [$(typeof(A))] x B [$(typeof(B))]). Falling back to generic implementation. This may be slow." maxlog=1
        A′, B′ = A, B
    else
        @warn lazy"Mixed-Precision `matmul_cpu_fallback!` detected and Octavian.jl cannot be used for this set of inputs (C [$(typeof(C))]: A [$(typeof(A))] x B [$(typeof(B))]). Converting to common type to to attempt to use BLAS. This may be slow." maxlog=1
        A′, B′ = Utils.ofeltype_array(T, A), Utils.ofeltype_array(T, B)
    end
    matmul_linalg_default!(C, A′, B′, α, β)
    return
end

@inline function matmul_linalg_default!(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α::Number, β::Number)
    mul!(C, A, B, α, β)
    return
end

for serial in (true, false)
    opname = serial ? :serial_matmul_loopvec! : :matmul_loopvec!
    @eval @inline function $opname(
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

@inline function matmuladd_loopvec!(
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

@inline function matmuladd_cpu_fallback!(
        C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    C .= bias
    matmul_cpu_fallback!(C, A, B, true, true)
    return
end

# ChainRules
function CRC.rrule(::typeof(matmul), A::AbstractMatrix, B::AbstractMatrix)
    𝒫A, 𝒫B = CRC.ProjectTo(A), CRC.ProjectTo(B)
    ∇matmul = @closure Δ′ -> begin
        Δ = CRC.unthunk(Δ′)
        ∂A = CRC.@thunk(𝒫A(matmul(Δ, B')))
        ∂B = CRC.@thunk(𝒫B(matmul(A', Δ)))
        return ∂∅, ∂A, ∂B
    end
    return matmul(A, B), ∇matmul
end

function CRC.rrule(
        ::typeof(matmuladd), A::AbstractMatrix, B::AbstractMatrix, bias::AbstractVector)
    𝒫A, 𝒫B, 𝒫bias = CRC.ProjectTo(A), CRC.ProjectTo(B), CRC.ProjectTo(bias)
    ∇matmuladd = @closure Δ′ -> begin
        Δ = CRC.unthunk(Δ′)
        ∂A = CRC.@thunk(𝒫A(matmul(Δ, B')))
        ∂B = CRC.@thunk(𝒫B(matmul(A', Δ)))
        ∂bias = CRC.@thunk(𝒫bias(∇bias_add(bias, Δ)))
        return ∂∅, ∂A, ∂B, ∂bias
    end
    return matmuladd(A, B, bias), ∇matmuladd
end

# EnzymeRules
## ReverseMode
Utils.@enzyme_alternative matmul_octavian! matmul_linalg_default!
Utils.@enzyme_alternative serial_matmul_loopvec! matmul_linalg_default!
Utils.@enzyme_alternative matmul_loopvec! matmul_linalg_default!

Utils.@enzyme_alternative matmuladd_loopvec! matmuladd_cpu_fallback!

## ForwardMode
# NOTE: forward mode works fine with LoopVectorization but not with Octavian
function EnzymeRules.forward(
        ::EnzymeCore.Const{typeof(matmul_octavian!)}, ::Type{RT}, args...) where {RT}
    return EnzymeCore.autodiff(
        EnzymeCore.Forward, EnzymeCore.Const(matmul_linalg_default!), RT, args...)
end
