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
## `matmul!`
function EnzymeRules.augmented_primal(
        cfg::EnzymeRules.ConfigWidth, func::EnzymeCore.Const{typeof(matmul!)},
        ::Type{RT}, C::EnzymeCore.Annotation{<:AbstractMatrix},
        opmode::EnzymeCore.Const{LoopedArrayOp},
        A::EnzymeCore.Annotation{<:AbstractMatrix},
        B::EnzymeCore.Annotation{<:AbstractMatrix}) where {RT}
    if typeof(C) <: EnzymeCore.Duplicated || typeof(C) <: EnzymeCore.BatchDuplicated
        func.val(C.val, A.val, B.val)
    end

    primal = EnzymeRules.needs_primal(cfg) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(cfg) ? C.dval : nothing

    cache_A = (EnzymeRules.overwritten(cfg)[3] &&
               !(typeof(C) <: EnzymeCore.Const) &&
               !(typeof(B) <: EnzymeCore.Const)) ? copy(A.val) : nothing
    cache_B = (EnzymeRules.overwritten(cfg)[4] &&
               !(typeof(C) <: EnzymeCore.Const) &&
               !(typeof(A) <: EnzymeCore.Const)) ? copy(B.val) : nothing

    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_B))
end

function EnzymeRules.reverse(
        cfg::EnzymeRules.ConfigWidth, func::EnzymeCore.Const{typeof(matmul!)},
        ::Type{RT}, cache, C::EnzymeCore.Annotation{<:AbstractMatrix},
        opmode::EnzymeCore.Const{LoopedArrayOp},
        A::EnzymeCore.Annotation{<:AbstractMatrix},
        B::EnzymeCore.Annotation{<:AbstractMatrix}) where {RT}
    cache_A, cache_B = cache

    if !(typeof(B) <: EnzymeCore.Const) && !(typeof(C) <: EnzymeCore.Const)
        if !EnzymeRules.overwritten(cfg)[3]
            cache_A = A.val
        end
    end

    if !(typeof(A) <: EnzymeCore.Const) && !(typeof(C) <: EnzymeCore.Const)
        if !EnzymeRules.overwritten(cfg)[3]
            cache_B = B.val
        end
    end

    dCs = C.dval
    dAs = (typeof(A) <: EnzymeCore.Const) ? dCs : A.dval
    dBs = (typeof(B) <: EnzymeCore.Const) ? dCs : B.dval

    if EnzymeRules.width(cfg) == 1
        dCs = (dCs,)
        dAs = (dAs,)
        dBs = (dBs,)
    end

    for (dC, dA, dB) in zip(dCs, dAs, dBs)
        if !(typeof(C) <: EnzymeCore.Const) && dC !== C.val
            if !(typeof(A) <: EnzymeCore.Const) && dA !== A.val
                func.val(dA, opmode.val, dC, B.val')
            end

            if !(typeof(B) <: EnzymeCore.Const) && dB !== B.val
                func.val(dB, opmode.val, A.val', dC)
            end

            dC .= 0
        end
    end

    return ntuple(Returns(nothing), 4)
end

## `matmuladd!`
function EnzymeRules.augmented_primal(
        cfg::EnzymeRules.ConfigWidth, func::EnzymeCore.Const{typeof(matmuladd!)},
        ::Type{RT}, C::EnzymeCore.Annotation{<:AbstractMatrix},
        opmode::EnzymeCore.Const{LoopedArrayOp},
        A::EnzymeCore.Annotation{<:AbstractMatrix},
        B::EnzymeCore.Annotation{<:AbstractMatrix},
        bias::EnzymeCore.Annotation{<:AbstractVector}) where {RT}
    if typeof(C) <: EnzymeCore.Duplicated || typeof(C) <: EnzymeCore.BatchDuplicated
        func.val(C.val, A.val, B.val, bias.val)
    end

    primal = EnzymeRules.needs_primal(cfg) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(cfg) ? C.dval : nothing

    cache_A = (EnzymeRules.overwritten(cfg)[3] &&
               !(typeof(C) <: EnzymeCore.Const) &&
               !(typeof(B) <: EnzymeCore.Const)) ? copy(A.val) : nothing
    cache_B = (EnzymeRules.overwritten(cfg)[4] &&
               !(typeof(C) <: EnzymeCore.Const) &&
               !(typeof(A) <: EnzymeCore.Const)) ? copy(B.val) : nothing
    cache_bias = (EnzymeRules.overwritten(cfg)[5] && !(typeof(C) <: EnzymeCore.Const)) ?
                 copy(bias.val) : nothing

    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_B, cache_bias))
end

function EnzymeRules.reverse(
        cfg::EnzymeRules.ConfigWidth, func::EnzymeCore.Const{typeof(matmuladd!)},
        ::Type{RT}, cache, C::EnzymeCore.Annotation{<:AbstractMatrix},
        opmode::EnzymeCore.Const{LoopedArrayOp},
        A::EnzymeCore.Annotation{<:AbstractMatrix},
        B::EnzymeCore.Annotation{<:AbstractMatrix},
        bias::EnzymeCore.Annotation{<:AbstractVector}) where {RT}
    cache_A, cache_B, cache_bias = cache

    if !(typeof(B) <: EnzymeCore.Const) && !(typeof(C) <: EnzymeCore.Const)
        if !EnzymeRules.overwritten(cfg)[3]
            cache_A = A.val
        end
    end

    if !(typeof(A) <: EnzymeCore.Const) && !(typeof(C) <: EnzymeCore.Const)
        if !EnzymeRules.overwritten(cfg)[4]
            cache_B = B.val
        end
    end

    if !(typeof(C) <: EnzymeCore.Const)
        if !EnzymeRules.overwritten(cfg)[5]
            cache_bias = bias.val
        end
    end

    dCs = C.dval
    dAs = (typeof(A) <: EnzymeCore.Const) ? dCs : A.dval
    dBs = (typeof(B) <: EnzymeCore.Const) ? dCs : B.dval
    dbiases = (typeof(bias) <: EnzymeCore.Const) ? dCs : bias.dval

    if EnzymeRules.width(cfg) == 1
        dCs = (dCs,)
        dAs = (dAs,)
        dBs = (dBs,)
        dbiases = (dbiases,)
    end

    for (dC, dA, dB, dbias) in zip(dCs, dAs, dBs, dbiases)
        if !(typeof(C) <: EnzymeCore.Const) && dC !== C.val
            if !(typeof(A) <: EnzymeCore.Const) && dA !== A.val
                matmul!(dA, opmode.val, dC, B.val')
            end

            if !(typeof(B) <: EnzymeCore.Const) && dB !== B.val
                matmul!(dB, opmode.val, A.val', dC)
            end

            if !(typeof(bias) <: EnzymeCore.Const) && dbias !== bias.val
                sum!(dbias, dC)
            end

            dC .= 0
        end
    end

    return ntuple(Returns(nothing), 5)
end
