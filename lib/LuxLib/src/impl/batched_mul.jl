# Entry Point
function batched_matmul(x::AbstractArray{<:Number, 3}, y::AbstractArray{<:Number, 3})
    return batched_matmul(internal_operation_mode((x, y)), x, y)
end

function batched_matmul(
        ::GenericBroadcastOp, x::AbstractArray{<:Number, 3}, y::AbstractArray{<:Number, 3})
    return NNlib.batched_mul(x, y)
end

function batched_matmul(::GPUBroadcastOp{<:AbstractGPUDevice},
        x::AbstractArray{<:Number, 3}, y::AbstractArray{<:Number, 3})
    return NNlib.batched_mul(x, y)  # GPU versions are well optimized
end

function batched_matmul(::GPUBroadcastOp{AMDGPUDevice}, x::AbstractArray{<:Complex, 3},
        y::AbstractArray{<:Complex, 3})
    if (size(x, 3) != size(y, 3) && size(x, 3) != 1 && size(y, 3) != 1) ||
       (size(x, 2) != size(y, 1))
        throw(DimensionMismatch(lazy"size(x) = $(size(x)), size(y) = $(size(y)) inconsistent for batched_matmul."))
    end
    @warn "Using fallback implementation of `batched_matmul` for complex numbers on \
           AMDGPUDevice" maxlog=1
    size(x, 3) == size(y, 3) && return stack(*, Utils.batchview(x), Utils.batchview(y))
    size(x, 3) == 1 && return stack(Base.Fix1(*, Utils.batchview(x, 1)), Utils.batchview(y))
    return stack(Base.Fix2(*, Utils.batchview(y, 1)), Utils.batchview(x))
end

function batched_matmul(
        opmode::LoopedArrayOp, x::AbstractArray{<:Number, 3}, y::AbstractArray{<:Number, 3})
    if (size(x, 3) != size(y, 3) && size(x, 3) != 1 && size(y, 3) != 1) ||
       (size(x, 2) != size(y, 1))
        throw(DimensionMismatch(lazy"size(x) = $(size(x)), size(y) = $(size(y)) inconsistent for batched_matmul."))
    end
    z = similar(x, promote_type(eltype(x), eltype(y)), size(x, 1),
        size(y, 2), max(size(x, 3), size(y, 3)))
    batched_matmul!(z, opmode, x, y)
    return z
end

function batched_matmul!(z::AbstractArray{<:Number, 3}, ::AbstractInternalArrayOpMode,
        x::AbstractArray{<:Number, 3}, y::AbstractArray{<:Number, 3})
    batched_mul!(z, x, y)
    return
end

function batched_matmul!(z::AbstractArray{<:Number, 3}, ::LoopedArrayOp,
        x::AbstractArray{<:Number, 3}, y::AbstractArray{<:Number, 3})
    if !LV.check_args(
        Utils.batchview(z, 1), Utils.batchview(x, 1), Utils.batchview(y, 1)) ||
       Utils.known(System.explicit_blas_loaded())
        NNlib.batched_mul!(z, x, y)
        return
    end
    batched_matmul_loopvec_impl!(z, x, y)
    return
end

function batched_matmul_loopvec_impl!(
        z::AbstractArray{<:Number, 3}, x::AbstractArray{<:Number, 3},
        y::AbstractArray{<:Number, 3}, α::Number=true, β::Number=false)
    if size(x, 3) == size(y, 3)
        @batch for L in indices((z, x, y), 3)
            serial_matmul_loopvec!(
                Utils.batchview(z, L), Utils.batchview(x, L), Utils.batchview(y, L), α, β)
        end
    elseif size(x, 3) == 1
        @batch for L in indices((z, y), 3)
            serial_matmul_loopvec!(
                Utils.batchview(z, L), Utils.batchview(x, 1), Utils.batchview(y, L), α, β)
        end
    else # has to be size(y, 3) == 1
        @batch for L in indices((z, x), 3)
            serial_matmul_loopvec!(
                Utils.batchview(z, L), Utils.batchview(x, L), Utils.batchview(y, 1), α, β)
        end
    end
end

function CRC.rrule(::typeof(batched_matmul), x::AbstractArray{<:Number, 3},
        y::AbstractArray{<:Number, 3})
    ∇batched_matmul = @closure Δ_ -> begin
        Δ = CRC.unthunk(Δ_)
        ∂x = CRC.@thunk begin
            tmp = batched_matmul(Δ, NNlib.batched_adjoint(y))
            size(x, 3) == 1 ? sum(tmp; dims=3) : tmp
        end
        ∂y = CRC.@thunk begin
            tmp = batched_matmul(NNlib.batched_adjoint(x), Δ)
            size(y, 3) == 1 ? sum(tmp; dims=3) : tmp
        end
        return ∂∅, ∂x, ∂y
    end
    return batched_matmul(x, y), ∇batched_matmul
end

# This is type-piracy but needed to fix a blocking issue. TODO: upstream to NNlib
# Enzyme causes a "active variables passed by value to jl_new_task are not yet supported"
# warning without this patch.
for func in (NNlib.batched_mul!, batched_matmul_loopvec_impl!)
    @eval begin
        function EnzymeRules.augmented_primal(
                cfg::EnzymeRules.ConfigWidth, ::EnzymeCore.Const{typeof($(func))},
                ::Type{RT}, C::EnzymeCore.Annotation{<:AbstractArray{<:Any, 3}},
                A::EnzymeCore.Annotation{<:AbstractArray{<:Any, 3}},
                B::EnzymeCore.Annotation{<:AbstractArray{<:Any, 3}}) where {RT}
            if typeof(C) <: EnzymeCore.Duplicated || typeof(C) <: EnzymeCore.BatchDuplicated
                $(func)(C.val, A.val, B.val)
            end

            primal = EnzymeRules.needs_primal(cfg) ? C.val : nothing
            shadow = EnzymeRules.needs_shadow(cfg) ? C.dval : nothing

            cache_A = (EnzymeRules.overwritten(cfg)[3] &&
                       !(typeof(C) <: EnzymeCore.Const) &&
                       !(typeof(B) <: EnzymeCore.Const)) ? copy(A.val) : nothing
            cache_B = (EnzymeRules.overwritten(cfg)[3] &&
                       !(typeof(C) <: EnzymeCore.Const) &&
                       !(typeof(A) <: EnzymeCore.Const)) ? copy(B.val) : nothing

            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_B))
        end

        function EnzymeRules.reverse(
                cfg::EnzymeRules.ConfigWidth, ::EnzymeCore.Const{typeof($(func))},
                ::Type{RT}, cache, C::EnzymeCore.Annotation{<:AbstractArray{<:Any, 3}},
                A::EnzymeCore.Annotation{<:AbstractArray{<:Any, 3}},
                B::EnzymeCore.Annotation{<:AbstractArray{<:Any, 3}}) where {RT}
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

            # NOTE: The implementation here is memory efficient and non-allocating. However,
            #       for maximum performance we would want to reuse the parallel batched_mul
            #       followed by a reduction.
            for (dC, dA, dB) in zip(dCs, dAs, dBs)
                if !(typeof(C) <: EnzymeCore.Const) && dC !== C.val
                    if !(typeof(A) <: EnzymeCore.Const) && dA !== A.val
                        if size(dA, 3) == 1 && size(B.val, 3) != 1
                            B′ = NNlib.batched_adjoint(B.val)
                            dA′ = Utils.batchview(dA, 1)
                            for L in indices(B′, 3)
                                mul!(dA′, Utils.batchview(dC, L),
                                    Utils.batchview(B′, L), true, true)
                            end
                        else
                            $(func)(dA, dC, NNlib.batched_adjoint(B.val), true, true)
                        end
                    end

                    if !(typeof(B) <: EnzymeCore.Const) && dB !== B.val
                        if size(dB, 3) == 1 && size(A.val, 3) != 1
                            A′ = NNlib.batched_adjoint(A.val)
                            dB′ = Utils.batchview(dB, 1)
                            for L in indices(A′, 3)
                                mul!(dB′, Utils.batchview(A′, L),
                                    Utils.batchview(dC, L), true, true)
                            end
                        else
                            $(func)(dB, NNlib.batched_adjoint(A.val), dC, true, true)
                        end
                    end

                    dC .= 0
                end
            end

            return ntuple(Returns(nothing), 3)
        end
    end
end
