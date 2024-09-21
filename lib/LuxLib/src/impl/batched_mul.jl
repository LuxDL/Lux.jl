# Entry Point
function batched_matmul(x::AbstractArray{xT, 3}, y::AbstractArray{yT, 3}) where {xT, yT}
    return batched_matmul(internal_operation_mode((x, y)), x, y)
end

function batched_matmul(::GenericBroadcastOp, x::AbstractArray{xT, 3},
        y::AbstractArray{yT, 3}) where {xT, yT}
    return NNlib.batched_mul(x, y)
end

for dev in (AMDGPUDevice, CUDADevice)
    @eval function batched_matmul(::GPUBroadcastOp{$(dev)},
            x::AbstractArray{xT, 3}, y::AbstractArray{yT, 3}) where {xT, yT}
        return NNlib.batched_mul(x, y)  # GPU versions are well optimized
    end
end

function batched_matmul(opmode::GPUBroadcastOp{<:AbstractGPUDevice},
        x::AbstractArray{xT, 3}, y::AbstractArray{yT, 3}) where {xT, yT}
    if isconcretetype(Core.Compiler._return_type(
        NNlib.batched_mul, Tuple{typeof(x), typeof(y)}))
        return NNlib.batched_mul(x, y)  # GPU versions are well optimized
    end
    return fallback_batched_matmul(opmode, x, y)
end

function batched_matmul(
        opmode::GPUBroadcastOp{AMDGPUDevice}, x::AbstractArray{<:Complex, 3},
        y::AbstractArray{<:Complex, 3})
    return fallback_batched_matmul(opmode, x, y)
end

function batched_matmul(opmode::LoopedArrayOp, x::AbstractArray{xT, 3},
        y::AbstractArray{yT, 3}) where {xT, yT}
    if (size(x, 3) != size(y, 3) && size(x, 3) != 1 && size(y, 3) != 1) ||
       (size(x, 2) != size(y, 1))
        throw(DimensionMismatch(lazy"size(x) = $(size(x)), size(y) = $(size(y)) inconsistent for batched_matmul."))
    end
    z = similar(x, promote_type(eltype(x), eltype(y)), size(x, 1),
        size(y, 2), max(size(x, 3), size(y, 3)))
    batched_matmul!(z, opmode, x, y)
    return z
end

function batched_matmul!(z::AbstractArray{zT, 3}, ::AbstractInternalArrayOpMode,
        x::AbstractArray{xT, 3}, y::AbstractArray{yT, 3}) where {zT, xT, yT}
    batched_mul!(z, x, y)
    return
end

function batched_matmul!(z::AbstractArray{zT, 3}, ::LoopedArrayOp,
        x::AbstractArray{xT, 3}, y::AbstractArray{yT, 3}) where {zT, xT, yT}
    if !LV.check_args(batchview(z, 1), batchview(x, 1), batchview(y, 1)) ||
       unsafe_known(explicit_blas_loaded())
        NNlib.batched_mul!(z, x, y)
        return
    end
    batched_matmul_loopvec_impl!(z, x, y)
    return
end

function batched_matmul_loopvec_impl!(
        z::AbstractArray{zT, 3}, x::AbstractArray{xT, 3},
        y::AbstractArray{yT, 3}, α::Number=true, β::Number=false) where {zT, xT, yT}
    if size(x, 3) == size(y, 3)
        @batch for L in indices((z, x, y), 3)
            serial_matmul_loopvec!(batchview(z, L), batchview(x, L), batchview(y, L), α, β)
        end
    elseif size(x, 3) == 1
        @batch for L in indices((z, y), 3)
            serial_matmul_loopvec!(batchview(z, L), batchview(x, 1), batchview(y, L), α, β)
        end
    else # has to be size(y, 3) == 1
        @batch for L in indices((z, x), 3)
            serial_matmul_loopvec!(batchview(z, L), batchview(x, L), batchview(y, 1), α, β)
        end
    end
end

function fallback_batched_matmul(
        dev, x::AbstractArray{xT, 3}, y::AbstractArray{yT, 3}) where {xT, yT}
    z = similar(x, promote_type(eltype(x), eltype(y)), size(x, 1),
        size(y, 2), max(size(x, 3), size(y, 3)))
    fallback_batched_matmul!(z, dev, x, y)
    return z
end

function fallback_batched_matmul!(
        z::AbstractArray{zT, 3}, dev, x::AbstractArray{xT, 3},
        y::AbstractArray{yT, 3}) where {zT, xT, yT}
    @warn "Using fallback Batched Matrix Multiply routine for $(dev) with A: size = \
           $(size(x)) eltype = $(xT) and B: size = $(size(y)) eltype = $(yT). This may be \
           slow." maxlog=1
    if (size(x, 3) != size(y, 3) && size(x, 3) != 1 && size(y, 3) != 1) ||
       (size(x, 2) != size(y, 1))
        throw(DimensionMismatch(lazy"size(x) = $(size(x)), size(y) = $(size(y)) inconsistent for batched_matmul."))
    end
    if size(x, 3) == size(y, 3)
        Threads.@threads for L in indices((x, y), 3)
            mul!(batchview(z, L), batchview(x, L), batchview(y, L))
        end
    elseif size(x, 3) == 1
        Threads.@threads for L in indices((x, y), 3)
            mul!(batchview(z, L), batchview(x, 1), batchview(y, L))
        end
    else # has to be size(y, 3) == 1
        Threads.@threads for L in indices((x, y), 3)
            mul!(batchview(z, L), batchview(x, L), batchview(y, 1))
        end
    end
end

function CRC.rrule(::typeof(batched_matmul), x::AbstractArray{xT, 3},
        y::AbstractArray{yT, 3}) where {xT, yT}
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
                cfg::EnzymeRules.RevConfigWidth, ::EnzymeCore.Const{typeof($(func))},
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
                cfg::EnzymeRules.RevConfigWidth, ::EnzymeCore.Const{typeof($(func))},
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
            dAs = A isa EnzymeCore.Const ? dCs : A.dval
            dBs = B isa EnzymeCore.Const ? dCs : B.dval

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
                            dA′ = batchview(dA, 1)
                            for L in indices(B′, 3)
                                mul!(dA′, batchview(dC, L),
                                    batchview(B′, L), true, true)
                            end
                        else
                            $(func)(dA, dC, NNlib.batched_adjoint(B.val), true, true)
                        end
                    end

                    if !(typeof(B) <: EnzymeCore.Const) && dB !== B.val
                        if size(dB, 3) == 1 && size(A.val, 3) != 1
                            A′ = NNlib.batched_adjoint(A.val)
                            dB′ = batchview(dB, 1)
                            for L in indices(A′, 3)
                                mul!(dB′, batchview(A′, L),
                                    batchview(dC, L), true, true)
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
