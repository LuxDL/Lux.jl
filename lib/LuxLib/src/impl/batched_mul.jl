# Entry Point
function batched_matmul(
    x::AbstractArray{xT,N},
    y::AbstractArray{yT,N};
    lhs_contracting_dim::Int=2,
    rhs_contracting_dim::Int=1,
    lhs_batching_dims::Dims{M}=ntuple(Base.Fix2(+, 2), Val(N - 2)),
    rhs_batching_dims::Dims{M}=ntuple(Base.Fix2(+, 2), Val(N - 2)),
) where {xT,yT,N,M}
    assert_batched_matmul_checks(
        x, y; lhs_contracting_dim, rhs_contracting_dim, lhs_batching_dims, rhs_batching_dims
    )

    x_repeats, y_repeats, batch_size_tuple = get_batched_matmul_repeat_dims(
        x, y, lhs_batching_dims, rhs_batching_dims
    )
    # for 3D case, we can do a simple pass through and the implementation will handle it
    # without expanding the batching dims
    if length(lhs_batching_dims) != 1 && length(rhs_batching_dims) != 1
        x = Utils.maybe_repeat(x, x_repeats)
        y = Utils.maybe_repeat(y, y_repeats)
    end

    if (
        lhs_batching_dims != ntuple(Base.Fix2(+, 2), Val(N - 2)) ||
        rhs_batching_dims != ntuple(Base.Fix2(+, 2), Val(N - 2)) ||
        lhs_contracting_dim != 2 ||
        rhs_contracting_dim != 1
    )
        lhs_non_contracting_dims = get_non_contracting_dim(
            N, lhs_contracting_dim, lhs_batching_dims
        )
        rhs_non_contracting_dims = get_non_contracting_dim(
            N, rhs_contracting_dim, rhs_batching_dims
        )
        x_permuted = Utils.maybe_permutedims(
            x, (lhs_non_contracting_dims, lhs_contracting_dim, lhs_batching_dims...)
        )
        y_permuted = Utils.maybe_permutedims(
            y, (rhs_contracting_dim, rhs_non_contracting_dims, rhs_batching_dims...)
        )
    else
        x_permuted, y_permuted = x, y
    end

    x_flattened = Utils.maybe_reshape(
        x_permuted,
        (size(x_permuted, 1), size(x_permuted, 2), prod(size(x_permuted)[3:end])),
    )
    y_flattened = Utils.maybe_reshape(
        y_permuted,
        (size(y_permuted, 1), size(y_permuted, 2), prod(size(y_permuted)[3:end])),
    )

    res = batched_matmul_fallback(x_flattened, y_flattened)

    length(lhs_batching_dims) == 1 && return res
    return reshape(res, size(res, 1), size(res, 2), batch_size_tuple...)
end

function get_non_contracting_dim(
    N::Int, contracting_dim::Int, batching_dims::Dims{M}
) where {M}
    return only(setdiff(1:N, (contracting_dim, batching_dims...)))
end

CRC.@non_differentiable get_non_contracting_dim(::Any...)

function batched_matmul_fallback(
    x::AbstractArray{xT,3}, y::AbstractArray{yT,3}
) where {xT,yT}
    return batched_matmul(internal_operation_mode((x, y)), x, y)
end

function get_batched_matmul_repeat_dims(
    x::AbstractArray{xT,N},
    y::AbstractArray{yT,N},
    lhs_batching_dims::Dims{M},
    rhs_batching_dims::Dims{M},
) where {xT,yT,N,M}
    x_repeats = ones(Int, N)
    y_repeats = ones(Int, N)
    batch_sizes = ones(Int, M)

    for (i, (lhs_dim, rhs_dim)) in enumerate(zip(lhs_batching_dims, rhs_batching_dims))
        sz_x = size(x, lhs_dim)
        sz_y = size(y, rhs_dim)
        batch_sizes[i] = max(sz_x, sz_y)
        sz_x == sz_y && continue
        if sz_x == 1
            x_repeats[lhs_dim] = sz_y
        elseif sz_y == 1
            y_repeats[rhs_dim] = sz_x
        else
            throw(
                DimensionMismatch(
                    lazy"size(x, lhs_dim) = $(sz_x) inconsistent with size(y, rhs_dim) = $(sz_y).",
                ),
            )
        end
    end

    return (
        ntuple(Base.Fix1(getindex, x_repeats), Val(N)),
        ntuple(Base.Fix1(getindex, y_repeats), Val(N)),
        ntuple(Base.Fix1(getindex, batch_sizes), Val(M)),
    )
end

CRC.@non_differentiable get_batched_matmul_repeat_dims(::Any...)

function assert_batched_matmul_checks(
    x::AbstractArray{xT,N},
    y::AbstractArray{yT,N};
    lhs_contracting_dim::Int=2,
    rhs_contracting_dim::Int=1,
    lhs_batching_dims::Dims{M}=ntuple(Base.Fix2(+, 2), Val(N - 2)),
    rhs_batching_dims::Dims{M}=ntuple(Base.Fix2(+, 2), Val(N - 2)),
) where {xT,yT,N,M}
    @assert N ≥ 3 "N must be at least 3"
    @assert M == N - 2 "M = $M must be equal to N - 2 = $N - 2"
    @assert 1 ≤ lhs_contracting_dim ≤ N "lhs_contracting_dim must be between 1 and $N"
    @assert 1 ≤ rhs_contracting_dim ≤ N "rhs_contracting_dim must be between 1 and $N"
    for (lhs_batching_dim, rhs_batching_dim) in zip(lhs_batching_dims, rhs_batching_dims)
        @assert 1 ≤ lhs_batching_dim ≤ N "lhs_batching_dim must be between 1 and $N"
        @assert 1 ≤ rhs_batching_dim ≤ N "rhs_batching_dim must be between 1 and $N"
        @assert lhs_batching_dim ≠ lhs_contracting_dim "lhs_batching_dim must be different \
                                                        from lhs_contracting_dim"
        @assert rhs_batching_dim ≠ rhs_contracting_dim "rhs_batching_dim must be different \
                                                        from rhs_contracting_dim"
        if !(
            size(x, lhs_batching_dim) == size(y, rhs_batching_dim) ||
            size(x, lhs_batching_dim) == 1 ||
            size(y, rhs_batching_dim) == 1
        )
            throw(
                DimensionMismatch(
                    "Batching dimensions mismatch: size(x, $lhs_batching_dim) = $(size(x, lhs_batching_dim)), size(y, $rhs_batching_dim) = $(size(y, rhs_batching_dim))",
                ),
            )
        end
    end
end

function batched_matmul(
    ::GenericBroadcastOp, x::AbstractArray{xT,3}, y::AbstractArray{yT,3}
) where {xT,yT}
    return NNlib.batched_mul(x, y)
end

for dev in (AMDGPUDevice, CUDADevice)
    @eval function batched_matmul(
        ::GPUBroadcastOp{$(dev)}, x::AbstractArray{xT,3}, y::AbstractArray{yT,3}
    ) where {xT,yT}
        return NNlib.batched_mul(x, y)  # GPU versions are well optimized
    end
end

function batched_matmul(
    opmode::GPUBroadcastOp{<:AbstractGPUDevice},
    x::AbstractArray{xT,3},
    y::AbstractArray{yT,3},
) where {xT,yT}
    if isconcretetype(
        Core.Compiler.return_type(NNlib.batched_mul, Tuple{typeof(x),typeof(y)})
    )
        return NNlib.batched_mul(x, y)  # GPU versions are well optimized
    end
    return fallback_batched_matmul(opmode, x, y)
end

function batched_matmul(
    opmode::Union{GPUBroadcastOp{AMDGPUDevice},GenericBroadcastOp{AMDGPUDevice}},
    x::AbstractArray{<:Complex,3},
    y::AbstractArray{<:Complex,3},
)
    return fallback_batched_matmul(opmode, x, y)
end

function batched_matmul(
    opmode::LoopedArrayOp, x::AbstractArray{xT,3}, y::AbstractArray{yT,3}
) where {xT,yT}
    if (size(x, 3) != size(y, 3) && size(x, 3) != 1 && size(y, 3) != 1) ||
        (size(x, 2) != size(y, 1))
        throw(
            DimensionMismatch(
                lazy"size(x) = $(size(x)), size(y) = $(size(y)) inconsistent for batched_matmul.",
            ),
        )
    end
    z = similar(
        x,
        promote_type(eltype(x), eltype(y)),
        size(x, 1),
        size(y, 2),
        max(size(x, 3), size(y, 3)),
    )
    batched_matmul!(z, opmode, x, y)
    return z
end

function batched_matmul!(
    z::AbstractArray{zT,3},
    ::AbstractInternalArrayOpMode,
    x::AbstractArray{xT,3},
    y::AbstractArray{yT,3},
) where {zT,xT,yT}
    NNlib.batched_mul!(z, x, y)
    return nothing
end

function batched_matmul!(
    z::AbstractArray{zT,3}, ::LoopedArrayOp, x::AbstractArray{xT,3}, y::AbstractArray{yT,3}
) where {zT,xT,yT}
    batched_matmul_cpu!(z, x, y)
    return nothing
end

function batched_matmul_cpu!(
    z::AbstractArray{zT,3}, x::AbstractArray{xT,3}, y::AbstractArray{yT,3}
) where {zT,xT,yT}
    if (
        can_loopvec_args(batchview(z, 1), batchview(x, 1), batchview(y, 1)) &&
        !unsafe_known(explicit_blas_loaded())
    )
        batched_matmul_loopvec_impl!(z, x, y)
        return nothing
    end
    NNlib.batched_mul!(z, x, y)
    return nothing
end

function batched_matmul_loopvec_impl! end

function fallback_batched_matmul(
    opmode, x::AbstractArray{xT,3}, y::AbstractArray{yT,3}
) where {xT,yT}
    z = similar(
        x,
        promote_type(eltype(x), eltype(y)),
        size(x, 1),
        size(y, 2),
        max(size(x, 3), size(y, 3)),
    )
    fallback_batched_matmul!(z, opmode, x, y)
    return z
end

function fallback_batched_matmul!(
    z::AbstractArray{zT,3}, opmode, x::AbstractArray{xT,3}, y::AbstractArray{yT,3}
) where {zT,xT,yT}
    @warn "Using fallback Batched Matrix Multiply routine for $(opmode) with A: size = \
           $(size(x)) eltype = $(xT) and B: size = $(size(y)) eltype = $(yT). This may be \
           slow." maxlog = 1

    if (size(x, 3) != size(y, 3) && size(x, 3) != 1 && size(y, 3) != 1) ||
        (size(x, 2) != size(y, 1))
        throw(
            DimensionMismatch(
                lazy"size(x) = $(size(x)), size(y) = $(size(y)) inconsistent for batched_matmul.",
            ),
        )
    end

    if use_threaded_batched_matmul(get_device_type(x))
        unsafe_fallback_threaded_batched_matmul!(z, x, y)
    else
        unsafe_fallback_serial_batched_matmul!(z, x, y)
    end

    return nothing
end

function unsafe_fallback_serial_batched_matmul!(
    z::AbstractArray{zT,3}, x::AbstractArray{xT,3}, y::AbstractArray{yT,3}
) where {zT,xT,yT}
    return if size(x, 3) == size(y, 3)
        for L in axes(z, 3)
            mul!(batchview(z, L), batchview(x, L), batchview(y, L))
        end
    elseif size(x, 3) == 1
        for L in axes(z, 3)
            mul!(batchview(z, L), batchview(x, 1), batchview(y, L))
        end
    else # has to be size(y, 3) == 1
        for L in axes(z, 3)
            mul!(batchview(z, L), batchview(x, L), batchview(y, 1))
        end
    end
end

function unsafe_fallback_threaded_batched_matmul!(
    z::AbstractArray{zT,3}, x::AbstractArray{xT,3}, y::AbstractArray{yT,3}
) where {zT,xT,yT}
    old_threads = maybe_reduce_BLAS_threads(z)

    if size(x, 3) == size(y, 3)
        Threads.@threads for L in axes(z, 3)
            mul!(batchview(z, L), batchview(x, L), batchview(y, L))
        end
    elseif size(x, 3) == 1
        Threads.@threads for L in axes(z, 3)
            mul!(batchview(z, L), batchview(x, 1), batchview(y, L))
        end
    else # has to be size(y, 3) == 1
        Threads.@threads for L in axes(z, 3)
            mul!(batchview(z, L), batchview(x, L), batchview(y, 1))
        end
    end

    reset_BLAS_threads(old_threads)
    return nothing
end

use_threaded_batched_matmul(::Type) = false
use_threaded_batched_matmul(::Type{CUDADevice}) = true
use_threaded_batched_matmul(::Type{CPUDevice}) = true

function CRC.rrule(
    ::typeof(batched_matmul_fallback), x::AbstractArray{xT,3}, y::AbstractArray{yT,3}
) where {xT,yT}
    ∇batched_matmul = @closure Δ -> begin
        ∂x = CRC.@thunk begin
            tmp = batched_matmul_fallback(recursive_unthunk(Δ), NNlib.batched_adjoint(y))
            size(x, 3) == 1 ? sum(tmp; dims=3) : tmp
        end
        ∂y = CRC.@thunk begin
            tmp = batched_matmul_fallback(NNlib.batched_adjoint(x), recursive_unthunk(Δ))
            size(y, 3) == 1 ? sum(tmp; dims=3) : tmp
        end
        return ∂∅, ∂x, ∂y
    end
    return batched_matmul_fallback(x, y), ∇batched_matmul
end

# COV_EXCL_START

# This is type-piracy but needed to fix a blocking issue. TODO: upstream to NNlib
# Enzyme causes a "active variables passed by value to jl_new_task are not yet supported"
# warning without this patch.
for func in (NNlib.batched_mul!, batched_matmul_loopvec_impl!)
    @eval begin
        function EnzymeRules.augmented_primal(
            cfg::EnzymeRules.RevConfigWidth,
            ::EnzymeCore.Const{typeof($(func))},
            ::Type{RT},
            C::EnzymeCore.Annotation{<:AbstractArray{<:Any,3}},
            A::EnzymeCore.Annotation{<:AbstractArray{<:Any,3}},
            B::EnzymeCore.Annotation{<:AbstractArray{<:Any,3}},
        ) where {RT}
            if typeof(C) <: EnzymeCore.Duplicated || typeof(C) <: EnzymeCore.BatchDuplicated
                $(func)(C.val, A.val, B.val)
            end

            primal = EnzymeRules.needs_primal(cfg) ? C.val : nothing
            shadow = EnzymeRules.needs_shadow(cfg) ? C.dval : nothing

            cache_A =
                if (
                    EnzymeRules.overwritten(cfg)[3] &&
                    !(typeof(C) <: EnzymeCore.Const) &&
                    !(typeof(B) <: EnzymeCore.Const)
                )
                    copy(A.val)
                else
                    nothing
                end
            cache_B =
                if (
                    EnzymeRules.overwritten(cfg)[3] &&
                    !(typeof(C) <: EnzymeCore.Const) &&
                    !(typeof(A) <: EnzymeCore.Const)
                )
                    copy(B.val)
                else
                    nothing
                end

            return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_B))
        end

        function EnzymeRules.reverse(
            cfg::EnzymeRules.RevConfigWidth,
            ::EnzymeCore.Const{typeof($(func))},
            ::Type{RT},
            cache,
            C::EnzymeCore.Annotation{<:AbstractArray{<:Any,3}},
            A::EnzymeCore.Annotation{<:AbstractArray{<:Any,3}},
            B::EnzymeCore.Annotation{<:AbstractArray{<:Any,3}},
        ) where {RT}
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
                            for L in axes(B′, 3)
                                mul!(dA′, batchview(dC, L), batchview(B′, L), true, true)
                            end
                        else
                            $(func)(dA, dC, NNlib.batched_adjoint(B.val), true, true)
                        end
                    end

                    if !(typeof(B) <: EnzymeCore.Const) && dB !== B.val
                        if size(dB, 3) == 1 && size(A.val, 3) != 1
                            A′ = NNlib.batched_adjoint(A.val)
                            dB′ = batchview(dB, 1)
                            for L in axes(A′, 3)
                                mul!(dB′, batchview(A′, L), batchview(dC, L), true, true)
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

# COV_EXCL_STOP
