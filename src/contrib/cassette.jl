# Logic for hoisting all allocations is based on https://github.com/oxinabox/AutoPreallocation.jl
# with extensions to support AD and GPU Arrays
# The idea is to also switch "slower" NNlib operations to corresponding LuxLib versions
using Cassette: Cassette
using LinearAlgebra: LinearAlgebra
using LoopVectorization: LoopVectorization
using LuxLib: LuxLib, get_device_type
using UnrolledUtilities: UnrolledUtilities

## Allocation Records
abstract type AbstractAllocationRecord end

struct AllocationRecord <: AbstractAllocationRecord
    allocations::Vector{AbstractArray}
    initial_sizes::Vector
end

AllocationRecord() = AllocationRecord(AbstractArray[], [])

@concrete struct FrozenAllocationRecord <: AbstractAllocationRecord
    allocations
    initial_sizes
end

function FrozenAllocationRecord(record::AllocationRecord)
    return FrozenAllocationRecord(Tuple(record.allocations), Tuple(record.initial_sizes))
end

function Base.copy(record::AllocationRecord)
    return AllocationRecord(copy.(record.allocations), record.initial_sizes)
end

function Base.copy(record::FrozenAllocationRecord)
    return FrozenAllocationRecord(copy.(record.allocations), record.initial_sizes)
end

function reinitialize!(record)
    for ii in eachindex(record.allocations)
        alloc = record.allocations[ii]
        sz = record.initial_sizes[ii]
        reinit!(alloc, sz)  # function barrier here prevents allocations
    end
    return record
end

function reinit!(alloc, sz)  # this is a function barrier for inside `reinitialize!`
    # only vectors can be resized, and
    # don't check `size(alloc)` as this allocates, unlike `length(alloc)`
    if ndims(alloc) == 1 && length(alloc) !== first(sz)
        # fix any vectors that were e.g. `push!`ed to.
        resize!(alloc, first(sz))
    end
end

## RecordingCtx

Cassette.@context RecordingCtx
new_recording_ctx() = Cassette.disablehooks(RecordingCtx(; metadata=AllocationRecord()))

function record_alloc!(record::AllocationRecord, val)
    push!(record.initial_sizes, size(val))
    push!(record.allocations, val)
end
record_alloc!(ctx::RecordingCtx, val) = record_alloc!(ctx.metadata, val)

# TODO: Handle GPUArrays
@inline function Cassette.overdub(
        ctx::RecordingCtx, ::Type{Array{T, N}}, ::UndefInitializer, dims) where {T, N}
    ret = Array{T, N}(undef, dims)
    record_alloc!(ctx, ret)
    return ret
end

function record_allocations(f::F, args...; kwargs...) where {F}
    ctx = new_recording_ctx()
    value = Cassette.overdub(ctx, f, args...; kwargs...)
    return (; value, allocation_record=ctx.metadata)
end

struct PreallocatedMethod{F, Args <: Tuple, N, R}
    f::F
    replay_ctxs::NTuple{N, R} # one replay context per thread

    function PreallocatedMethod{F, Args}(f::F, ctxs::NTuple{N, R}) where {F, Args, N, R}
        return new{F, Args, N, R}(f, ctxs)
    end
end

function Base.show(io::IO, f::PreallocatedMethod)
    print(io, "PreallocatedMethod(")
    show(io, f.f)
    print(io, ")")
end

function (f::PreallocatedMethod)(xs...)
    ctx = f.replay_ctxs[Threads.threadid()]
    reinitialize!(ctx.metadata)
    return Cassette.overdub(ctx, f.f, xs...)
end

function preallocate(f::F, xs...) where {F}
    x, record = record_allocations(f, xs...)
    record = FrozenAllocationRecord(record)
    ctxs = ntuple(Threads.nthreads()) do k
        new_replay_ctx(k == 1 ? copy(record) : record)
    end
    return x, PreallocatedMethod{F, typeof(xs)}(f, ctxs)
end

## ReplayCtx

@concrete struct AllocationReplay
    record <: AbstractAllocationRecord
    step::Base.RefValue{Int}
end

AllocationReplay(record) = AllocationReplay(record, Ref(1))

Cassette.@context ReplayCtx

new_replay_ctx(record) = new_replay_ctx(AllocationReplay(record))
function new_replay_ctx(replay::AllocationReplay)
    reinitialize!(replay)
    return Cassette.disablehooks(ReplayCtx(; metadata=replay))
end

function reinitialize!(replay::AllocationReplay)
    reinitialize!(replay.record)
    replay.step[] = 1
    return replay
end

@inline function next_scheduled_alloc!(replay::AllocationReplay)
    step = replay.step[]::Int
    alloc = replay.record.allocations[step]::Array
    replay.step[] = step + 1
    return alloc
end
@inline next_scheduled_alloc!(ctx::ReplayCtx) = next_scheduled_alloc!(ctx.metadata)

@inline function Cassette.overdub(ctx::ReplayCtx, ::Type{Array{T, N}},
        ::UndefInitializer, dims)::Array{T, N} where {T, N}
    scheduled = next_scheduled_alloc!(ctx)::Array{T, N}

    # Commented out until we can workout how to do this without allocations on the happy path
    # It seems like having any branch here makes it allocate
    # TODO: reenable this
    #==
    if  typeof(scheduled) !== Array{T,N} || size(scheduled) !== dims
        @warn "Allocation reuse failed. Indicates value dependent allocations." step=ctx.metadata.step[] expected_T=eltype(scheduled) actual_T=T expected_size=size actual_size=dims
        # Fallback to just doing the allocation
        return Array{T,N}(undef, dims)
    end
    ==#

    return scheduled
end

function avoid_allocations(record, f, args...; kwargs...)
    ctx = new_replay_ctx(record)
    return Cassette.overdub(ctx, f, args...; kwargs...)
end

## Fix inference issues

#! format: off
BLACK_LIST = (
    Base.promote_op,
    Base.to_shape,
    Core.getfield,
    Core.:(===),
    # Core.IntrinsicFunction,
    Base.iterate,
    Broadcast.broadcasted,
    Broadcast.preprocess,
    Broadcast.combine_axes,
    Base.not_int,
    Base.size,
    Base.haskey,
    Base.reduced_indices,
    LinearAlgebra.gemv!,
    Tuple,
    LoopVectorization.avx_launch,
    LoopVectorization._turbo_!,
    # Noe we define the optimized LuxLib ones here
    LuxLib.matmul!,
    LuxLib.matmuladd!,
    LuxLib._fast_activation!,
    get_device_type,
    UnrolledUtilities.unrolled_map,
    UnrolledUtilities.unrolled_mapreduce,
    UnrolledUtilities.unrolled_filter,
    Lux._vec,
)
#! format: on

for F in BLACK_LIST
    @show F
    @eval @inline Cassette.overdub(::RecordingCtx, f::typeof($F), xs...) = f(xs...)
    @eval @inline Cassette.overdub(::ReplayCtx, f::typeof($F), xs...) = f(xs...)
end

@inline Cassette.overdub(::RecordingCtx, ::Type{Val}, x) = Val(x)
@inline Cassette.overdub(::ReplayCtx, ::Type{Val}, x) = Val(x)

@inline function Cassette.overdub(::RecordingCtx, ::typeof(getindex), x::IdDict, key)
    return getindex(x, key)
end
@inline Cassette.overdub(::ReplayCtx, ::typeof(getindex), x::IdDict, key) = getindex(x, key)
