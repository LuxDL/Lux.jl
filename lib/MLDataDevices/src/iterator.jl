"""
    DeviceIterator(dev::AbstractDevice, iterator; prefetch::Int=1)

Create a `DeviceIterator` that iterates through the provided `iterator` via `iterate`. Upon
each iteration, the current batch is copied to the device `dev`, and the previous iteration
is marked as freeable from GPU memory (via `unsafe_free!`) (no-op for a CPU device).

The conversion follows the same semantics as `dev(<item from iterator>)`.

When `prefetch > 0` and `dev` supports asynchronous prefetching (such as `CUDADevice`),
batches are staged via reusable pinned host memory and prefetched up to `prefetch` batches
ahead on a non-blocking transfer stream. Pass `prefetch=0` to force synchronous transfers.

!!! tip "Similarity to `CUDA.CuIterator`"

    The design inspiration was taken from `CUDA.CuIterator` and was generalized to work with
    other backends and more complex iterators (using `Functors`).

!!! tip "`MLUtils.DataLoader`"

    Calling `dev(::MLUtils.DataLoader)` will automatically convert the dataloader to use the
    same semantics as `DeviceIterator`. This is generally preferred over looping over the
    dataloader directly and transferring the data to the device.

## Examples

The following was run on a computer with an NVIDIA GPU.

```julia-repl
julia> using MLDataDevices, MLUtils

julia> X = rand(Float64, 3, 33);

julia> dataloader = DataLoader(X; batchsize=13, shuffle=false);

julia> for (i, x) in enumerate(dataloader)
           @show i, summary(x)
       end
(i, summary(x)) = (1, "3×13 Matrix{Float64}")
(i, summary(x)) = (2, "3×13 Matrix{Float64}")
(i, summary(x)) = (3, "3×7 Matrix{Float64}")

julia> for (i, x) in enumerate(CUDADevice()(dataloader))
           @show i, summary(x)
       end
(i, summary(x)) = (1, "3×13 CuArray{Float32, 2, CUDA.DeviceMemory}")
(i, summary(x)) = (2, "3×13 CuArray{Float32, 2, CUDA.DeviceMemory}")
(i, summary(x)) = (3, "3×7 CuArray{Float32, 2, CUDA.DeviceMemory}")
```
"""
struct DeviceIterator{D<:Function,I}
    dev::D
    iterator::I
    prefetch::Int

    function DeviceIterator(dev::D, iterator::I; prefetch::Int=1) where {D<:Function,I}
        prefetch >= 0 || throw(ArgumentError("`prefetch` must be >= 0, got $(prefetch)"))
        return new{D,I}(dev, iterator, prefetch)
    end
end

struct SyncIteratorState{S,B}
    iter_state::S
    prev_batch::B
end

struct PrefetchError
    exception::Any
    backtrace::Any
end

mutable struct PrefetchQueueState
    const lock::ReentrantLock
    const cond::Threads.Condition
    const capacity::Int
    const queue::Vector{Any}
    iter_state::Any
    exhausted::Bool
    worker_running::Bool
    worker::Union{Nothing,Task}
    error::Union{Nothing,PrefetchError}
end

struct PrefetchIteratorState{BK,B}
    backend::BK
    qstate::PrefetchQueueState
    prev_batch::B
end

function _prefetch_worker_loop!(
    c::DeviceIterator, backend, qstate::PrefetchQueueState, prev_worker::Union{Nothing,Task}
)
    if prev_worker !== nothing
        try
            wait(prev_worker)
        catch
        end
    end
    try
        Internal.with_prefetch_context(backend) do
            while true
                state, should_continue = Base.@lock qstate.cond begin
                    if qstate.exhausted ||
                        qstate.error !== nothing ||
                        length(qstate.queue) >= qstate.capacity
                        qstate.worker_running = false
                        notify(qstate.cond)
                        nothing, false
                    else
                        qstate.iter_state, true
                    end
                end
                should_continue || break

                item = iterate(c.iterator, state)
                if item === nothing
                    Internal.cleanup_prefetch_backend!(backend)
                    Base.@lock qstate.cond begin
                        qstate.exhausted = true
                        qstate.worker_running = false
                        notify(qstate.cond)
                    end
                    break
                end

                batch, next_state = item
                dev_batch, ready_token = Internal.prefetch_transfer!(c.dev, backend, batch)

                done = Base.@lock qstate.cond begin
                    push!(qstate.queue, (dev_batch, ready_token))
                    qstate.iter_state = next_state
                    at_cap = length(qstate.queue) >= qstate.capacity
                    if at_cap
                        qstate.worker_running = false
                    end
                    notify(qstate.cond)
                    at_cap
                end
                done && break
            end
        end
    catch e
        bt = catch_backtrace()
        Base.@lock qstate.cond begin
            qstate.error = PrefetchError(e, bt)
            qstate.worker_running = false
            notify(qstate.cond)
        end
    finally
        Base.@lock qstate.cond begin
            if qstate.worker_running
                qstate.worker_running = false
                notify(qstate.cond)
            end
        end
    end
    return nothing
end

function _ensure_prefetch_worker!(c::DeviceIterator, backend, qstate::PrefetchQueueState)
    Base.@lock qstate.cond begin
        if qstate.worker_running ||
            qstate.exhausted ||
            qstate.error !== nothing ||
            length(qstate.queue) >= qstate.capacity
            return nothing
        end
        prev_worker = qstate.worker
        qstate.worker_running = true
        task = Threads.@spawn _prefetch_worker_loop!(c, backend, qstate, prev_worker)
        qstate.worker = task
    end
    return nothing
end

function Base.iterate(c::DeviceIterator)
    item = iterate(c.iterator)
    item === nothing && return nothing
    batch, next_state = item
    dev_batch = c.dev(batch)
    if c.prefetch > 0 && Internal.supports_prefetch(c.dev)
        backend = Internal.init_prefetch_backend(c.dev, c.prefetch)
        if backend !== nothing
            lock = ReentrantLock()
            qstate = PrefetchQueueState(
                lock,
                Threads.Condition(lock),
                c.prefetch,
                Any[],
                next_state,
                false,
                false,
                nothing,
                nothing,
            )
            _ensure_prefetch_worker!(c, backend, qstate)
            return dev_batch, PrefetchIteratorState(backend, qstate, dev_batch)
        end
    end
    return dev_batch, SyncIteratorState(next_state, dev_batch)
end

function Base.iterate(c::DeviceIterator, state::SyncIteratorState)
    item = iterate(c.iterator, state.iter_state)
    item === nothing && return nothing
    batch, next_state = item
    Internal.unsafe_free!(state.prev_batch)  # free the previous batch
    dev_batch = c.dev(batch)
    return dev_batch, SyncIteratorState(next_state, dev_batch)
end

function Base.iterate(c::DeviceIterator, state::PrefetchIteratorState)
    qstate = state.qstate
    entry, err = Base.@lock qstate.cond begin
        while isempty(qstate.queue) && qstate.error === nothing && !qstate.exhausted
            if !qstate.worker_running
                break
            end
            wait(qstate.cond)
        end
        if !isempty(qstate.queue)
            popfirst!(qstate.queue), nothing
        elseif qstate.error !== nothing
            nothing, qstate.error
        else
            nothing, nothing
        end
    end
    if err !== nothing
        throw(err.exception)
    end
    if entry === nothing
        if !qstate.exhausted
            _ensure_prefetch_worker!(c, state.backend, qstate)
            return iterate(c, state)
        end
        return nothing
    end
    dev_batch, ready_token = entry
    Internal.prefetch_retire!(state.backend, state.prev_batch)
    Internal.prefetch_consume!(state.backend, ready_token)
    _ensure_prefetch_worker!(c, state.backend, qstate)
    return dev_batch, PrefetchIteratorState(state.backend, qstate, dev_batch)
end

Base.IteratorSize(::Type{DeviceIterator{D,I}}) where {D,I} = Base.IteratorSize(I)
Base.length(c::DeviceIterator) = length(c.iterator)
Base.axes(c::DeviceIterator) = axes(c.iterator)

Base.IteratorEltype(::Type{DeviceIterator{D,I}}) where {D,I} = Base.EltypeUnknown()
