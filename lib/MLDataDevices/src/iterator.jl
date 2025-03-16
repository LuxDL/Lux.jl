"""
    DeviceIterator(dev::AbstractDevice, iterator)

Create a `DeviceIterator` that iterates through the provided `iterator` via `iterate`. Upon
each iteration, the current batch is copied to the device `dev`, and the previous iteration
is marked as freeable from GPU memory (via `unsafe_free!`) (no-op for a CPU device).

The conversion follows the same semantics as `dev(<item from iterator>)`.

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
end

function Base.iterate(c::DeviceIterator)
    item = iterate(c.iterator)
    item === nothing && return nothing
    batch, next_state = item
    dev_batch = c.dev(batch)
    return dev_batch, (next_state, dev_batch)
end

function Base.iterate(c::DeviceIterator, (state, prev_batch))
    item = iterate(c.iterator, state)
    item === nothing && return nothing
    batch, next_state = item
    Internal.unsafe_free!(prev_batch)  # free the previous batch
    dev_batch = c.dev(batch)
    return dev_batch, (next_state, dev_batch)
end

Base.IteratorSize(::Type{DeviceIterator{D,I}}) where {D,I} = Base.IteratorSize(I)
Base.length(c::DeviceIterator) = length(c.iterator)
Base.axes(c::DeviceIterator) = axes(c.iterator)

Base.IteratorEltype(::Type{DeviceIterator{D,I}}) where {D,I} = Base.EltypeUnknown()
