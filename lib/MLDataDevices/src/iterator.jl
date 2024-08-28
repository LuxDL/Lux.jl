abstract type AbstractDeviceIterator{D <: AbstractDevice, I} end

function Base.IteratorSize(::Type{AbstractDeviceIterator{D, I}}) where {D, I}
    return Base.IteratorSize(I)
end
Base.length(c::AbstractDeviceIterator) = length(c.iterator)
Base.axes(c::AbstractDeviceIterator) = axes(c.iterator)

function Base.IteratorEltype(::Type{AbstractDeviceIterator{D, I}}) where {D, I}
    return Base.IteratorEltype(I)
end
Base.eltype(c::AbstractDeviceIterator) = eltype(c.iterator)

# This is based on CuIterator but generalized to work with any device
struct DeviceIterator{D, I} <: AbstractDeviceIterator{D, I}
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
