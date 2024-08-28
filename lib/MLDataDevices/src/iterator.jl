# This is based on CuIterator but generalized to work with any device
struct DeviceIterator{D <: AbstractDevice, I}
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

Base.IteratorSize(::Type{DeviceIterator{D, I}}) where {D, I} = Base.IteratorSize(I)
Base.length(c::DeviceIterator) = length(c.iterator)
Base.axes(c::DeviceIterator) = axes(c.iterator)

Base.IteratorEltype(::Type{DeviceIterator{D, I}}) where {D, I} = Base.EltypeUnknown()
