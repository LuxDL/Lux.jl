module MLDataDevicesMLUtilsExt

using MLDataDevices: MLDataDevices, AbstractDevice, AbstractDeviceIterator, CPUDevice,
                     CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice, DeviceIterator,
                     Internal
using MLUtils: MLUtils, DataLoader

for dev in (CPUDevice, CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice)
    @eval function (D::$(dev))(dataloader::DataLoader)
        if dataloader.parallel
            if dataloader.buffer
                @warn "Using `buffer=true` for parallel DataLoader with automatic device \
                       transfer is currently not implemented. Ignoring `buffer=true`."
            end
            return ParallelDeviceDataLoader(D, dataloader)
        end
        return DeviceIterator(D, dataloader)
    end
end

# Parallel DataLoader that does the device transfer in the same task
struct ParallelDeviceDataLoader{D <: AbstractDevice, DL <: DataLoader} <:
       AbstractDeviceIterator{D, DL}
    dev::D
    iterator::DL
end

# Mostly from https://github.com/JuliaML/MLUtils.jl/blob/main/src/eachobs.jl
function Base.iterate(c::ParallelDeviceDataLoader)
    data = MLUtils.ObsView(c.iterator.data)

    data = c.iterator.shuffle ? MLUtils.shuffleobs(c.iterator.rng, data) : data
    data = if c.iterator.batchsize > 0
        MLUtils.BatchView(
            data; c.iterator.batchsize, c.iterator.partial, c.iterator.collate)
    else
        data
    end

    iter = eachobsparallel(c.dev, data)
    item = iterate(iter)
    item === nothing && return nothing
    dev_batch, next_state = item
    return dev_batch, ((iter, next_state), dev_batch)
end

function Base.iterate(::ParallelDeviceDataLoader, ((iter, state), prev_batch))
    item = iterate(iter, state)
    item === nothing && return nothing
    dev_batch, next_state = item
    Internal.unsafe_free!(prev_batch)  # free the previous batch
    return dev_batch, ((iter, next_state), dev_batch)
end

function eachobsparallel(dev::AbstractDevice, data)
    return MLUtils.Loader(1:MLUtils.numobs(data)) do ch, i
        obs = MLUtils.getobs(data, i)
        put!(ch, dev(obs))
    end
end

end
