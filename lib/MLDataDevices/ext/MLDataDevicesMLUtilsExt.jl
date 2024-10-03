module MLDataDevicesMLUtilsExt

using MLDataDevices: MLDataDevices, AbstractDevice, CPUDevice, CUDADevice, AMDGPUDevice,
                     MetalDevice, oneAPIDevice, XLADevice, DeviceIterator
using MLUtils: MLUtils, DataLoader

for dev in (CPUDevice, CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice, XLADevice)
    @eval function (D::$(dev))(dataloader::DataLoader)
        if dataloader.parallel
            if dataloader.buffer
                @warn "Using `buffer=true` for parallel DataLoader with automatic device \
                       transfer is currently not implemented. Ignoring `buffer=true`."
            end

            # Mostly from https://github.com/JuliaML/MLUtils.jl/blob/main/src/eachobs.jl
            data = MLUtils.ObsView(dataloader.data)
            data = dataloader.shuffle ? MLUtils.shuffleobs(data) : data
            data = if dataloader.batchsize > 0
                MLUtils.BatchView(
                    data; dataloader.batchsize, dataloader.partial, dataloader.collate)
            else
                data
            end

            return DeviceIterator(D, eachobsparallel(D, data))
        end
        return DeviceIterator(D, dataloader)
    end
end

function eachobsparallel(dev::AbstractDevice, data)
    return MLUtils.Loader(1:MLUtils.numobs(data)) do ch, i
        obs = MLUtils.getobs(data, i)
        put!(ch, dev(obs))
    end
end

end
