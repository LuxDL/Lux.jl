module MLUtilsExt

using Adapt: Adapt
using MLDataDevices: MLDataDevices, AbstractDevice, DeviceIterator
using MLUtils: MLUtils, DataLoader

MLDataDevices.isleaf(::DataLoader) = true

function Adapt.adapt_structure(dev::AbstractDevice, dataloader::DataLoader)
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
                data; dataloader.batchsize, dataloader.partial, dataloader.collate
            )
        else
            data
        end

        return DeviceIterator(identity, eachobsparallel(dev, data))
    end
    return DeviceIterator(dev, dataloader)
end

function eachobsparallel(dev::AbstractDevice, data)
    return MLUtils.Loader(1:MLUtils.numobs(data)) do ch, i
        obs = MLUtils.getobs(data, i)
        put!(ch, dev(obs))
    end
end

end
