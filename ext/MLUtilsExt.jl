module MLUtilsExt

using MLUtils: MLUtils

using Lux: Lux, DistributedUtils

Lux.is_extension_loaded(::Val{:MLUtils}) = true

function DistributedUtils.construct_distributed_data_container(
    backend::Lux.AbstractLuxDistributedBackend, data
)
    total_size = MLUtils.numobs(data)
    split_across = DistributedUtils.total_workers(backend)
    size_per_worker = Int(ceil(total_size / split_across))

    partitions = collect(Iterators.partition(1:total_size, size_per_worker))
    idxs = collect(partitions[DistributedUtils.local_rank(backend) + 1])

    return DistributedUtils.DistributedDataContainer(data, idxs)
end

function MLUtils.getobs(dc::DistributedUtils.DistributedDataContainer, idx)
    return MLUtils.getobs(dc.data, dc.idxs[idx])
end

end
