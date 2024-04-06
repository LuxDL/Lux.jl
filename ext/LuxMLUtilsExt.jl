module LuxMLUtilsExt

using Lux: DistributedUtils
using MLUtils: numobs

function DistributedUtils.__construct_distributed_data_container(
        backend::DistributedUtils.AbstractLuxDistributedBackend, data)
    total_size = numobs(data)
    split_across = DistributedUtils.total_workers(backend)
    size_per_worker = Int(ceil(total_size / split_across))

    partitions = collect(Iterators.partition(1:total_size, size_per_worker))
    idxs = collect(partitions[DistributedUtils.local_rank(backend) + 1])

    return DistributedUtils.DistributedDataContainer(data, idxs)
end

end