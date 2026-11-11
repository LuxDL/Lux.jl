module MLUtilsExt

using MLUtils: MLUtils

using Lux: Lux, DistributedUtils

Lux.is_extension_loaded(::Val{:MLUtils}) = true

function DistributedUtils.construct_distributed_data_container(
    backend::Lux.AbstractLuxDistributedBackend, data
)
    total_size = MLUtils.numobs(data)
    split_across = DistributedUtils.total_workers(backend)
    size_per_worker = cld(total_size, split_across)
    rank = DistributedUtils.local_rank(backend)
    start_idx = min(rank * size_per_worker + 1, total_size + 1)
    end_idx = min((rank + 1) * size_per_worker, total_size)
    idxs = collect(start_idx:end_idx)

    return DistributedUtils.DistributedDataContainer(data, idxs)
end

function MLUtils.getobs(dc::DistributedUtils.DistributedDataContainer, idx)
    return MLUtils.getobs(dc.data, dc.idxs[idx])
end

end
