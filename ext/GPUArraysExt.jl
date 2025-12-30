module GPUArraysExt

using GPUArrays: AllocCache, @cached
using Lux: Training
using MLDataDevices: AbstractGPUDevice

Training.get_allocator_cache(::AbstractGPUDevice) = AllocCache()

function Training.compute_gradients_impl_with_allocator_cache(
    backend, alloc_cache::AllocCache, obj_fn::F, data, ts::Training.TrainState
) where {F}
    @cached alloc_cache begin
        return Training.compute_gradients_impl(backend, obj_fn, data, ts)
    end
end

function Training.apply_gradients_with_allocator_cache!(
    alloc_cache::AllocCache, ts::Training.TrainState, grads
)
    @cached alloc_cache begin
        return Training.apply_gradients_impl!(ts, grads)
    end
end

function Training.single_train_step_impl_with_allocator_cache!(
    backend, alloc_cache::AllocCache, obj_fn::F, data, ts::Training.TrainState
) where {F}
    @cached alloc_cache begin
        return Training.single_train_step_impl!(backend, obj_fn, data, ts)
    end
end

end
