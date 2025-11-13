module LuxGPUArraysExt

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

for inplace in ("!", "")
    step_with_alloc_cache = Symbol(:single_train_step_impl_with_allocator_cache, inplace)
    step_inner = Symbol(:single_train_step_impl, inplace)
    apply_gradients_with_alloc_cache = Symbol(
        :apply_gradients_with_allocator_cache, inplace
    )
    apply_fn = Symbol(:apply_gradients_impl, inplace)

    @eval begin
        function Training.$(apply_gradients_with_alloc_cache)(
            alloc_cache::AllocCache, ts::Training.TrainState, grads
        )
            @cached alloc_cache begin
                return Training.$(apply_fn)(ts, grads)
            end
        end

        function Training.$(step_with_alloc_cache)(
            backend, alloc_cache::AllocCache, obj_fn::F, data, ts::Training.TrainState
        ) where {F}
            @cached alloc_cache begin
                return Training.$(step_inner)(backend, obj_fn, data, ts)
            end
        end
    end
end

end
