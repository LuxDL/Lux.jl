# Distributed Data Parallel Training

!!! tip

    For a fully functional example, see the
    [ImageNet Training Example](https://github.com/LuxDL/Lux.jl/tree/main/examples/ImageNet).

DDP Training using `Lux.DistributedUtils` is a spiritual successor to
[FluxMPI.jl](https://github.com/avik-pal/FluxMPI.jl), but has some key differences.

## Guide to Integrating DistributedUtils into your code

* Initialize the respective backend with [`DistributedUtils.initialize`](@ref), by passing
  in a backend type. It is important that you pass in the type, i.e. `NCCLBackend` and not
  the object `NCCLBackend()`.

  ```julia
  DistributedUtils.initialize(NCCLBackend)
  ```

* Obtain the backend via [`DistributedUtils.get_distributed_backend`](@ref) by passing in
  the type of the backend (same note as last point applies here again).

  ```julia
  backend = DistributedUtils.get_distributed_backend(NCCLBackend)
  ```

  It is important that you use this function instead of directly constructing the backend,
  since there are certain internal states that need to be synchronized.

* Next synchronize the parameters and states of the model. This is done by calling
  [`DistributedUtils.synchronize!!`](@ref) with the backend and the respective input.

  ```julia
  ps = DistributedUtils.synchronize!!(backend, ps)
  st = DistributedUtils.synchronize!!(backend, st)
  ```

* To split the data uniformly across the processes use
  [`DistributedUtils.DistributedDataContainer`](@ref). Alternatively, one can manually
  split the data. For the provided container to work
  [`MLUtils.jl`](https://github.com/JuliaML/MLUtils.jl) must be installed and loaded.

  ```julia
  data = DistributedUtils.DistributedDataContainer(backend, data)
  ```

* Wrap the optimizer in [`DistributedUtils.DistributedOptimizer`](@ref) to ensure that the
  optimizer is correctly synchronized across all processes before parameter updates. After
  initializing the state of the optimizer, synchronize the state across all processes.

  ```julia
  opt = DistributedUtils.DistributedOptimizer(backend, opt)
  opt_state = Optimisers.setup(opt, ps)
  opt_state = DistributedUtils.synchronize!!(backend, opt_state)
  ```

* Finally change all logging and serialization code to trigger on
  `local_rank(backend) == 0`. This ensures that only the master process logs and serializes
  the model.

## Migration Guide from `FluxMPI.jl`

Let's compare the changes we need to make wrt the
[FluxMPI.jl integration guide](https://avik-pal.github.io/FluxMPI.jl/dev/guide/).

1. `FluxMPI.Init` is now [`DistributedUtils.initialize`](@ref).
2. `FluxMPI.synchronize!(x)` needs to be changed to
   `x_new = DistributedUtils.synchronize!!(backend, x)`.
3. [`DistributedUtils.DistributedDataContainer`](@ref),
   [`DistributedUtils.local_rank`](@ref), and
   [`DistributedUtils.DistributedOptimizer`](@ref) need `backend` as  the first input.

And that's pretty much it!

### Removed Functionality

1. `FluxMPI.allreduce_gradients` no longer exists. Previously this was needed when CUDA
   communication was flaky, with `NCCL.jl` this is no longer the case.
2. `FluxMPIFluxModel` has been removed. `DistributedUtils` no longer works with `Flux`.

### Key Differences

1. `FluxMPI.synchronize!` is now `DistributedUtils.synchronize!!` to highlight the fact
   that some of the inputs are not updated in-place.
2. All of the functions now require a [communication backend](@ref communication-backends)
   as input.
3. We don't automatically determine if the MPI Implementation is CUDA or ROCM aware. See
   [GPU-aware MPI](@ref gpu-aware-mpi-preferences) for more information.
4. Older (now non-existent) `Lux.gpu` implementations used to "just work" with `FluxMPI.jl`.
   We expect [`gpu_device`](@ref) to continue working as expected, however, we recommend
   using [`gpu_device`](@ref) after calling [`DistributedUtils.initialize`](@ref) to avoid
   any mismatch between the device set via `DistributedUtils` and the device stores in
   `CUDADevice` or `AMDGPUDevice`.

## Known Shortcomings

1. Currently we don't run tests with CUDA or ROCM aware MPI, use those features at your own
   risk. We are working on adding tests for these features.
2. Native AMDGPU.jl support is experimental and causes deadlocks in certain situations. **For
   AMD GPUs, we strongly recommend using Reactant instead of native AMDGPU.jl for distributed
   training.** If you have a minimal reproducer for AMDGPU.jl issues, please open an issue.
