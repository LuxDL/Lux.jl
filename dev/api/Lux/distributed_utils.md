---
url: /dev/api/Lux/distributed_utils.md
---
# Distributed Utils {#Distributed-Utils}

::: tip Note

These functionalities are available via the `Lux.DistributedUtils` module.

:::

## Backends {#communication-backends}

```julia
MPIBackend(comm = nothing)
```

Create an MPI backend for distributed training. Users should not use this function directly. Instead use [`DistributedUtils.get_distributed_backend(MPIBackend)`](/api/Lux/distributed_utils#Lux.DistributedUtils.get_distributed_backend).

source

```julia
NCCLBackend(comm = nothing, mpi_backend = nothing)
```

Create an NCCL backend for distributed training. Users should not use this function directly. Instead use [`DistributedUtils.get_distributed_backend(NCCLBackend)`](/api/Lux/distributed_utils#Lux.DistributedUtils.get_distributed_backend).

source

## Initialization {#Initialization}

```julia
initialize(backend::Type{<:AbstractLuxDistributedBackend}; kwargs...)
```

Initialize the given backend. Users can supply `cuda_devices` and `amdgpu_devices` to initialize the backend with the given devices. These can be set to `missing` to prevent initialization of the given device type. If set to `nothing`, and the backend is functional we assign GPUs in a round-robin fashion. Finally, a list of integers can be supplied to initialize the backend with the given devices.

Possible values for `backend` are:

* `MPIBackend`: MPI backend for distributed training. Requires `MPI.jl` to be installed.

* `NCCLBackend`: NCCL backend for CUDA distributed training. Requires `CUDA.jl`, `MPI.jl`, and `NCCL.jl` to be installed. This also wraps `MPI` backend for non-CUDA communications.

source

```julia
initialized(backend::Type{<:AbstractLuxDistributedBackend})
```

Check if the given backend is initialized.

source

```julia
get_distributed_backend(backend::Type{<:AbstractLuxDistributedBackend})
```

Get the distributed backend for the given backend type. Possible values are:

* `MPIBackend`: MPI backend for distributed training. Requires `MPI.jl` to be installed.

* `NCCLBackend`: NCCL backend for CUDA distributed training. Requires `CUDA.jl`, `MPI.jl`, and `NCCL.jl` to be installed. This also wraps `MPI` backend for non-CUDA communications.

::: danger Danger

`initialize(backend; kwargs...)` must be called before calling this function.

:::

source

## Helper Functions {#Helper-Functions}

```julia
local_rank(backend::AbstractLuxDistributedBackend)
```

Get the local rank for the given backend.

source

```julia
total_workers(backend::AbstractLuxDistributedBackend)
```

Get the total number of workers for the given backend.

source

## Communication Primitives {#Communication-Primitives}

```julia
allreduce!(backend::AbstractLuxDistributedBackend, sendrecvbuf, op)
allreduce!(backend::AbstractLuxDistributedBackend, sendbuf, recvbuf, op)
```

Backend Agnostic API to perform an allreduce operation on the given buffer `sendrecvbuf` or `sendbuf` and store the result in `recvbuf`.

`op` allows a special `DistributedUtils.avg` operation that averages the result across all workers.

source

```julia
bcast!(backend::AbstractLuxDistributedBackend, sendrecvbuf; root::Int=0)
bcast!(backend::AbstractLuxDistributedBackend, sendbuf, recvbuf; root::Int=0)
```

Backend Agnostic API to broadcast the given buffer `sendrecvbuf` or `sendbuf` to all workers into `recvbuf`. The value at `root` will be broadcasted to all other workers.

source

```julia
reduce!(backend::AbstractLuxDistributedBackend, sendrecvbuf, op; root::Int=0)
reduce!(backend::AbstractLuxDistributedBackend, sendbuf, recvbuf, op; root::Int=0)
```

Backend Agnostic API to perform a reduce operation on the given buffer `sendrecvbuf` or `sendbuf` and store the result in `recvbuf`.

`op` allows a special `DistributedUtils.avg` operation that averages the result across all workers.

source

```julia
synchronize!!(backend::AbstractLuxDistributedBackend, ps; root::Int=0)
```

Synchronize the given structure `ps` using the given backend. The value at `root` will be broadcasted to all other workers.

source

## Optimizers.jl Integration {#Optimizers.jl-Integration}

```julia
DistributedOptimizer(backend::AbstractLuxDistributedBacked, optimizer)
```

Wrap the `optimizer` in a `DistributedOptimizer`. Before updating the parameters, this averages the gradients across the processes using Allreduce.

**Arguments**

* `optimizer`: An Optimizer compatible with the Optimisers.jl package

source

## MLUtils.jl Integration {#MLUtils.jl-Integration}

```julia
DistributedDataContainer(backend::AbstractLuxDistributedBackend, data)
```

`data` must be compatible with `MLUtils` interface. The returned container is compatible with `MLUtils` interface and is used to partition the dataset across the available processes.

::: danger Load `MLUtils.jl`

`MLUtils.jl` must be installed and loaded before using this.

:::

source
