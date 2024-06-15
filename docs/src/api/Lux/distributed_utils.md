# Distributed Utils

!!! note

    These functionalities are available via the `Lux.DistributedUtils` module.

## Index

```@index
Pages = ["distributed_utils.md"]
```

## [Backends](@id communication-backends)

```@docs
MPIBackend
NCCLBackend
```

## Initialization

```@docs
DistributedUtils.initialize
DistributedUtils.initialized
DistributedUtils.get_distributed_backend
```

## Helper Functions

```@docs
DistributedUtils.local_rank
DistributedUtils.total_workers
```

## Communication Primitives

```@docs
DistributedUtils.allreduce!
DistributedUtils.bcast!
DistributedUtils.reduce!
DistributedUtils.synchronize!!
```

## Optimizers.jl Integration

```@docs
DistributedUtils.DistributedOptimizer
```

## MLUtils.jl Integration

```@docs
DistributedUtils.DistributedDataContainer
```
