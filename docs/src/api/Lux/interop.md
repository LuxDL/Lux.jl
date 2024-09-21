```@meta
CollapsedDocStrings = true
CurrentModule = Lux
```

# Interoperability between Lux and other packages

## Switching from older frameworks

### [Flux Models to Lux Models](@id flux-to-lux-migrate-api)

`Flux.jl` has been around in the Julia ecosystem for a long time and has a large userbase,
hence we provide a way to convert `Flux` models to `Lux` models.

!!! tip

    Accessing these functions require manually loading `Flux`, i.e., `using Flux` must be
    present somewhere in the code for these to be used.

```@docs
Lux.adapt(::FromFluxAdaptor, l)
FromFluxAdaptor
FluxLayer
```

## Using a different backend for Lux

### Lux Models to Simple Chains

`SimpleChains.jl` provides a way to train Small Neural Networks really fast on CPUs.
See [this blog post](https://julialang.org/blog/2022/04/simple-chains/) for more details.
This section describes how to convert `Lux` models to `SimpleChains` models while
preserving the [layer interface](@ref lux-interface).

!!! tip

    Accessing these functions require manually loading `SimpleChains`, i.e.,
    `using SimpleChains` must be present somewhere in the code for these to be used.

```@docs
Adapt.adapt(from::ToSimpleChainsAdaptor, L::AbstractLuxLayer)
ToSimpleChainsAdaptor
SimpleChainsLayer
```
