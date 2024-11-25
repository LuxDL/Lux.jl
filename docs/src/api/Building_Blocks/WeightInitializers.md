```@meta
CollapsedDocStrings = true
```

# [WeightInitializers](@id WeightInitializers-API)

This package is a light dependency providing common weight initialization schemes for deep
learning models.

## [Supported RNG Types](@id Supported-RNG-Types-WeightInit)

| **RNG Type / Package**            | **Returned Array Type** | **Unsupported Functions**                        |
| --------------------------------- | ----------------------- | ------------------------------------------------ |
| `Random.jl`                       | `Array`                 |                                                  |
| `StableRNGs.jl`                   | `Array`                 |                                                  |
| `CUDA.CURAND.default_rng()`       | `CuArray`               |                                                  |
| `CUDA.default_rng()`              | `CuArray`               |                                                  |
| `GPUArrays.default_rng(CuArray)`  | `CuArray`               |                                                  |
| `AMDGPU.rocrand_rng()`            | `ROCArray`              |                                                  |
| `AMDGPU.gpuarrays_rng()`          | `ROCArray`              |                                                  |
| `GPUArrays.default_rng(ROCArray)` | `ROCArray`              |                                                  |
| `Metal.gpuarrays_rng()`           | `MtlArray`              | [`orthogonal`](@ref)                             |
| `GPUArrays.default_rng(MtlArray)` | `MtlArray`              | [`orthogonal`](@ref)                             |
| `oneAPI.gpuarrays_rng()`          | `oneArray`              | [`orthogonal`](@ref), [`truncated_normal`](@ref) |
| `GPUArrays.default_rng(oneArray)` | `oneArray`              | [`orthogonal`](@ref), [`truncated_normal`](@ref) |

## API Reference

### Main Functions

```@docs
glorot_normal
glorot_uniform
identity_init
kaiming_normal
kaiming_uniform
sparse_init
truncated_normal
orthogonal
```

### Other Convenience Functions

!!! warning "Beware"

    Unlike the other functions these ones don't take a type argument.

```@docs
zeros16
ones16
rand16
randn16
zeros32
ones32
rand32
randn32
zeros64
ones64
rand64
randn64
zerosC16
onesC16
randC16
randnC16
zerosC32
onesC32
randC32
randnC32
zerosC64
onesC64
randC64
randnC64
```
