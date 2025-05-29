# MLDataDevices

`MLDataDevices.jl` is a lightweight package defining rules for transferring data across
devices. It is used in deep learning frameworks such as [Lux.jl](https://lux.csail.mit.edu/) and [Flux.jl](https://fluxml.ai/).

Currently we provide support for the following backends:

1. `CPUDevice`: for CPUs -- no additional packages required.
2. `CUDADevice`: `CUDA.jl` for NVIDIA GPUs.
3. `AMDGPUDevice`: `AMDGPU.jl` for AMD ROCM GPUs.
4. `MetalDevice`: `Metal.jl` for Apple Metal GPUs. **(Experimental)**
5. `oneAPIDevice`: `oneAPI.jl` for Intel GPUs. **(Experimental)**
6. `ReactantDevice`: `Reactant.jl` for XLA Support.

## Updating to v1.0

- Package was renamed from `LuxDeviceUtils.jl` to `MLDataDevices.jl`.
- `Lux(***)Device` has been renamed to `(***)Device`.
- `Lux(***)Adaptor` objects have been removed. Use `(***)Device` objects instead.
