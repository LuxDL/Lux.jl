
<a id='Installation'></a>

# Installation


Install [Julia v1.6 or above](https://julialang.org/downloads/). Lux.jl is available through the Julia package manager. You can enter it by pressing `]` in the REPL and then typing


```julia
pkg> add Lux
```


Alternatively, you can also do


```julia
import Pkg; Pkg.add("Lux")
```


:::tip


The Julia Compiler is always improving. As such, we recommend using the latest stable version of Julia instead of the LTS.


:::


<a id='Additional Packages'></a>

## Additional Packages


`LuxDL` hosts various packages that provide additional functionality for Lux.jl. All packages mentioned in this documentation are available via the Julia General Registry.


You can install all those packages via `import Pkg; Pkg.add(<package name>)`.


<a id='GPU Support'></a>

## GPU Support


GPU Support for Lux.jl requires loading additional packages:


  * [`LuxCUDA.jl`](https://github.com/LuxDL/LuxCUDA.jl) for CUDA support.
  * [`LuxAMDGPU.jl`](https://github.com/LuxDL/LuxAMDGPU.jl) for AMDGPU support.
  * [`Metal.jl`](https://github.com/JuliaGPU/Metal.jl) for Apple Metal support.

