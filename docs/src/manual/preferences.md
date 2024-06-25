# Preferences for Lux.jl

!!! tip "How to set Preferences"

    [PreferenceTools.jl](https://github.com/cjdoris/PreferenceTools.jl) provides an
    interactive way to set preferences. First run the following command:

    ```julia
    using PreferenceTools
    ```

    Then in the pkg mode (press `]` in the REPL), run the following command:

    ```julia
    pkg> preference add Lux <preference-name>=<value>
    ```

Lux.jl relies on several preferences to make decision on how to run your code. Here is an
exhaustive list of preferences that Lux.jl uses.

## Nested Automatic Differentiation

1. `automatic_nested_ad_switching` - Set this to `true` to disable automatic switching
   of backends for nested automatic differentiation. See the manual section on
   [nested automatic differentiation](@ref nested_autodiff) for more details.

## GPU-Aware MPI Support

1. `cuda_aware_mpi` - Set this to `true` if your MPI build is CUDA aware.
2. `rocm_aware_mpi` - Set this to `true` if your MPI build is ROCM aware.

## GPU Backend Selection

1. `gpu_backend` - Set this to bypass the automatic backend selection and use a specific
   gpu backend. Valid options are "cuda", "rocm", "metal", and "oneapi". This preference
   needs to be set for `LuxDeviceUtils` package. It is recommended to use
   [`LuxDeviceUtils.gpu_backend!`](@ref) to set this preference.

## Automatic Eltype Conversion

1. `automatic_eltype_conversion` - Preference controlling what happens when layers get
   different eltypes as input:

    1. `none` - This is the default, where no automatic eltype conversion is done. Type
       promotion rules of Julia are followed.
    2. `warn` - A warning is issued when eltypes are different and we determine that this
       was non-intentional. However, no automatic conversion is done and promotion rules of
       Julia are followed.
    3. `convert` - Automatic conversion is done to the "most performant" eltype (warning is
       still printed). This is not recommended for most users. Instead we recommend using
       the next preference, and fixing the problematic layers.
    4. `error` - An error is thrown when eltypes are different. This is the recommended when
       debugging performance issues.
