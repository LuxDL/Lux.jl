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

1. `automatic_nested_ad_switching` - Set this to `false` to disable automatic switching
   of backends for nested automatic differentiation. See the manual section on
   [nested automatic differentiation](@ref nested_autodiff) for more details.

## [GPU-Aware MPI Support](@id gpu-aware-mpi-preferences)

If you are using a custom MPI build that supports CUDA or ROCM, you can use the following
preferences with [Preferences.jl](https://github.com/JuliaPackaging/Preferences.jl):

1. `cuda_aware_mpi` - Set this to `true` if your MPI build is CUDA aware.
2. `rocm_aware_mpi` - Set this to `true` if your MPI build is ROCM aware.

By default, both of these preferences are set to `false`.

## GPU Backend Selection

1. `gpu_backend` - Set this to bypass the automatic backend selection and use a specific
   gpu backend. Valid options are "cuda", "rocm", "metal", and "oneapi". This preference
   needs to be set for `LuxDeviceUtils` package. It is recommended to use
   [`LuxDeviceUtils.gpu_backend!`](@ref) to set this preference.

## [Automatic Eltype Conversion](@id automatic-eltypes-preference)

1. `eltype_mismatch_handling` - Preference controlling what happens when layers get
   different eltypes as input. See the documentation on [`match_eltype`](@ref) for more
   details.
