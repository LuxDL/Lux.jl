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
    pkg> preference add LuxLib <preference-name>=<value>
    pkg> preference add LuxCore <preference-name>=<value>
    ```

Lux.jl relies on several preferences to make decision on how to run your code. Here is an
exhaustive list of preferences that Lux.jl uses.

## Nested Automatic Differentiation

1. `automatic_nested_ad_switching` - Set this to `false` to disable automatic switching
   of backends for nested automatic differentiation. See the manual section on
   [nested automatic differentiation](@ref nested_autodiff) for more details.

## Training with Reactant

1. `precision_config` - Set this to `"auto"` to use select `HIGH` precision for CUDA and
   `DEFAULT` precision for other backends. The choice made here can change across versions
   of Lux.jl and is generally recommended for best performance. Alternatively, you can set
   this to `"default"` to use `DEFAULT` precision or `"high"` to use `HIGH` precision or
   `"highest"` to use `HIGHEST` precision.

## [GPU-Aware MPI Support](@id gpu-aware-mpi-preferences)

If you are using a custom MPI build that supports CUDA or ROCM, you can use the following
preferences with [Preferences.jl](https://github.com/JuliaPackaging/Preferences.jl):

1. `cuda_aware_mpi` - Set this to `true` if your MPI build is CUDA aware.
2. `rocm_aware_mpi` - Set this to `true` if your MPI build is ROCM aware.

By default, both of these preferences are set to `false`.

## GPU Backend Selection

1. `gpu_backend` - Set this to bypass the automatic backend selection and use a specific
   gpu backend. Valid options are "cuda", "rocm", "metal", and "oneapi". This preference
   needs to be set for `MLDataDevices` package. It is recommended to use
   [`MLDataDevices.gpu_backend!`](@ref) to set this preference.

## [Automatic Eltype Conversion](@id automatic-eltypes-preference)

1. `eltype_mismatch_handling` - Preference controlling what happens when layers get
   different eltypes as input. See the documentation on [`match_eltype`](@ref) for more
   details.

## [Dispatch Doctor](@id dispatch-doctor-preference)

1. `instability_check` - Preference controlling the dispatch doctor. See the documentation
   on [`Lux.set_dispatch_doctor_preferences!`](@ref) for more details. The preferences need
   to be set for `LuxCore` and `LuxLib` packages. Both of them default to `disable`.
   - Setting the `LuxCore` preference sets the check at the level of `LuxCore.apply`. This
     essentially activates the dispatch doctor for all Lux layers.
   - Setting the `LuxLib` preference sets the check at the level of functional layer of
     Lux, for example, [`fused_dense_bias_activation`](@ref). These functions are supposed
     to be type stable for common input types and can be used to guarantee type stability.

## [Disabling Loop Vectorization / Octavian](@id disable_loop_vectorization)

`LoopVectorization.jl` and `Octavian.jl` are optional dependencies that are used to
accelerate certain CPU operations. However, these packages are tightly coupled with julia
and might not work with all julia versions and systems. If these packages are loaded in any
form LuxLib will use the optimized versions of the functions. But it might be desirable to
disable these packages and use the default implementations instead. This can be done by
setting the `disable_loop_vectorization` preference to `true` for `LuxLib`.
