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

1. `DisableAutomaticNestedADSwitching` - Set this to `true` to disable automatic switching
   of backends for nested automatic differentiation. See the manual section on
   [nested automatic differentiation](@ref nested_autodiff) for more details.

## GPU-Aware MPI Support

1. `LuxDistributedMPICUDAAware` - Set this to `true` if your MPI build is CUDA aware.
2. `LuxDistributedMPIROCMAware` - Set this to `true` if your MPI build is ROCM aware.
