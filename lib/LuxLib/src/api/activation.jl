"""
    fast_activation!!(σ::F, x::AbstractArray) where {F}

Compute `σ.(x)` with the best possible implementation available. If it is possible to
rewrite `x` in-place, it does so. If `x` is an immutable array, it falls back to the
generic implementation.

!!! note

    This function doesn't replace `σ` with `NNlib.fast_act(σ, ...)`, that needs to be
    done by the user if needed.

!!! tip

    Certain activation functions are replaced with specialized implementations from
    [SLEEFPirates.jl](https://github.com/JuliaSIMD/SLEEFPirates.jl). This might lead to
    faster performance but can cause slight decrease in accuracy (in the floating point
    limit).

## Arguments

  - `σ`: Activation function
  - `x`: Input array

## Returns

  - Output Array with the same size as `x`
"""
function fast_activation!!(σ::F, x::AbstractArray) where {F}
    return _fast_activation!!(__is_immutable_array_or_dual_val((x,)), σ, x)
end

function _fast_activation!!(::Val{true}, σ::F, x::AbstractArray) where {F}
    return _fast_activation(σ, x)
end

_fast_activation!!(::Val{false}, σ::F, x::AbstractArray) where {F} = _fast_activation!(σ, x)
