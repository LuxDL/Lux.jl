@doc doc"""
    vector_jacobian_product(f, backend::AbstractADType, x, u)

Compute the Vector-Jacobian Product ``\left(\frac{\partial f}{\partial x}\right)^T u``.
This is a wrapper around AD backends but allows us to compute gradients of vector-jacobian
products efficiently using mixed-mode AD.

## Backends & AD Packages

| Supported Backends | Packages Needed |
| :----------------- | :-------------- |
| `AutoZygote`       | `Zygote.jl`     |

!!! warning

    Gradient wrt `u` in the reverse pass is always dropped.

## Arguments

  - `f`: The function to compute the jacobian of.
  - `backend`: The backend to use for computing the VJP.
  - `x`: The input to the function.
  - `u`: An object of the same structure as `f(x)`.

## Returns

  - `v`: The Vector Jacobian Product.
"""
function vector_jacobian_product(f::F, backend::AbstractADType, x, u) where {F}
    @assert backend isa AutoZygote "Only `AutoZygote` is supported for \
                                    `vector_jacobian_product`."
    if !_is_extension_loaded(Val(:Zygote))
        error("`Zygote.jl` must be loaded for `vector_jacobian_product` \
               to work with `$(backend)`.")
    end
    return __vector_jacobian_product_impl(f, backend, x, u)
end

function __vector_jacobian_product_impl end

@doc doc"""
    jacobian_vector_product(f, backend::AbstractADType, x, u)

Compute the Jacobian-Vector Product ``\left(\frac{\partial f}{\partial x}\right) u``.
This is a wrapper around AD backends but allows us to compute gradients of jacobian-vector
products efficiently using mixed-mode AD.

## Backends & AD Packages

| Supported Backends | Packages Needed  |
| :----------------- | :--------------- |
| `AutoForwardDiff`  | `ForwardDiff.jl` |

!!! warning

    Gradient wrt `u` in the reverse pass is always dropped.

## Arguments

  - `f`: The function to compute the jacobian of.
  - `backend`: The backend to use for computing the JVP.
  - `x`: The input to the function.
  - `u`: An object of the same structure as `x`.

## Returns

  - `v`: The Jacobian Vector Product.
"""
function jacobian_vector_product(f::F, backend::AbstractADType, x, u) where {F}
    @assert backend isa AutoForwardDiff "Only `AutoForwardDiff` is supported for \
                                        `jacobian_vector_product`."
    if !_is_extension_loaded(Val(:ForwardDiff))
        error("`ForwardDiff.jl` must be loaded for `jacobian_vector_product` \
               to work with `$(backend)`.")
    end
    return __jacobian_vector_product_impl(f, backend, x, u)
end

function __jacobian_vector_product_impl end

"""
    batched_jacobian(f, backend::AbstractADType, x::AbstractArray)

Computes the Jacobian of a function `f` with respect to a batch of inputs `x`. This expects
the following properties for `y = f(x)`:

 1. `ndims(y) ≥ 2`
 2. `size(y, ndims(y)) == size(x, ndims(x))`

## Backends & AD Packages

| Supported Backends | Packages Needed  |
|:------------------ |:---------------- |
| `AutoForwardDiff`  | `ForwardDiff.jl` |

## Arguments

  - `f`: The function to compute the jacobian of.
  - `backend`: The backend to use for computing the jacobian.
  - `x`: The input to the function. Must have `ndims(x) ≥ 2`.

## Returns

  - `J`: The Jacobian of `f` with respect to `x`. This will be a 3D Array. If the dimensions
    of `x` are `(N₁, N₂, ..., Nₙ, B)` and of `y` are `(M₁, M₂, ..., Mₘ, B)`, then `J` will
    be a `((M₁ × M₂ × ... × Mₘ), (N₁ × N₂ × ... × Nₙ), B)` Array.

!!! danger

    `f(x)` must not be inter-mixing the batch dimensions, else the result will be incorrect.
    For example, if `f` contains operations like batch normalization, then the result will
    be incorrect.
"""
function batched_jacobian(f::F, backend::AbstractADType, x::AbstractArray) where {F}
    ndims(x) ≤ 1 && error("`batched_jacobian` only supports batched inputs (ndims(x) > 1).")
    @assert backend isa AutoForwardDiff "Only `AutoForwardDiff` is currently supported for \
                                        `batched_jacobian`."
    if !_is_extension_loaded(Val(:ForwardDiff))
        error("`ForwardDiff.jl` must be loaded for `batched_jacobian` to work with \
               `$(backend)`.")
    end
    return __batched_jacobian(f, backend, x)
end

function __batched_jacobian end
function __batched_jacobian_impl end
