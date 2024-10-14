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
function vector_jacobian_product end

function vector_jacobian_product(::F, backend::AbstractADType, _, __) where {F}
    throw(ArgumentError("`vector_jacobian_product` is not implemented for `$(backend)`."))
end

function vector_jacobian_product(f::F, backend::AutoZygote, x, u) where {F}
    assert_backend_loaded(:vector_jacobian_product, backend)
    return AutoDiffInternalImpl.vector_jacobian_product(f, backend, x, u)
end

@doc doc"""
    jacobian_vector_product(f, backend::AbstractADType, x, u)

Compute the Jacobian-Vector Product ``\left(\frac{\partial f}{\partial x}\right) u``.
This is a wrapper around AD backends but allows us to compute gradients of jacobian-vector
products efficiently using mixed-mode AD.

## Backends & AD Packages

| Supported Backends | Packages Needed  |
| :----------------- | :--------------- |
| `AutoForwardDiff`  |                  |

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
function jacobian_vector_product end

function jacobian_vector_product(::F, backend::AbstractADType, _, __) where {F}
    throw(ArgumentError("`jacobian_vector_product` is not implemented for `$(backend)`."))
end

function jacobian_vector_product(f::F, backend::AutoForwardDiff, x, u) where {F}
    return AutoDiffInternalImpl.jacobian_vector_product(f, backend, x, u)
end

"""
    batched_jacobian(f, backend::AbstractADType, x::AbstractArray)

Computes the Jacobian of a function `f` with respect to a batch of inputs `x`. This expects
the following properties for `y = f(x)`:

 1. `ndims(y) ≥ 2`
 2. `size(y, ndims(y)) == size(x, ndims(x))`

## Backends & AD Packages

| Supported Backends | Packages Needed | Note                                           |
|:------------------ |:--------------- |:---------------------------------------------- |
| `AutoForwardDiff`  |                 |                                                |
| `AutoZygote`       | `Zygote.jl`     |                                                |
| `AutoEnzyme`       | `Enzyme.jl`     | Not compatible with ChainRules based Nested AD |

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
function batched_jacobian end

function batched_jacobian(::F, backend::AbstractADType, x::AbstractArray) where {F}
    throw(ArgumentError("`batched_jacobian` is not implemented for `$(backend)`."))
end

for implemented_backend in (AutoForwardDiff, AutoZygote, AutoEnzyme)
    @eval function batched_jacobian(
            f::F, backend::$(implemented_backend), x::AbstractArray) where {F}
        assert_backend_loaded(:batched_jacobian, backend)
        return AutoDiffInternalImpl.batched_jacobian(f, backend, x)
    end
end

function assert_backend_loaded(fname::Symbol, ad::AbstractADType)
    return assert_backend_loaded(fname, ad, adtype_to_backend(ad))
end
function assert_backend_loaded(fname::Symbol, ad::AbstractADType, backend::Val{B}) where {B}
    if !is_extension_loaded(backend)
        error("$(fname) with `$(ad)` requires $(B).jl to be loaded.")
    end
    return
end

adtype_to_backend(::AutoEnzyme) = Val(:Enzyme)
adtype_to_backend(::AutoForwardDiff) = Val(:ForwardDiff)
adtype_to_backend(::AutoZygote) = Val(:Zygote)
