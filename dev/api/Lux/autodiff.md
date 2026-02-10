---
url: /dev/api/Lux/autodiff.md
---
# Automatic Differentiation Helpers {#autodiff-lux-helpers}

## JVP & VJP Wrappers {#JVP-and-VJP-Wrappers}

```julia
jacobian_vector_product(f, backend::AbstractADType, x, u)
```

Compute the Jacobian-Vector Product $\left(\frac{\partial f}{\partial x}\right) u$. This is a wrapper around AD backends but allows us to compute gradients of jacobian-vector products efficiently using mixed-mode AD.

**Backends & AD Packages**

| Supported Backends | Packages Needed             | Notes                                             |
|:------------------ |:--------------------------- |:------------------------------------------------- |
| `AutoEnzyme`       | `Enzyme.jl` / `Reactant.jl` | For nested AD support directly using `Enzyme.jl`. |
| `AutoForwardDiff`  |                             |                                                   |

::: warning Only for ChainRules-based AD like Zygote

Gradient wrt `u` in the reverse pass is always dropped.

:::

**Arguments**

* `f`: The function to compute the jacobian of.

* `backend`: The backend to use for computing the JVP.

* `x`: The input to the function.

* `u`: An object of the same structure as `x`.

**Returns**

* `v`: The Jacobian Vector Product.

source

```julia
vector_jacobian_product(f, backend::AbstractADType, x, u)
```

Compute the Vector-Jacobian Product $\left(\frac{\partial f}{\partial x}\right)^T u$. This is a wrapper around AD backends but allows us to compute gradients of vector-jacobian products efficiently using mixed-mode AD.

**Backends & AD Packages**

| Supported Backends | Packages Needed             | Notes                                             |
|:------------------ |:--------------------------- |:------------------------------------------------- |
| `AutoEnzyme`       | `Enzyme.jl` / `Reactant.jl` | For nested AD support directly using `Enzyme.jl`. |
| `AutoZygote`       | `Zygote.jl`                 |                                                   |

::: warning Only for ChainRules-based AD like Zygote

Gradient wrt `u` in the reverse pass is always dropped.

:::

**Arguments**

* `f`: The function to compute the jacobian of.

* `backend`: The backend to use for computing the VJP.

* `x`: The input to the function.

* `u`: An object of the same structure as `f(x)`.

**Returns**

* `v`: The Vector Jacobian Product.

source

## Batched AD {#Batched-AD}

```julia
batched_jacobian(f, backend::AbstractADType, x::AbstractArray)
```

Computes the Jacobian of a function `f` with respect to a batch of inputs `x`. This expects the following properties for `y = f(x)`:

1. `ndims(y) ≥ 2`

2. `size(y, ndims(y)) == size(x, ndims(x))`

**Backends & AD Packages**

| Supported Backends | Packages Needed |
|:------------------ |:--------------- |
| `AutoEnzyme`       | `Reactant.jl`   |
| `AutoForwardDiff`  |                 |
| `AutoZygote`       | `Zygote.jl`     |

**Arguments**

* `f`: The function to compute the jacobian of.

* `backend`: The backend to use for computing the jacobian.

* `x`: The input to the function. Must have `ndims(x) ≥ 2`.

**Returns**

* `J`: The Jacobian of `f` with respect to `x`. This will be a 3D Array. If the dimensions of `x` are `(N₁, N₂, ..., Nₙ, B)` and of `y` are `(M₁, M₂, ..., Mₘ, B)`, then `J` will be a `((M₁ × M₂ × ... × Mₘ), (N₁ × N₂ × ... × Nₙ), B)` Array.

::: danger Danger

`f(x)` must not be inter-mixing the batch dimensions, else the result will be incorrect. For example, if `f` contains operations like batch normalization, then the result will be incorrect.

:::

source

## Nested 2nd Order AD {#Nested-2nd-Order-AD}

Consult the [manual page on Nested AD](/manual/nested_autodiff#nested_autodiff) for information on nested automatic differentiation.
