@doc doc"""
    vector_jacobian_product(f, backend::AbstractADType, x, u)

Compute the Vector Jacobian Product ``\left(\frac{\partial f}{\partial x}\right)^T u``.
This is a wrapper around AD backends but allows us to compute gradients of vector-jacobian
products efficiently using mixed-mode AD.

The following backends are supported:

  - `AutoZygote`: `Zygote.jl` must be loaded.

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

Compute the Vector Jacobian Product ``\left(\frac{\partial f}{\partial x}\right) u``.
This is a wrapper around AD backends but allows us to compute gradients of jacobian-vector
products efficiently using mixed-mode AD.

The following packages must be loaded for this function to work:

  - `AutoForwardDiff`: `ForwardDiff.jl` must be loaded.

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
