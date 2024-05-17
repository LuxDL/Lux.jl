# These are not exported but considered part of the public API
"""
    fast_neg_exp_sq(x) = fast_neg_exp_sq(Base.FastMath.exp_fast, x)
    fast_neg_exp_sq(f::F, x) where {F}

Computes `exp.(-x.^2)` using the exp function `f`.

## Arguments

 - `f` must be an exponential function, providing other functions will lead to incorrect
   results. Defaults to `Base.`
"""
@inline fast_neg_exp_sq(x::AbstractArray) = fast_neg_exp_sq(Base.FastMath.exp_fast, x)
@inline function fast_neg_exp_sq(f::F, x::AbstractArray) where {F}
    hasmethod(f, Tuple{typeof(x)}) && return f(@. -x .^ 2)
    return @. f.(-x ^ 2)
end

function CRC.rrule(::typeof(fast_neg_exp_sq), f::F, x::AbstractArray) where {F}
    y = fast_neg_exp_sq(f, x)
    ∇fast_neg_fast_sq = @closure Δ -> (NoTangent(), NoTangent(), @fastmath(@.(-2*x*y*Δ)))
    return y, ∇fast_neg_fast_sq
end

"""
    bspline(x, grid, ::Val{order}, ::Val{extend} = Val(true)) where {order, extend}

Evaluates a B-spline basis function at `x` with `grid`.

# Arguments

  - `x`: A matrix of size `(N, B)` where `N` is the number of splines and `B` is the batch
    size.
  - `grid`: A matrix of size `(N, Grid size)`
  - `order`: The order of the B-spline basis function. Defaults to `3`. (Note that the
    order is a `Val` type.)
  - `extend`: Whether to extend the grid. Defaults to `true`. (Note that the extend is a
    `Val` type.)

# Returns

A matrix of size `(N, C, B)` where `N` is the number of splines and `B` is the batch size,
and `C` is the number of coefficients.
"""
@views @generated function bspline(x::AbstractMatrix{<:Real}, grid::AbstractMatrix{<:Real},
        ::Val{order}, ::Val{extend}=Val(true)) where {order, extend}
    grid_expr = extend ? :(grid = __extend_grid(grid, $(Val(order)))) : :()
    return quote
        $(grid_expr)
        return _bspline(reshape(x, size(x, 1), 1, size(x, 2)),
            reshape(grid, size(grid, 1), size(grid, 2), 1), $(Val(order)))
    end
end

@views @generated function __extend_grid(grid::AbstractMatrix, ::Val{k}=Val(3)) where {k}
    k == 0 && return :(grid)
    calls = [:(grid = hcat(grid[:, 1:1] .- h, grid, grid[:, end:end] .+ h)) for _ in 1:k]
    return quote
        h = (grid[:, end:end] .- grid[:, 1:1]) ./ $(k)
        $(calls...)
        return grid
    end
end

@views function _bspline(
        x::AbstractArray{<:Real, 3}, grid::AbstractArray{<:Real, 3}, ::Val{0})
    return (x .≥ grid[:, 1:(end - 1), :]) .* (x .< grid[:, 2:end, :])
end

CRC.@non_differentiable _bspline(
    ::AbstractArray{<:Real, 3}, ::AbstractArray{<:Real, 3}, ::Val{0})

@views function _bspline(x::AbstractArray{<:Real, 3},
        grid::AbstractArray{<:Real, 3}, ::Val{order}) where {order}
    y = _bspline(x, grid, Val(order - 1))

    return @. (x - grid[:, 1:(end - order - 1), :]) /
              (grid[:, (order + 1):(end - 1), :] - grid[:, 1:(end - order - 1), :]) *
              y[:, 1:(end - 1), :] +
              (grid[:, (order + 2):end, :] - x) /
              (grid[:, (order + 2):end, :] - grid[:, 2:(end - order), :]) * y[:, 2:end, :]
end

@views function CRC.rrule(::typeof(_bspline), x::AbstractArray{<:Real, 3},
        grid::AbstractArray{<:Real, 3}, ::Val{order}) where {order}
    y = _bspline(x, grid, Val(order - 1))

    y₁ = y[:, 1:(end - 1), :]
    y₂ = y[:, 2:end, :]

    m₁ = @. y₁ / (grid[:, (order + 1):(end - 1), :] - grid[:, 1:(end - order - 1), :])
    m₂ = @. y₂ / (grid[:, (order + 2):end, :] - grid[:, 2:(end - order), :])

    res = @. (x - grid[:, 1:(end - order - 1), :]) * m₁ +
             (grid[:, (order + 2):end, :] - x) * m₂

    ∇bspline = let m₁ = m₁, m₂ = m₂, x = x
        Δ -> begin
            (Δ isa CRC.NoTangent || Δ isa CRC.ZeroTangent) &&
                return ntuple(Returns(NoTangent()), 4)
            Δ_ = CRC.unthunk(Δ)
            if ArrayInterface.fast_scalar_indexing(x)  # Proxy check for GPUs
                ∂x = @. order * (m₁[:, 1, :] - m₂[:, 1, :]) * Δ_[:, 1, :]
                @inbounds for i in eachindex(axes(m₁, 2))[2:end]
                    @. ∂x += order * (m₁[:, i, :] - m₂[:, i, :]) * Δ_[:, i, :]
                end
            else
                ∂B = @. order * (m₁ - m₂) * Δ_
                ∂x = similar(x)
                sum!(∂x, ∂B)
            end
            return NoTangent(), ∂x, NoTangent(), NoTangent()
        end
    end

    return res, ∇bspline
end
