@concrete struct KANLayer{spline_trainable, base_trainable, order} <: AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    num_grid_intervals::Int
    activation
    scale_noise
    scale_base
    scale_spline
    grid_eps
    grid_range
end

function KANLayer(mapping::Pair{<:Int, <:Int}, num_grid_intervals::Int=5,
        activation=silu; spline_order::Union{Int, Val}=Val(3),
        scale_noise=0.1f0, scale_base=1.0f0, scale_spline=1.0f0, grid_eps=1.0f-2,
        grid_range=(-1.0f0, 1.0f0), spline_trainable::Union{Bool, Val}=Val(true),
        base_trainable::Union{Bool, Val}=Val(true), allow_fast_activation::Bool=true)
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
    in_dims, out_dims = mapping
    return KANLayer{__unwrap_val(spline_trainable),
        __unwrap_val(base_trainable), __unwrap_val(spline_order)}(
        in_dims, out_dims, num_grid_intervals, activation,
        scale_noise, scale_base, scale_spline, grid_eps, grid_range)
end

function initialparameters(
        rng::AbstractRNG, kan::KANLayer{spT, bT, order}) where {spT, bT, order}
    grid = __generate_kan_grid(kan)  # unfortunately we need to generate the grid here as well
    noises = (randn(
        rng, eltype(kan.scale_noise), kan.in_dims * kan.out_dims, size(grid, 2)) .-
              (1 // 2)) .* kan.scale_noise ./ kan.num_grid_intervals
    coefficients = __curve_to_coefficient(grid, noises, grid, Val(order))
    ps = (; coefficients)
    spT && (ps = merge(ps, (; scale_spline=__generate_scale_arr(kan, kan.scale_spline))))
    bT && (ps = merge(ps, (; scale_base=__generate_scale_arr(kan, kan.scale_base))))
    return ps
end

function initialstates(::AbstractRNG, kan::KANLayer{spT, bT, order}) where {spT, bT, order}
    grid = __generate_kan_grid(kan)
    mask = ones(eltype(grid), (kan.in_dims, kan.out_dims))
    st = (; grid, mask, weight_sharing=Colon(), lock_counter=0,
        lock_id=zeros(Int, kan.in_dims, kan.out_dims))
    !spT && (st = merge(st, (; scale_spline=__generate_scale_arr(kan, kan.scale_spline))))
    !bT && (st = merge(st, (; scale_base=__generate_scale_arr(kan, kan.scale_base))))
    return st
end

# @generated Needed else Zygote is type unstable
@views @generated function (kan::KANLayer{spT, bT, order})(
        x::AbstractMatrix, ps, st::NamedTuple) where {spT, bT, order}
    scale_expr = spT ? :(ps.scale_spline) : :(st.scale_spline)
    base_expr = bT ? :(ps.scale_base) : :(st.scale_base)
    return quote
        B = size(x, 2)
        scale_spline = $(scale_expr)
        scale_base = $(base_expr)
        x_ = reshape(x, :, 1, B)
        preacts = x_ .* __ones_like(x, (1, kan.out_dims, 1))                  # I x O x B
        y = __coefficient_to_curve(reshape(preacts, :, B), st.grid[st.weight_sharing, :],
            ps.coefficients[st.weight_sharing, :], $(Val(order)))             # (I x O) x B
        postspline = reshape(y, kan.in_dims, kan.out_dims, B)
        mask = CRC.ignore_derivatives(st.mask)
        postacts = @. (scale_base * kan.activation(x_) + scale_spline * postspline) * mask
        res = dropdims(sum(postacts; dims=1); dims=1)                         # O x B
        return (res, preacts, postacts, postspline), st
    end
end

@inline function __generate_scale_arr(kan::KANLayer, scale_spline::T) where {T <: Number}
    return __generate_scale_arr((kan.in_dims, kan.out_dims), scale_spline)
end
@inline function __generate_scale_arr(size, scale_spline::T) where {T <: Number}
    return ones(T, size) .* scale_spline
end
@inline function __generate_scale_arr(
        _size, scale_spline::AbstractArray{T}) where {T <: Number}
    @assert prod(_size) == length(scale_spline)
    x = similar(scale_spline, _size)
    copyto!(x, scale_spline)
    return x
end

@inline function __generate_kan_grid(kan::KANLayer)
    return ones(Float32, kan.in_dims * kan.out_dims) *
           LinRange(kan.grid_range[1], kan.grid_range[2], kan.num_grid_intervals + 1)'
end

# Exported KANUtils
module KANUtils

using ..Lux: Lux

function update_grid_from_samples end
function initialize_grid_from_parent end
function get_subset end
function lock_activations end
function unlock_activations end

end

# utilities for splines
## TODO: Upstream to NNlib
@inline silu(x) = x * sigmoid(x)  # Not present in NNlib
@inline silu_fast(x) = x * sigmoid_fast(x)

@inline NNlib.fast_act(::typeof(silu), ::AbstractArray=1:0) = silu_fast

@views @generated function __extend_grid(grid::AbstractMatrix, ::Val{k}=Val(3)) where {k}
    k == 0 && return :(grid)
    calls = [:(grid = hcat(grid[:, 1:1] .- h, grid, grid[:, end:end] .+ h)) for _ in 1:k]
    return quote
        h = (grid[:, end:end] .- grid[:, 1:1]) ./ $(k)
        $(calls...)
        return grid
    end
end

# x       --> N splines x B
# grid    --> N splines x Grid size
# Outputs --> N splines x N coeffs x B
@views @generated function __bspline_evaluate(
        x::AbstractMatrix, grid::AbstractMatrix, ::Val{order}=Val(3),
        ::Val{extend}=Val(true)) where {order, extend}
    grid_expr = extend ? :(grid = __extend_grid(grid, $(Val(order)))) : :()
    return quote
        $(grid_expr)
        return __bspline_evaluate(reshape(x, size(x, 1), 1, size(x, 2)),
            reshape(grid, size(grid, 1), size(grid, 2), 1), $(Val(order)))
    end
end

@views function __bspline_evaluate(
        x::AbstractArray{T1, 3}, grid::AbstractArray{T2, 3}, ::Val{0}) where {T1, T2}
    return (x .≥ grid[:, 1:(end - 1), :]) .* (x .< grid[:, 2:end, :])
end

CRC.@non_differentiable __bspline_evaluate(
    ::AbstractArray{<:Any, 3}, ::AbstractArray{<:Any, 3}, ::Val{0})

@views function __bspline_evaluate(x::AbstractArray{T1, 3}, grid::AbstractArray{T2, 3},
        ::Val{order}) where {T1, T2, order}
    y = __bspline_evaluate(x, grid, Val(order - 1))

    return @. (x - grid[:, 1:(end - order - 1), :]) /
              (grid[:, (order + 1):(end - 1), :] - grid[:, 1:(end - order - 1), :]) *
              y[:, 1:(end - 1), :] +
              (grid[:, (order + 2):end, :] - x) /
              (grid[:, (order + 2):end, :] - grid[:, 2:(end - order), :]) * y[:, 2:end, :]
end

# grid should be fixed, so we don't compute the gradient wrt the grid
@views function CRC.rrule(::typeof(__bspline_evaluate), x::AbstractArray{T1, 3},
        grid::AbstractArray{T2, 3}, ::Val{order}) where {T1, T2, order}
    y = __bspline_evaluate(x, grid, Val(order - 1))

    y₁ = y[:, 1:(end - 1), :]
    y₂ = y[:, 2:end, :]

    m₁ = @. y₁ / (grid[:, (order + 1):(end - 1), :] - grid[:, 1:(end - order - 1), :])
    m₂ = @. y₂ / (grid[:, (order + 2):end, :] - grid[:, 2:(end - order), :])

    res = @. (x - grid[:, 1:(end - order - 1), :]) * m₁ +
             (grid[:, (order + 2):end, :] - x) * m₂

    ∇bspline_evaluate = let m₁ = m₁, m₂ = m₂, x = x
        Δ -> begin
            (Δ isa CRC.NoTangent || Δ isa CRC.ZeroTangent) &&
                return ntuple(Returns(NoTangent()), 4)
            Δ_ = CRC.unthunk(Δ)
            ∂B = @. order * (m₁ - m₂) * Δ_
            ∂x = similar(x)
            sum!(∂x, ∂B)
            return NoTangent(), ∂x, NoTangent(), NoTangent()
        end
    end

    return res, ∇bspline_evaluate
end

# x       --> N splines x B
# grid    --> N splines x Grid size
# coef    --> N splines x N coeffs
# Outputs --> N splines x B
function __coefficient_to_curve(x::AbstractMatrix, grid::AbstractMatrix,
        coef::AbstractMatrix, O::Val{order}) where {order}
    return dropdims(
        sum(
            __bspline_evaluate(x, grid, O) .*
            reshape(coef, size(coef, 1), size(coef, 2), 1);
            dims=2);
        dims=2)
end

# x       --> N splines x B
# y       --> N splines x B
# grid    --> N splines x Grid size
# Outputs --> N splines x N coeffs
function __curve_to_coefficient(x::AbstractMatrix, y::AbstractMatrix,
        grid::AbstractMatrix, O::Val{order}) where {order}
    # For GPU Arrays avoid using lazy wrappers, we use specialized LAPACK routines there
    A = __maybe_lazy_permutedims(__bspline_evaluate(x, grid, O), Val((1, 3, 2)))
    return __batched_least_squares(A, y)
end

# TODO: GPUs typically ship faster routines for batched least squares.
# NOTE: The batching here is over the first dimension of A and b
function __batched_least_squares(A::AbstractArray{T, 3}, b::AbstractMatrix) where {T}
    return mapreduce(__expanddims1 ∘ \, vcat, eachslice(A; dims=1), eachslice(b; dims=1))
end
