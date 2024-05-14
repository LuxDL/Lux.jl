@concrete struct KANDense{mode} <: AbstractExplicitContainerLayer{(:model,)}
    model
end

@inline KANDense(mode::Symbol, args...; kwargs...) = KANDense{mode}(args...; kwargs...)

@inline function KANDense{:BSpline}(args...; kwargs...)
    error("BSpline is not yet implemented")
end

@inline function KANDense{:RBF}(
        in_dims::Int, out_dims::Int, num_grids::Int=5, activation::A=swish;
        normalizer::Union{AbstractExplicitLayer, Function}=LayerNorm((in_dims,)),
        grid_range=(-1.0f0, 1.0f0), denominator=nothing, spline_weight_init_scale=1.0f0,
        use_base_update::Bool=true, allow_fast_activation::Bool=true, kwargs...) where {A}
    if !(normalizer isa AbstractExplicitLayer)
        normalizer = Base.Fix1(
            broadcast, allow_fast_activation ? NNlib.fast_act(normalizer) : normalizer)
    end
    rbf = RadialBasisFunction(;
        grid_min=first(grid_range), grid_max=last(grid_range), num_grids, denominator)
    spline_linear = Dense(in_dims * num_grids, out_dims; use_bias=false,
        init_weight=truncated_normal(; std=spline_weight_init_scale))
    main_model = @compact(; normalizer, rbf, in_dims, num_grids, spline_linear) do x
        @return spline_linear(reshape(rbf(normalizer(x)), in_dims * num_grids, :))
    end
    !use_base_update && return KANDense{:RBF}(main_model)
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
    final_model = @compact(; main_model, activation,
        base_linear=Dense(in_dims, out_dims; use_bias=false)) do x
        @return main_model(x) .+ base_linear(activation.(x))
    end
    return KANDense{:RBF}(final_model)
end

@inline (kan::KANDense{:RBF})(x::AbstractMatrix, ps, st::NamedTuple) = Lux.apply(
    kan.model, x, ps, st)

# Basis Functions
@concrete struct RadialBasisFunction{T} <: AbstractExplicitLayer
    grid_min::T
    grid_max::T
    num_grids::Int
    denominator
end

function RadialBasisFunction(;
        grid_min=-1.0f0, grid_max=1.0f0, num_grids::Int=5, denominator=nothing)
    T = promote_type(typeof(grid_min), typeof(grid_max))
    denominator = denominator === nothing ? T((grid_max - grid_min) / (num_grids - 1)) :
                  denominator
    return RadialBasisFunction(grid_min, grid_max, num_grids, denominator)
end

function initialstates(::AbstractRNG, rbf::RadialBasisFunction{T}) where {T}
    return (; grid=collect(LinRange(rbf.grid_min, rbf.grid_max, rbf.num_grids)))
end

function (rbf::RadialBasisFunction)(x::AbstractArray, ps, st::NamedTuple)
    y = reshape(x, 1, size(x)...)
    grid = CRC.ignore_derivatives(st.grid)
    return @.(__fast_exp_sq((y - grid) / rbf.denominator)), st
end

@inline __fast_exp_sq(x::Number) = @fastmath(exp(-x^2))

function CRC.rrule(::typeof(Broadcast.broadcasted), ::typeof(__fast_exp_sq),
        x::Union{Broadcast.Broadcasted, AbstractArray{<:Number}, Number})
    y = __fast_exp_sq.(x)
    ∇fast_exp_sq = let y = y
        Δ -> (NoTangent(), NoTangent(), @fastmath(@.(-2*x*y*Δ)))
    end
    return y, ∇fast_exp_sq
end

# @concrete struct KANDense{spline_trainable, base_trainable, order} <: AbstractExplicitLayer
#     in_dims::Int
#     out_dims::Int
#     num_grid_intervals::Int
#     activation

#     scale_noise
#     init_noise
#     init_scale_base
#     init_scale_spline

#     grid_range
# end

# @inline function KANDense(mapping::Pair{<:Int, <:Int}, args...; kwargs...)
#     return KANDense(mapping[1], mapping[2], args...; kwargs...)
# end

# function KANDense(in_dims, out_dims, grid_size=5, activation=swish;
#         spline_order::Union{Int, Val}=Val(3), scale_noise=0.1f0,
#         init_noise=randn32, init_scale_base=__scale_base_init,
#         init_scale_spline=ones32, grid_range=(-1.0f0, 1.0f0),
#         grid_eps=0.02f0, allow_fast_activation::Bool=true)
#     return
# end

# function KANDense(mapping::Pair{<:Int, <:Int}, activation=swish; num_grid_intervals::Int=5,
#         spline_order::Union{Int, Val}=Val(3), scale_noise=0.1f0, init_noise=randn32,
#         init_scale_base=__scale_base_init, init_scale_spline=ones32,
#         grid_range=(-1.0f0, 1.0f0), spline_trainable::Union{Bool, Val}=Val(true),
#         base_trainable::Union{Bool, Val}=Val(true), allow_fast_activation::Bool=true)
#     activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
#     in_dims, out_dims = mapping
#     return KANDense{__unwrap_val(spline_trainable),
#         __unwrap_val(base_trainable), __unwrap_val(spline_order)}(
#         in_dims, out_dims, num_grid_intervals, activation, scale_noise,
#         init_noise, init_scale_base, init_scale_spline, grid_range)
# end

# function initialparameters(
#         rng::AbstractRNG, kan::KANDense{spT, bT, order}) where {spT, bT, order}
#     grid = __init_kan_grid(
#         kan.in_dims, kan.out_dims, kan.num_grid_intervals, kan.grid_range)
#     noises = (kan.init_noise(rng, kan.in_dims * kan.out_dims, size(grid, 2)) .- (1 // 2)) .*
#              kan.scale_noise ./ kan.num_grid_intervals
#     coefficients = __curve_to_coefficient(grid, noises, grid, Val(order))
#     ps = (; coefficients)
#     spT && (ps = merge(
#         ps, (; scale_spline=kan.init_scale_spline(rng, kan.in_dims, kan.out_dims))))
#     bT &&
#         (ps = merge(ps, (; scale_base=kan.init_scale_base(rng, kan.in_dims, kan.out_dims))))
#     return ps
# end

# function initialstates(::AbstractRNG, kan::KANDense{spT, bT, order}) where {spT, bT, order}
#     grid = __init_kan_grid(
#         kan.in_dims, kan.out_dims, kan.num_grid_intervals, kan.grid_range)
#     mask = ones(eltype(grid), (kan.in_dims, kan.out_dims))
#     st = (; grid, mask, weight_sharing=Colon(), lock_counter=0,
#         lock_id=zeros(Int, kan.in_dims, kan.out_dims))
#     !spT && (st = merge(
#         st, (; scale_spline=kan.init_scale_spline(rng, kan.in_dims, kan.out_dims))))
#     !bT &&
#         (st = merge(st, (; scale_base=kan.init_scale_base(rng, kan.in_dims, kan.out_dims))))
#     return st
# end

# function __init_kan_grid(
#         in_dims, out_dims, num_grid_intervals, grid_range::Tuple{T1, T2}) where {T1, T2}
#     T = promote_type(T1, T2)
#     nsplines = in_dims * out_dims
#     return ones(T, nsplines) .*
#            LinRange(grid_range[1], grid_range[2], num_grid_intervals + 1)'
# end

# # @generated needed else Zygote is type unstable
# @views @generated function (kan::KANDense{spT, bT, order})(
#         x::AbstractMatrix, ps, st::NamedTuple) where {spT, bT, order}
#     scale_expr = spT ? :(ps.scale_spline) : :(st.scale_spline)
#     base_expr = bT ? :(ps.scale_base) : :(st.scale_base)
#     return quote
#         B = size(x, 2)
#         scale_spline = $(scale_expr)
#         scale_base = $(base_expr)
#         x_ = reshape(x, :, 1, B)
#         preacts = x_ .* __ones_like(x, (1, kan.out_dims, 1))                  # I x O x B
#         y = __coefficient_to_curve(reshape(preacts, :, B), st.grid[st.weight_sharing, :],
#             ps.coefficients[st.weight_sharing, :], $(Val(order)))             # (I x O) x B
#         postspline = reshape(y, kan.in_dims, kan.out_dims, B)
#         mask = CRC.ignore_derivatives(st.mask)
#         postacts = @. (scale_base * kan.activation(x_) + scale_spline * postspline) * mask
#         res = dropdims(sum(postacts; dims=1); dims=1)                         # O x B
#         return res, st
#     end
# end

# utilities for splines
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

# TODO: Maybe upstream to WeightInitializers.jl
function __scale_base_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        scale_noise_base=0.1) where {T <: Real}
    @assert length(dims)==2 "Expected 2 dimensions for scale_base initialization"
    in_dims, out_dims = dims
    return T(inv(sqrt(in_dims))) .+
           (randn(rng, T, in_dims, out_dims) .* T(2) .- T(1)) .* T(scale_noise_base)
end

@inline function __scale_base_init(dims::Integer...; kwargs...)
    return __scale_base_init(WeightInitializers._default_rng(), Float32, dims...; kwargs...)
end

@inline function __scale_base_init(rng::AbstractRNG; kwargs...)
    return WeightInitializers.__partial_apply(__scale_base_init, (rng, (; kwargs...)))
end

@inline function __scale_base_init(rng::AbstractRNG, ::Type{T}; kwargs...) where {T <: Real}
    return WeightInitializers.__partial_apply(__scale_base_init, ((rng, T), (; kwargs...)))
end

@inline function __scale_base_init(rng::AbstractRNG, dims::Integer...; kwargs...)
    return __scale_base_init(rng, Float32, dims...; kwargs...)
end

@inline function __scale_base_init(; kwargs...)
    return WeightInitializers.__partial_apply(__scale_base_init, (; kwargs...))
end
