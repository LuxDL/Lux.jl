# Kolmogorov Arnold Networks using Radial Basis Functions
@concrete struct KANDenseRBF{use_base_update, T} <:
                 AbstractExplicitContainerLayer{(:normalizer, :spline_linear, :base_linear)}
    in_dims::Int
    out_dims::Int
    num_grids::Int

    normalizer
    spline_linear
    base_linear
    activation

    # rbf specific
    exp_fn
    grid_min::T
    grid_max::T
    inv_denom
end

@inline function KANDenseRBF(mapping::Pair{<:Int, <:Int}, args...; kwargs...)
    return KANDenseRBF(mapping[1], mapping[2], args...; kwargs...)
end

@inline function KANDenseRBF(
        in_dims::Int, out_dims::Int, num_grids::Int=5, activation::A=swish;
        normalizer::Union{AbstractExplicitLayer, Function}=LayerNorm((in_dims,)),
        grid_range=(-1.0f0, 1.0f0), denominator=nothing,
        spline_weight_init_scale=1.0f0, use_base_update::Union{Bool, Val}=Val(true),
        allow_fast_activation::Bool=true, exp_fn=Base.FastMath.exp_fast) where {A}
    if !(normalizer isa AbstractExplicitLayer)
        normalizer = WrappedFunction(Base.Fix1(
            broadcast, allow_fast_activation ? NNlib.fast_act(normalizer) : normalizer))
    end
    T = promote_type(eltype(grid_range[1]), eltype(grid_range[2]))
    denominator = denominator === nothing ?
                  T((grid_range[2] - grid_range[1]) / (num_grids - 1)) : denominator
    spline_linear = Dense(in_dims * num_grids => out_dims; use_bias=false,
        init_weight=truncated_normal(; std=spline_weight_init_scale))
    base_linear = __unwrap_val(use_base_update) ?
                  Dense(in_dims => out_dims; use_bias=false) : NoOpLayer()
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
    return KANDenseRBF{__unwrap_val(use_base_update)}(
        in_dims, out_dims, num_grids, normalizer, spline_linear, base_linear,
        activation, exp_fn, grid_range[1], grid_range[2], denominator)
end

# TODO: parameterlength and statelength

function initialstates(rng::AbstractRNG, kan::KANDenseRBF)
    return (; grid=collect(LinRange(kan.grid_min, kan.grid_max, kan.num_grids)),
        normalizer=initialstates(rng, kan.normalizer),
        spline_linear=initialstates(rng, kan.spline_linear),
        base_linear=initialstates(rng, kan.base_linear))
end

@generated function (kan::KANDenseRBF{use_base_update})(
        x::AbstractMatrix, ps, st::NamedTuple) where {use_base_update}
    base_update_expr = if use_base_update
        quote
            y5 = kan.activation.(x)
            y6, st4 = kan.base_linear(y5, ps.base_linear, st.base_linear)
            res = y6 .+ y4
            st_updated = (; normalizer=st1, spline_linear=st3, base_linear=st4)
        end
    else
        quote
            res = y4
            st_updated = (; normalizer=st1, spline_linear=st3)
        end
    end
    return quote
        y1, st1 = Lux.apply(kan.normalizer, x, ps.normalizer, st.normalizer)
        y2_rbf = reshape(y1, 1, size(y1)...)
        grid = CRC.ignore_derivatives(st.grid)
        y2 = (y2_rbf .- grid) * kan.inv_denom # Don't merge the broadcasting with the next line
        y2 = fast_neg_exp_sq(kan.exp_fn, y2)
        y3 = reshape(y2, kan.in_dims * kan.num_grids, :)
        y4, st3 = Lux.apply(kan.spline_linear, y3, ps.spline_linear, st.spline_linear)
        $(base_update_expr)
        return res, merge(st, st_updated)
    end
end

# @concrete struct KANDenseBSpline{order, T} <:
#                  AbstractExplicitContainerLayer{(:base_linear,)}
#     in_dims::Int
#     out_dims::Int
#     num_grids::Int
#     activation

#     base_linear

#     init_noise
#     scale_noise

#     grid_min::T
#     grid_max::T
# end

# @inline function KANDenseBSpline(mapping::Pair{<:Int, <:Int}, args...; kwargs...)
#     return KANDenseBSpline(mapping[1], mapping[2], args...; kwargs...)
# end

# @inline function KANDenseBSpline(
#         in_dims::Int, out_dims::Int, num_grids::Int=5, activation=swish;
#         spline_order::Union{Int, Val}=Val(3), scale_noise=0.1f0, init_noise=randn32,
#         scale_base=1.0f0, grid_range=(-1.0f0, 1.0f0), allow_fast_activation::Bool=true)
#     activation = allow_fast_activation ? NNlib.fast_act(activation) : activation

#     base_linear = Dense(in_dims => out_dims; use_bias=false,
#         init_weight=kaiming_uniform(; gain=oftype(scale_base, sqrt(5)) * scale_base))

#     T = promote_type(eltype(grid_range[1]), eltype(grid_range[2]))

#     return KANDenseBSpline{__unwrap_val(spline_order)}(
#         in_dims, out_dims, num_grids, activation, base_linear,
#         init_noise, scale_noise, grid_range[1], grid_range[2])
# end

# # TODO: parameterlength and statelength

# function initialparameters(
#         rng::AbstractRNG, kan::KANDenseBSpline{order, T}) where {order, T}
#     grid = ones(T, kan.in_dims * kan.out_dims) .*
#            LinRange(kan.grid_min, kan.grid_max, kan.num_grids)'
#     noise = (kan.init_noise(rng, kan.in_dims * kan.out_dims, size(grid, 2)) .- (1 // 2)) .*
#             kan.scale_noise ./ kan.num_grids
#     spline_weight = __curve_to_coefficient(grid, noise, grid, Val(order))
#     return (; spline_weight, base_linear=initialparameters(rng, kan.base_linear))
# end

# function initialstates(rng::AbstractRNG, kan::KANDenseBSpline{order, T}) where {order, T}
#     return (; base_linear=initialstates(rng, kan.base_linear),
#         grid=ones(T, kan.in_dims * kan.out_dims) .*
#              LinRange(kan.grid_min, kan.grid_max, kan.num_grids + 1)')
# end

# function (kan::KANDenseBSpline{order})(x::AbstractMatrix, ps, st::NamedTuple) where {order}
#     base_output, st1 = kan.base_linear(kan.activation.(x), ps.base_linear, st.base_linear)
#     @show size(x), size(st.grid)
#     y = bspline(x, st.grid, Val(order))
#     spline_output = dropdims(
#         sum(y .* reshape(ps.spline_weight, size(ps.spline_weight)..., 1); dims=2); dims=2)
#     @show size(spline_output), size(base_output)
#     res = base_output .+ spline_output
#     return res, merge(st, st1)
# end

# function __curve_to_coefficient(x::AbstractMatrix, y::AbstractMatrix,
#         grid::AbstractMatrix, O::Val{order}) where {order}
#     A = __maybe_lazy_permutedims(bspline(x, grid, O), Val((1, 3, 2)))
#     return __batched_least_squares(A, y)
# end

# # NOTE: The batching here is over the first dimension of A and b
# function __batched_least_squares(A::AbstractArray{T, 3}, b::AbstractMatrix) where {T}
#     return mapreduce(__expanddims1 ∘ \, vcat, eachslice(A; dims=1), eachslice(b; dims=1))
# end
