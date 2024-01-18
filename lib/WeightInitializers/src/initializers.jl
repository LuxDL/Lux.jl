for T in ("16", "32", "64", "C16", "C32", "C64"), fname in (:ones, :zeros, :rand, :randn)
    name = Symbol(fname, T)
    docstring = __generic_docstring(string(name))
    TP = NUM_TO_FPOINT[Symbol(T)]
    if fname in (:ones, :zeros)
        @eval begin
            @doc $docstring
            function $(name)(rng::AbstractRNG, dims::Integer...; kwargs...)
                return $(fname)($TP, dims...; kwargs...)
            end
        end
    else
        @eval begin
            @doc $docstring
            function $(name)(rng::AbstractRNG, dims::Integer...; kwargs...)
                return $(fname)(rng, $TP, dims...; kwargs...)
            end
        end
    end
end

"""
    glorot_uniform([::AbstractRNG=_default_rng()], [T=Float32], size...;
        gain = 1) -> AbstractArray{T, length(size)}

Return an `AbstractArray{T}` of the given `size` containing random numbers drawn from a
uniform distribution on the interval ``[-x, x]``, where
`x = gain * sqrt(6 / (fan_in + fan_out))`. This method is described in [1] and also known as
Xavier initialization.

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep
feedforward neural networks." _Proceedings of the thirteenth international conference on
artificial intelligence and statistics_. 2010.
"""
function glorot_uniform(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        gain::Number=1) where {T <: Number}
    scale = T(gain) * sqrt(T(24) / sum(_nfan(dims...)))
    return (rand(rng, T, dims...) .- T(1 // 2)) .* scale
end

"""
    glorot_normal([::AbstractRNG=_default_rng()], [T=Float32], size...;
        gain = 1) -> AbstractArray{T, length(size)}

Return an `AbstractArray{T}` of the given `size` containing random numbers drawn from a
normal distribution with standard deviation `gain * sqrt(2 / (fan_in + fan_out))`. This
method is described in [1] and also known as Xavier initialization.

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep
feedforward neural networks." _Proceedings of the thirteenth international conference on
artificial intelligence and statistics_. 2010.
"""
function glorot_normal(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        gain::Number=1) where {T <: Number}
    std = T(gain) * sqrt(T(2) / sum(_nfan(dims...)))
    return randn(rng, T, dims...) .* std
end

"""
    kaiming_uniform([::AbstractRNG=_default_rng()], [T=Float32], size...;
        gain = √T(2)) -> AbstractArray{T, length(size)}

Return an `AbstractArray{T}` of the given `size` containing random numbers drawn from a
uniform distribution on the interval `[-x, x]`, where `x = gain * sqrt(3/fan_in)`.

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on
imagenet classification." _Proceedings of the IEEE international conference on computer
vision_. 2015.
"""
function kaiming_uniform(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        gain::Number=√T(2)) where {T <: Number}
    bound = √T(3) * gain / sqrt(T(first(_nfan(dims...))))
    return (rand(rng, T, dims...) .- T(1 // 2)) .* 2 * bound
end

"""
    kaiming_normal([::AbstractRNG=_default_rng()], [T=Float32], size...;
        gain = √T(2)) -> AbstractArray{T, length(size)}

Return an `AbstractArray{T}` of the given `size` containing random numbers taken from a
normal distribution standard deviation `gain / sqrt(fan_in)`

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on
imagenet classification." _Proceedings of the IEEE international conference on computer
vision_. 2015.
"""
function kaiming_normal(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        gain::Number=√T(2)) where {T <: Number}
    std = gain / sqrt(T(first(_nfan(dims...))))
    return randn(rng, T, dims...) .* std
end

"""
    truncated_normal([::AbstractRNG=_default_rng()], [T=Float32], size...; mean = 0,
        std = 1, lo = -2, hi = 2) -> AbstractArray{T, length(size)}

Return an `AbstractArray{T}` of the given `size` where each element is drawn from a
truncated normal distribution. The numbers are distributed like
`filter(x -> lo ≤ x ≤ hi, mean .+ std .* randn(100))`.
"""
function truncated_normal(rng::AbstractRNG, ::Type{T}, dims::Integer...; mean=T(0),
        std=T(1), lo=-T(2), hi=T(2)) where {T <: Real}
    if (mean < lo - 2 * std) || (mean > hi + 2 * std)
        @warn "Mean is more than 2 std outside the limits in truncated_normal, so the distribution of values may be inaccurate."
    end
    l = _norm_cdf((lo - mean) / std)
    u = _norm_cdf((hi - mean) / std)
    xs = rand(rng, T, dims...)
    broadcast!(xs, xs) do x
        x = x * 2(u - l) + (2l - 1)
        x = erfinv(x)
        return clamp(x * std * √2 + mean, lo, hi)
    end
    return xs
end

"""
    orthogonal(rng::AbstractRNG, ::Type{T}, dims::Integer...; gain = 1) where {T <: Real} -> AbstractArray{T, length(dims)}
    orthogonal(rng::AbstractRNG; kw...) -> Function

Return an `AbstractArray{T}` of the given dimensions (`dims`) which is a (semi) orthogonal matrix, as described in [^Saxe14]

The function constructs an orthogonal or semi-orthogonal matrix depending on the specified dimensions. For two dimensions, it returns a matrix where `dims = (rows, cols)`. For more than two dimensions, it computes an orthogonal matrix of size `prod(dims[1:(end - 1)])` by `dims[end]` before reshaping it to the original dimensions.

Cannot construct a vector, i.e., `length(dims) == 1` is forbidden.

# Arguments

  - `rng::AbstractRNG`: Random number generator.
  - `T::Type{<:Real}`: The type of the elements in the array.
  - `dims::Integer...`: The dimensions of the array.
  - `gain::Number`: Scaling factor for the elements of the orthogonal matrix.

# References

[^Saxe14] Saxe, McClelland, Ganguli. "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks", ICLR 2014, https://arxiv.org/abs/1312.6120
"""
function orthogonal(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        gain::Number=1) where {T <: Real}
    @assert length(dims) > 1 "Creating vectors (length(dims) == 1) is not allowed"
    rows, cols = dims
    if rows < cols
        return permutedims(orthogonal(rng, T, cols, rows; gain))
    end
    mat = randn(rng, T, rows, cols)
    Q, R = LinearAlgebra.qr(mat)
    mat .= Array(Q) * sign.(LinearAlgebra.Diagonal(R)) .* T(gain)
    return mat
end

# Default Fallbacks for all functions
for initializer in (:glorot_uniform, :glorot_normal, :kaiming_uniform, :kaiming_normal,
    :truncated_normal, :orthogonal)
    NType = ifelse(initializer === :truncated_normal, Real, Number)
    @eval function ($initializer)(dims::Integer...; kwargs...)
        return $initializer(_default_rng(), Float32, dims...; kwargs...)
    end
    @eval function ($initializer)(rng::AbstractRNG, dims::Integer...; kwargs...)
        return $initializer(rng, Float32, dims...; kwargs...)
    end
    @eval function ($initializer)(::Type{T},
            dims::Integer...; kwargs...) where {T <: $NType}
        return $initializer(_default_rng(), T, dims...; kwargs...)
    end
    @eval function ($initializer)(rng::AbstractRNG; kwargs...)
        return __partial_apply($initializer, (rng, (; kwargs...)))
    end
    @eval function ($initializer)(rng::AbstractRNG,
            ::Type{T}; kwargs...) where {T <: $NType}
        return __partial_apply($initializer, ((rng, T), (; kwargs...)))
    end
    @eval ($initializer)(; kwargs...) = __partial_apply($initializer, (; kwargs...))
end

for tp in ("16", "32", "64", "C16", "C32", "C64"), func in (:zeros, :ones, :randn, :rand)
    initializer = Symbol(func, tp)
    @eval function ($initializer)(dims::Integer...; kwargs...)
        return $initializer(_default_rng(), dims...; kwargs...)
    end
    @eval function ($initializer)(rng::AbstractRNG; kwargs...)
        return __partial_apply($initializer, (rng, (; kwargs...)))
    end
    @eval ($initializer)(; kwargs...) = __partial_apply($initializer, (; kwargs...))
end
