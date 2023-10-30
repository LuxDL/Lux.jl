"""
    zeros32([::AbstractRNG=_default_rng()], size...) -> Array{Float32, length(size)}

Return an `Array{Float32}` of zeros of the given `size`. (`rng` is ignored)
"""
zeros32(::AbstractRNG, dims...) = zeros(Float32, dims...)

"""
    ones32([::AbstractRNG=_default_rng()], size...) -> Array{Float32, length(size)}

Return an `Array{Float32}` of ones of the given `size`. (`rng` is ignored)
"""
ones32(::AbstractRNG, dims...) = ones(Float32, dims...)

"""
    randn32([::AbstractRNG=_default_rng()], size...) -> Array{Float32, length(size)}

Return an `Array{Float32}` of random numbers from a standard normal distribution of the
given `size`.
"""
randn32(rng::AbstractRNG, dims...) = randn(rng, Float32, dims...)

"""
    rand32([::AbstractRNG=_default_rng()], size...) -> Array{Float32, length(size)}

Return an `Array{Float32}` of random numbers from a uniform distribution of the given
`size`.
"""
rand32(rng::AbstractRNG, dims...) = rand(rng, Float32, dims...)

"""
    glorot_uniform([::AbstractRNG=_default_rng()], [T=Float32], size...;
        gain = 1) -> Array{T, length(size)}

Return an `Array{T}` of the given `size` containing random numbers drawn from a
uniform distribution on the interval ``[-x, x]``, where
`x = gain * sqrt(6 / (fan_in + fan_out))`. This method is described in [1] and also known as
Xavier initialization.

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep
feedforward neural networks." _Proceedings of the thirteenth international conference on
artificial intelligence and statistics_. 2010.
"""
function glorot_uniform(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        gain::Real=1) where {T <: Real}
    scale = T(gain) * sqrt(T(24) / sum(_nfan(dims...)))
    return (rand(rng, T, dims...) .- T(1 // 2)) .* scale
end

"""
    glorot_normal([::AbstractRNG=_default_rng()], [T=Float32], size...;
        gain = 1) -> Array{T, length(size)}

Return an `Array{T}` of the given `size` containing random numbers drawn from a normal
distribution with standard deviation `gain * sqrt(2 / (fan_in + fan_out))`. This method is
described in [1] and also known as Xavier initialization.

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep
feedforward neural networks." _Proceedings of the thirteenth international conference on
artificial intelligence and statistics_. 2010.
"""
function glorot_normal(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        gain::Real=1) where {T <: Real}
    std = T(gain) * sqrt(T(2) / sum(_nfan(dims...)))
    return randn(rng, T, dims...) .* std
end

"""
    kaiming_uniform([::AbstractRNG=_default_rng()], [T=Float32], size...;
        gain = √T(2)) -> Array{T, length(size)}

Return an `Array{T}` of the given `size` containing random numbers drawn from a
uniform distribution on the interval `[-x, x]`, where `x = gain * sqrt(3/fan_in)`.

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on
imagenet classification." _Proceedings of the IEEE international conference on computer
vision_. 2015.
"""
function kaiming_uniform(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        gain::Real=√T(2)) where {T <: Real}
    bound = √T(3) * gain / sqrt(T(first(_nfan(dims...))))
    return (rand(rng, T, dims...) .- T(1 // 2)) .* 2 * bound
end

"""
    kaiming_normal([::AbstractRNG=_default_rng()], [T=Float32], size...;
        gain = √T(2)) -> Array{T, length(size)}

Return an `Array{T}` of the given `size` containing random numbers taken from a normal
distribution standard deviation `gain / sqrt(fan_in)`

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on
imagenet classification." _Proceedings of the IEEE international conference on computer
vision_. 2015.
"""
function kaiming_normal(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        gain::Real=√T(2)) where {T <: Real}
    std = gain / sqrt(T(first(_nfan(dims...))))
    return randn(rng, T, dims...) .* std
end

"""
    truncated_normal([::AbstractRNG=_default_rng()], [T=Float32], size...; mean = 0, std = 1,
        lo = -2, hi = 2) -> Array{T, length(size)}

Return an `Array{T}` of the given `size` where each element is drawn from a truncated normal
distribution. The numbers are distributed like
`filter(x -> lo ≤ x ≤ hi, mean .+ std .* randn(100))`.
"""
function truncated_normal(rng::AbstractRNG, ::Type{T}, dims::Integer...; mean=T(0),
        std=T(1), lo=-T(2), hi=T(2)) where {T <: Real}
    if (mean < lo - 2 * std) || (mean > hi + 2 * std)
        @warn "Mean is more than 2 std outside the limits in truncated_normal, so the distribution of values may be inaccurate." maxlog=1
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

# Default Fallbacks for all functions
for initializer in (:glorot_uniform, :glorot_normal, :kaiming_uniform, :kaiming_normal,
    :truncated_normal)
    @eval function ($initializer)(dims::Integer...; kwargs...)
        return $initializer(_default_rng(), Float32, dims...; kwargs...)
    end
    @eval function ($initializer)(rng::AbstractRNG, dims::Integer...; kwargs...)
        return $initializer(rng, Float32, dims...; kwargs...)
    end
    @eval function ($initializer)(::Type{T}, dims::Integer...; kwargs...) where {T <: Real}
        return $initializer(_default_rng(), T, dims...; kwargs...)
    end
    @eval function ($initializer)(rng::AbstractRNG; kwargs...)
        return _partial_apply($initializer, (rng, (; kwargs...)))
    end
    @eval function ($initializer)(rng::AbstractRNG, ::Type{T}; kwargs...) where {T <: Real}
        return _partial_apply($initializer, ((rng, T), (; kwargs...)))
    end
    @eval ($initializer)(; kwargs...) = _partial_apply($initializer, (; kwargs...))
end

for initializer in (:zeros32, :ones32, :randn32, :rand32)
    @eval function ($initializer)(dims::Integer...; kwargs...)
        return $initializer(_default_rng(), dims...; kwargs...)
    end
    @eval function ($initializer)(rng::AbstractRNG; kwargs...)
        return _partial_apply($initializer, (rng, (; kwargs...)))
    end
end
