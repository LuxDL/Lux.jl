"""
    zeros32(rng::AbstractRNG, size...) = zeros(Float32, size...)

Return an `Array{Float32}` of zeros of the given `size`. (`rng` is ignored)
"""
zeros32(::AbstractRNG, dims...) = zeros(Float32, dims...)
zeros32(dims...) = zeros32(_default_rng(), dims...)

"""
    ones32(rng::AbstractRNG, size...) = ones(Float32, size...)

Return an `Array{Float32}` of ones of the given `size`. (`rng` is ignored)
"""
ones32(::AbstractRNG, dims...) = ones(Float32, dims...)
ones32(dims...) = ones32(_default_rng(), dims...)

"""
    randn32(rng::AbstractRNG, size...) = randn(rng, Float32, size...)

Return an `Array{Float32}` of random numbers from a standard normal distribution of the
given `size`.
"""
randn32(rng::AbstractRNG, dims...) = randn(rng, Float32, dims...)
randn32(dims...) = randn32(_default_rng(), dims...)

"""
    rand32(rng::AbstractRNG, size...) = rand(rng, Float32, size...)

Return an `Array{Float32}` of random numbers from a uniform distribution of the given
`size`.
"""
rand32(rng::AbstractRNG, dims...) = rand(rng, Float32, dims...)
rand32(dims...) = rand32(_default_rng(), dims...)

"""
    glorot_uniform(rng::AbstractRNG, size...; gain = 1)

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a
uniform distribution on the interval ``[-x, x]``, where
`x = gain * sqrt(6 / (fan_in + fan_out))`. This method is described in [1] and also known as
Xavier initialization.

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep
feedforward neural networks." _Proceedings of the thirteenth international conference on
artificial intelligence and statistics_. 2010.
"""
function glorot_uniform(rng::AbstractRNG, dims::Integer...; gain::Real=1)
    scale = Float32(gain) * sqrt(24.0f0 / sum(_nfan(dims...)))
    return (rand(rng, Float32, dims...) .- 0.5f0) .* scale
end

function glorot_uniform(dims::Integer...; kwargs...)
    return glorot_uniform(_default_rng(), dims...; kwargs...)
end

function glorot_uniform(; kwargs...)
    return glorot_uniform $ (; kwargs...)
end

"""
    glorot_normal(rng::AbstractRNG, size...; gain = 1)

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a normal
distribution with standard deviation `gain * sqrt(2 / (fan_in + fan_out))`. This method is
described in [1] and also known as Xavier initialization.

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep
feedforward neural networks." _Proceedings of the thirteenth international conference on
artificial intelligence and statistics_. 2010.
"""
function glorot_normal(rng::AbstractRNG, dims::Integer...; gain::Real=1)
    std = Float32(gain) * sqrt(2.0f0 / sum(_nfan(dims...)))
    return randn(rng, Float32, dims...) .* std
end

function glorot_normal(dims::Integer...; kwargs...)
    return glorot_normal(_default_rng(), dims...; kwargs...)
end

function glorot_normal(rng::AbstractRNG; kwargs...)
    return glorot_normal $ (; kwargs...)
end

"""
    kaiming_uniform(rng::AbstractRNG, size...; gain = √2f0)

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a
uniform distribution on the interval `[-x, x]`, where `x = gain * sqrt(3/fan_in)`.

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on
imagenet classification." _Proceedings of the IEEE international conference on computer
vision_. 2015.
"""
function kaiming_uniform(rng::AbstractRNG, dims::Integer...; gain::Real=√2.0f0)
    bound = Float32(√3.0f0 * gain / sqrt(first(_nfan(dims...))))
    return (rand(rng, Float32, dims...) .- 0.5f0) .* 2 * bound
end

function kaiming_uniform(dims::Integer...; kwargs...)
    return kaiming_uniform(_default_rng(), dims...; kwargs...)
end

function kaiming_uniform(rng::AbstractRNG; kwargs...)
    return kaiming_uniform $ (; kwargs...)
end

"""
    kaiming_normal(rng::AbstractRNG, size...; gain = √2f0)

Return an `Array{Float32}` of the given `size` containing random numbers taken from a normal
distribution standard deviation `gain / sqrt(fan_in)`

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on
imagenet classification." _Proceedings of the IEEE international conference on computer
vision_. 2015.
"""
function kaiming_normal(rng::AbstractRNG, dims::Integer...; gain::Real=√2.0f0)
    std = Float32(gain / sqrt(first(_nfan(dims...))))
    return randn(rng, Float32, dims...) .* std
end

function kaiming_normal(dims::Integer...; kwargs...)
    return kaiming_normal(_default_rng(), dims...; kwargs...)
end

function kaiming_normal(rng::AbstractRNG; kwargs...)
    return kaiming_normal $ (; kwargs...)
end

"""
    truncated_normal([rng = default_rng_value()], size...; mean = 0, std = 1, lo = -2, hi = 2)

Return an `Array{Float32}` of the given `size` where each element is drawn from a truncated normal distribution.
The numbers are distributed like `filter(x -> lo<=x<=hi, mean .+ std .* randn(100))`.
"""
function truncated_normal(rng::AbstractRNG, dims::Integer...; mean=0, std=1, lo=-2, hi=2)
    if (mean < lo - 2 * std) || (mean > hi + 2 * std)
        @warn "Mean is more than 2 std outside the limits in truncated_normal, so the distribution of values may be inaccurate." maxlog=1
    end
    l = _norm_cdf((lo - mean) / std)
    u = _norm_cdf((hi - mean) / std)
    xs = rand(rng, Float32, dims...)
    broadcast!(xs, xs) do x
        x = x * 2(u - l) + (2l - 1)
        x = erfinv(x)
        return x = clamp(x * std * √2 + mean, lo, hi)
    end
    return xs
end

function truncated_normal(dims::Integer...; kwargs...)
    return truncated_normal(_default_rng(), dims...; kwargs...)
end
function truncated_normal(rng::AbstractRNG; init_kwargs...)
    return (rng, dims...; kwargs...) -> truncated_normal(rng,
        dims...;
        init_kwargs...,
        kwargs...)
end
function truncated_normal(; kwargs...)
    return truncated_normal $ (; kwargs...)
end
