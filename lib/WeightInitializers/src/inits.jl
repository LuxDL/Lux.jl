
@inline _nfan() = 1, 1 # fan_in, fan_out
@inline _nfan(n) = 1, n # A vector is treated as a n×1 matrix
@inline _nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices
@inline _nfan(dims::Tuple) = _nfan(dims...)

function _default_rng()
    @static if VERSION >= v"1.7"
        return Xoshiro(1234)
    else
        return MersenneTwister(1234)
    end
end
  
"""
    default_rng_value()
  
Create an instance of the default RNG depending on Julia's version.
  - Julia version is < 1.7: `MersenneTwister(1234)`
  - Julia version is >= 1.7: `Xoshiro(1234)`
"""
_default_rng

"""
    zeros32(rng::AbstractRNG, size...) = zeros(Float32, size...)

Return an `Array{Float32}` of zeros of the given `size`. (`rng` is ignored)
"""
zeros32(rng::AbstractRNG, dims...) = zeros(rng, Float32, dims...)
zeros32(dims...) = zeros32(_default_rng(), dims...)
Base.zeros(rng::AbstractRNG, args...) = zeros(args...)
"""
    ones32(rng::AbstractRNG, size...) = ones(Float32, size...)

Return an `Array{Float32}` of ones of the given `size`. (`rng` is ignored)
"""
ones32(rng::AbstractRNG, dims...) = ones(rng, Float32, dims...)
ones32(dims...) = ones32(_default_rng(), dims...)
Base.ones(rng::AbstractRNG, dims...) = ones(dims...)

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
glorot_uniform(dims::Integer...; kw...) = glorot_uniform(_default_rng(), dims...; kwargs...)
glorot_uniform(rng::AbstractRNG=_default_rng(); init_kwargs...) = (dims...; kwargs...) -> glorot_uniform(rng, dims...; init_kwargs..., kwargs...)

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
glorot_normal(dims::Integer...; kwargs...) = glorot_normal(_default_rng(), dims...; kwargs...)
glorot_normal(rng::AbstractRNG=_default_rng(); init_kwargs...) = (dims...; kwargs...) -> glorot_normal(rng, dims...; init_kwargs..., kwargs...)


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
kaiming_uniform(dims::Integer...; kwargs...) = kaiming_uniform(_default_rng(), dims...; kwargs...)
kaiming_uniform(rng::AbstractRNG=_default_rng(); init_kwargs...) = (dims...; kwargs...) -> kaiming_uniform(rng, dims...; init_kwargs..., kwargs...)


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

kaiming_normal(dims::Integer...; kwargs...) = kaiming_normal(_default_rng(), dims...; kwargs...)
kaiming_normal(rng::AbstractRNG; init_kwargs...) = (dims...; kwargs...) -> kaiming_normal(rng, dims...; init_kwargs..., kwargs...)
