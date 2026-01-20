for T in ("16", "32", "64", "C16", "C32", "C64"), fname in (:ones, :zeros, :rand, :randn)
    name = Symbol(fname, T)
    docstring = Utils.generic_docstring(string(name))
    TP = Utils.NUM_TO_FPOINT[Symbol(T)]

    @eval begin
        @doc $docstring function $(name)(rng::AbstractRNG, dims::Integer...; kwargs...)
            return DeviceAgnostic.$(fname)(rng, $TP, dims...; kwargs...)
        end
    end
end

"""
    glorot_uniform([::AbstractRNG=Utils.default_rng()], [T=Float32], size...;
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
function glorot_uniform(
    rng::AbstractRNG, ::Type{T}, dims::Integer...; gain::Number=1
) where {T<:Number}
    scale = Utils.safe_type_conversion(T, gain * sqrt(24 / sum(Utils.nfan(dims...))))
    half = Utils.safe_type_conversion(T, 0.5)
    x = DeviceAgnostic.rand(rng, T, dims...)
    @. x = (x - half) * scale
    return x
end

"""
    glorot_normal([::AbstractRNG=Utils.default_rng()], [T=Float32], size...;
        gain = 1) -> AbstractArray{T, length(size)}

Return an `AbstractArray{T}` of the given `size` containing random numbers drawn from a
normal distribution with standard deviation `gain * sqrt(2 / (fan_in + fan_out))`. This
method is described in [1] and also known as Xavier initialization.

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep
feedforward neural networks." _Proceedings of the thirteenth international conference on
artificial intelligence and statistics_. 2010.
"""
function glorot_normal(
    rng::AbstractRNG, ::Type{T}, dims::Integer...; gain::Number=1
) where {T<:Number}
    std = Utils.safe_type_conversion(T, gain * sqrt(2 / sum(Utils.nfan(dims...))))
    x = DeviceAgnostic.randn(rng, T, dims...)
    x .*= std
    return x
end

"""
    kaiming_uniform([::AbstractRNG=Utils.default_rng()], [T=Float32], size...;
        gain = √T(2)) -> AbstractArray{T, length(size)}

Return an `AbstractArray{T}` of the given `size` containing random numbers drawn from a
uniform distribution on the interval `[-x, x]`, where `x = gain * sqrt(3/fan_in)`.

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on
imagenet classification." _Proceedings of the IEEE international conference on computer
vision_. 2015.
"""
function kaiming_uniform(
    rng::AbstractRNG, ::Type{T}, dims::Integer...; gain::Number=√T(2)
) where {T<:Number}
    bound = Utils.safe_type_conversion(T, √3 * gain / sqrt(first(Utils.nfan(dims...))))
    half = Utils.safe_type_conversion(T, 0.5)
    x = DeviceAgnostic.rand(rng, T, dims...)
    @. x = (x - half) * 2 * bound
    return x
end

"""
    kaiming_normal([::AbstractRNG=Utils.default_rng()], [T=Float32], size...;
        gain = √T(2)) -> AbstractArray{T, length(size)}

Return an `AbstractArray{T}` of the given `size` containing random numbers taken from a
normal distribution standard deviation `gain / sqrt(fan_in)`

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on
imagenet classification." _Proceedings of the IEEE international conference on computer
vision_. 2015.
"""
function kaiming_normal(
    rng::AbstractRNG, ::Type{T}, dims::Integer...; gain::Number=√T(2)
) where {T<:Number}
    std = Utils.safe_type_conversion(T, gain / sqrt(first(Utils.nfan(dims...))))
    x = DeviceAgnostic.randn(rng, T, dims...)
    @. x *= std
    return x
end

"""
    truncated_normal([::AbstractRNG=Utils.default_rng()], [T=Float32], size...; mean = 0,
        std = 1, lo = -2, hi = 2) -> AbstractArray{T, length(size)}

Return an `AbstractArray{T}` of the given `size` where each element is drawn from a
truncated normal distribution. The numbers are distributed like
`filter(x -> lo ≤ x ≤ hi, mean .+ std .* randn(100))`.
"""
function truncated_normal(
    rng::AbstractRNG, ::Type{T}, dims::Integer...; mean=T(0), std=T(1), lo=-T(2), hi=T(2)
) where {T<:Real}
    if (mean < lo - 2 * std) || (mean > hi + 2 * std)
        @warn "Mean is more than 2 std outside the limits in truncated_normal, so the \
               distribution of values may be inaccurate."
    end
    l = Utils.norm_cdf((T(lo) - T(mean)) / T(std))
    u = Utils.norm_cdf((T(hi) - T(mean)) / T(std))
    xs = DeviceAgnostic.rand(rng, T, dims...)
    broadcast!(xs, xs) do x
        x = x * 2(u - l) + (2l - one(T))
        x = erfinv(x)
        return clamp(x * T(std) * √T(2) + T(mean), T(lo), T(hi))
    end
    return xs
end

"""
    orthogonal([::AbstractRNG=Utils.default_rng()], [T=Float32], dims::Integer...;
        gain = 1)  -> AbstractArray{T, length(dims)}

Return an `AbstractArray{T}` of the given dimensions (`dims`) which is a
(semi) orthogonal matrix, as described in [1].

The function constructs an orthogonal or semi-orthogonal matrix depending on the specified
dimensions. For two dimensions, it returns a matrix where `dims = (rows, cols)`. For more
than two dimensions, it computes an orthogonal matrix of size `prod(dims[1:(end - 1)])` by
`dims[end]` before reshaping it to the original dimensions.

Cannot construct a vector, i.e., `length(dims) == 1` is forbidden.

# Arguments

  - `rng::AbstractRNG`: Random number generator.
  - `T::Type{<:Real}`: The type of the elements in the array.
  - `dims::Integer...`: The dimensions of the array.
  - `gain::Number`: Scaling factor for the elements of the orthogonal matrix.

# References

[1] Saxe, McClelland, Ganguli. "Exact solutions to the nonlinear dynamics of learning in
deep linear neural networks", ICLR 2014, https://arxiv.org/abs/1312.6120
"""
function orthogonal(
    rng::AbstractRNG, ::Type{T}, dims::Integer...; gain::Number=T(1.0)
) where {T<:Number}
    @assert length(dims) > 1 "Creating vectors (length(dims) == 1) is not allowed"

    rows, cols = length(dims) == 2 ? dims : (prod(dims[1:(end - 1)]), dims[end])
    rows < cols && return permutedims(orthogonal(rng, T, cols, rows; gain=T(gain)))

    mat = DeviceAgnostic.randn(rng, T, rows, cols)
    Q, R = qr(mat)
    mat .= Q * sign.(Diagonal(R)) .* T(gain)

    return length(dims) > 2 ? reshape(mat, dims) : mat
end

"""
    sparse_init([::AbstractRNG=Utils.default_rng()], [T=Float32], dims::Integer...;
        sparsity::Number, std::Number=0.01) -> AbstractArray{T}

Creates a sparsely initialized weight matrix with a specified proportion of zeroed elements,
using random numbers drawn from a normal distribution for the non-zero elements. This method
was introduced in [1].

!!! note

    The sparsity parameter controls the proportion of the matrix that will be zeroed. For
    example, a sparsity of 0.3 means that approximately 30% of the elements will be set to
    zero. The non-zero elements are distributed according to a normal distribution, scaled
    by the std parameter.

# Arguments

  - `rng::AbstractRNG`: The random number generator to use.
  - `T::Type{<:Number}`: The numeric type of the elements in the returned array.
  - `dims::Integer...`: The dimensions of the weight matrix to be generated.
  - `sparsity::Number`: The proportion of elements to be zeroed. Must be between 0 and 1.
  - `std::Number=0.01`: The standard deviation of the normal distribution before applying
    `gain`.

# Returns

  - `AbstractArray{T}`: A sparsely initialized weight matrix of dimensions `dims` and type
    `T`.

# Examples

```jldoctest
julia> y = sparse_init(Xoshiro(123), Float32, 5, 5; sparsity=0.3, std=0.01);

julia> y isa Matrix{Float32}
true

julia> size(y) == (5, 5)
true
```

# References

[1] Martens, J, "Deep learning via Hessian-free optimization" Proceedings of the 27th
International Conference on International Conference on Machine Learning. 2010.
"""
function sparse_init(
    rng::AbstractRNG, ::Type{T}, dims::Integer...; sparsity::Number, std::Number=T(0.01)
) where {T<:Number}
    if length(dims) != 2
        throw(ArgumentError("Only 2-dimensional outputs are supported for sparse \
                             initialization."))
    end

    rows, _ = dims
    prop_zero = min(1.0, sparsity)
    num_zeros = ceil(Integer, prop_zero * rows)

    sparse_array = DeviceAgnostic.randn(rng, T, dims...)
    sparse_array .*= T(std)
    fill!(view(sparse_array, 1:num_zeros, :), zero(T))

    if applicable(Random.rng_native_52, rng)
        @inbounds for i in axes(sparse_array, 2)
            @allowscalar Random.shuffle!(rng, view(sparse_array, :, i))
        end
    else
        @warn "`rng` is not supported by `Random.shuffle!`. Ignoring the `rng` for \
               shuffle." maxlog = 1
        @inbounds for i in axes(sparse_array, 2)
            @allowscalar Random.shuffle!(view(sparse_array, :, i))
        end
    end

    return sparse_array
end

"""
    identity_init([::AbstractRNG=Utils.default_rng()], [T=Float32], size...; gain::Number=1,
        shift::Union{Integer, Tuple{Integer, Integer}}=0) -> AbstractArray{T}

Constructs an array that aims to provide an identity mapping when used as parameters in
most layers of a neural network. The identity mapping is scaled by the `gain` parameter.

# Behavior

  - 1D: Returns a `Vector` of zeros (useful for biases in layers where
    `input_size == output_size`).
  - 2D: Returns an identity matrix
    (useful for fully connected layers with equal input and output sizes).
  - More than 2D: Returns a tensor where the central slice along the last
    two dimensions is an identity matrix, and the rest are zeros
    (useful for convolutional layers, simulating an identity convolution).

# Caveats

  - Not all layers will result in an identity mapping when using this initializer.
    Exceptions include recurrent and normalization layers.
  - Layers must have `input_size == output_size` for a perfect identity mapping.
    In cases where this condition is not met, the function pads extra dimensions with zeros.
  - For convolutional layers to achieve an identity mapping, kernel sizes must be odd,
    and appropriate padding must be applied to ensure the output feature maps are the same
    size as the input feature maps.

# Arguments

  - `rng::AbstractRNG`: An optional random number generator,
    included for consistency with other initializers but ignored since the
    output is deterministic.
  - `T::Type{<:Number}`: The numeric type of the array elements.
  - `size...`: The dimensions of the array to be initialized.
  - `gain::Number=1`: A scaling factor applied to the identity mapping.
  - `shift::Union{Integer, Tuple{Integer, Integer}}=0`: An integer or
    a tuple specifying the circular shift applied to the output array.

# Returns

  - `AbstractArray{T}`: An array initialized to represent an identity mapping,
    scaled by `gain` and optionally shifted by `shift`.

# Examples

```jldoctest
julia> identity_init(Xoshiro(123), Float32, 5, 5)
5×5 Matrix{Float32}:
 1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  1.0

julia> identity_init(Xoshiro(123), Float32, 3, 3, 1, 1; gain=1.5)
3×3×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 0.0  0.0  0.0
 0.0  1.5  0.0
 0.0  0.0  0.0
```
"""
function identity_init(
    rng::AbstractRNG, ::Type{T}, dims::Integer...; gain::Number=1, shift::Integer=0
) where {T<:Number}
    length(dims) == 1 && return DeviceAgnostic.zeros(rng, T, dims...)  # Bias initialization

    if length(dims) == 2
        rows, cols = dims
        mat = DeviceAgnostic.zeros(rng, T, rows, cols)
        fill!(view(mat, diagind(mat)), T(gain))
        return circshift(mat, shift)
    end

    # Convolution or more dimensions
    nin, nout = dims[end - 1], dims[end]
    centers = map(d -> cld(d, 2), dims[1:(end - 2)])
    weights = DeviceAgnostic.zeros(rng, T, dims...)
    @allowscalar for i in 1:min(nin, nout)
        index = (centers..., i, i)
        weights[index...] = T(gain)
    end
    return circshift(weights, (ntuple(d -> 0, length(dims) - 2)..., shift, shift))
end

# Default Fallbacks for all functions
for initializer in (
    :glorot_uniform,
    :glorot_normal,
    :kaiming_uniform,
    :kaiming_normal,
    :truncated_normal,
    :orthogonal,
    :sparse_init,
    :identity_init,
)
    NType = ifelse(initializer === :truncated_normal, Real, Number)
    @eval begin
        function ($initializer)(dims::Integer...; kwargs...)
            return $initializer(Utils.default_rng(), Float32, dims...; kwargs...)
        end
        function ($initializer)(rng::AbstractRNG, dims::Integer...; kwargs...)
            return $initializer(rng, Float32, dims...; kwargs...)
        end
        function ($initializer)(::Type{T}, dims::Integer...; kwargs...) where {T<:$NType}
            return $initializer(Utils.default_rng(), T, dims...; kwargs...)
        end

        # Partial application
        function ($initializer)(rng::AbstractRNG; kwargs...)
            return PartialFunction.Partial{Nothing}($initializer, rng, kwargs)
        end
        function ($initializer)(::Type{T}; kwargs...) where {T<:$NType}
            return PartialFunction.Partial{T}($initializer, nothing, kwargs)
        end
        function ($initializer)(rng::AbstractRNG, ::Type{T}; kwargs...) where {T<:$NType}
            return PartialFunction.Partial{T}($initializer, rng, kwargs)
        end
        function ($initializer)(; kwargs...)
            return PartialFunction.Partial{Nothing}($initializer, nothing, kwargs)
        end
    end
end

for tp in ("16", "32", "64", "C16", "C32", "C64"), func in (:zeros, :ones, :randn, :rand)
    initializer = Symbol(func, tp)
    @eval begin
        function ($initializer)(dims::Integer...; kwargs...)
            return $initializer(Utils.default_rng(), dims...; kwargs...)
        end
        function ($initializer)(::Type{T}, dims::Integer...; kwargs...) where {T}
            throw(ArgumentError(string($initializer) * " doesn't accept a type argument."))
        end
        function ($initializer)(
            ::AbstractRNG, ::Type{T}, dims::Integer...; kwargs...
        ) where {T}
            throw(ArgumentError(string($initializer) * " doesn't accept a type argument."))
        end

        # Partial application
        function ($initializer)(rng::AbstractRNG; kwargs...)
            return PartialFunction.Partial{Missing}($initializer, rng, kwargs)
        end
        function ($initializer)(rng::AbstractRNG, ::Type{T}; kwargs...) where {T}
            throw(ArgumentError(string($initializer) * " doesn't accept a type argument."))
        end
        function ($initializer)(; kwargs...)
            return PartialFunction.Partial{Missing}($initializer, nothing, kwargs)
        end
    end
end
