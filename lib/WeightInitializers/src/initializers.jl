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
    gain = gain isa T ? gain : convert(T, gain)
    scale = gain * sqrt(T(24) / sum(_nfan(dims...)))
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
    gain = gain isa T ? gain : convert(T, gain)
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
    gain = gain isa T ? gain : convert(T, gain)
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
    gain = gain isa T ? gain : convert(T, gain)
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
    mean = mean isa T ? mean : convert(T, mean)
    std = std isa T ? std : convert(T, std)
    lo = lo isa T ? lo : convert(T, lo)
    hi = hi isa T ? hi : convert(T, hi)
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
    orthogonal([::AbstractRNG=_default_rng()], [T=Float32], dims::Integer...;
        gain = 1)  -> AbstractArray{T, length(dims)}

Return an `AbstractArray{T}` of the given dimensions (`dims`) which is a
(semi) orthogonal matrix, as described in [^Saxe14]

The function constructs an orthogonal or semi-orthogonal matrix depending on the specified
dimensions. For two dimensions, it returns a matrix where `dims = (rows, cols)`.
For more than two dimensions, it computes an orthogonal matrix of
size `prod(dims[1:(end - 1)])` by `dims[end]` before reshaping it to
the original dimensions.

Cannot construct a vector, i.e., `length(dims) == 1` is forbidden.

# Arguments

  - `rng::AbstractRNG`: Random number generator.
  - `T::Type{<:Real}`: The type of the elements in the array.
  - `dims::Integer...`: The dimensions of the array.
  - `gain::Number`: Scaling factor for the elements of the orthogonal matrix.

# References

[^Saxe14] Saxe, McClelland, Ganguli. "Exact solutions to the nonlinear dynamics of
learning in deep linear neural networks",
ICLR 2014, https://arxiv.org/abs/1312.6120
"""
function orthogonal(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        gain::Number=T(1.0)) where {T <: Number}
    @assert length(dims)>1 "Creating vectors (length(dims) == 1) is not allowed"
    gain = gain isa T ? gain : convert(T, gain)

    if length(dims) == 2
        rows, cols = dims
    else
        rows = prod(dims[1:(end - 1)])
        cols = dims[end]
    end

    if rows < cols
        return permutedims(orthogonal(rng, T, cols, rows; gain))
    end

    mat = randn(rng, T, rows, cols)
    Q, R = qr(mat)
    mat .= Q * sign.(Diagonal(R)) .* T(gain)

    if length(dims) > 2
        return reshape(mat, dims)
    else
        return mat
    end
end

"""
    sparse_init([::AbstractRNG=_default_rng()], [T=Float32], dims::Integer...;
        sparsity::Number, std::Number=0.01) -> AbstractArray{T}

Creates a sparsely initialized weight matrix with a specified proportion of zeroed elements,
using random numbers drawn from a normal distribution for the non-zero elements.
This method is introduced in [^Martens2010].
Note: The sparsity parameter controls the proportion of the matrix that will be zeroed.
For example, a sparsity of 0.3 means that approximately 30% of the elements will be
set to zero. The non-zero elements are distributed according to a normal distribution,
scaled by the std parameter.

# Arguments

  - `rng::AbstractRNG`: The random number generator to use.
  - `T::Type{<:Number}`: The numeric type of the elements in the returned array.
  - `dims::Integer...`: The dimensions of the weight matrix to be generated.
  - `sparsity::Number`: The proportion of elements to be zeroed. Must be between 0 and 1.
  - `std::Number=0.01`: The standard deviation of the normal distribution
    before applying `gain`.

# Returns

  - `AbstractArray{T}`: A sparsely initialized weight matrix of dimensions `dims`
    and type `T`.

# Examples

```julia
using Random

# Initialize a 5x5 sparsely initialized matrix with 30% sparsity
rng = MersenneTwister(123)
matrix = sparse_init(rng, Float32, 5, 5; sparsity=0.3, std=0.01)
```

```
5×5 Matrix{Float64}:
  0.0          0.00273815    0.00592403   0.0          0.0
  0.00459416  -0.000754831  -0.00888936  -0.0077507    0.0
  0.0         -0.00194229    0.0          0.0         -0.00468489
  0.0114265    0.0           0.0         -0.00734886   0.00277726
 -0.00396679   0.0           0.00327215  -0.0071741   -0.00880897
```

# References

[^Martens2010] Martens, J, "Deep learning via Hessian-free optimization"
_Proceedings of the 27th International Conference on International Conference
on Machine Learning_. 2010.
"""
function sparse_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        sparsity::Number, std::Number=T(0.01)) where {T <: Number}
    if length(dims) != 2
        throw(ArgumentError("Only 2-dimensional outputs are supported for sparse initialization."))
    end

    std = std isa T ? std : convert(T, std)
    rows, cols = dims
    prop_zero = min(1.0, sparsity)
    num_zeros = ceil(Integer, prop_zero * rows)
    sparse_array = randn(rng, T, dims...) .* std
    sparse_array[1:num_zeros, :] .= zero(T)
    return mapslices(shuffle, sparse_array; dims=1)
end

"""
    identity_init([::AbstractRNG=_default_rng()], [T=Float32], size...; gain::Number=1,
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
    and appropriate padding must be applied to ensure the output
    feature maps are the same size as the input feature maps.

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

```julia
using Random

# Identity matrix for fully connected layer
identity_matrix = identity_init(MersenneTwister(123), Float32, 5, 5)

# Identity tensor for convolutional layer
identity_tensor = identity_init(MersenneTwister(123),
    Float32,        # Bias initialization
    3,
    3,
    5,        # Matrix multiplication
    5;
    gain=1.5,
    shift=(1, 0))
```
"""
function identity_init(rng::AbstractRNG, ::Type{T}, dims::Integer...;
        gain::Number=1, shift::Integer=0) where {T <: Number}
    gain = gain isa T ? gain : convert(T, gain)
    if length(dims) == 1
        # Bias initialization
        return zeros(T, dims...)
    elseif length(dims) == 2
        # Matrix multiplication
        rows, cols = dims
        mat = zeros(T, rows, cols)
        for i in 1:min(rows, cols)
            mat[i, i] = gain
        end
        return circshift(mat, shift)
    else
        # Convolution or more dimensions
        nin, nout = dims[end - 1], dims[end]
        centers = map(d -> cld(d, 2), dims[1:(end - 2)])
        weights = zeros(T, dims...)
        for i in 1:min(nin, nout)
            index = (centers..., i, i)
            weights[index...] = gain
        end
        return circshift(weights, (ntuple(d -> 0, length(dims) - 2)..., shift, shift))
    end
end

# Default Fallbacks for all functions
for initializer in (:glorot_uniform, :glorot_normal, :kaiming_uniform, :kaiming_normal,
    :truncated_normal, :orthogonal, :sparse_init, :identity_init)
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
