---
url: /dev/api/Building_Blocks/WeightInitializers.md
---
# WeightInitializers {#WeightInitializers-API}

This package is a light dependency providing common weight initialization schemes for deep learning models.

## Supported RNG Types {#Supported-RNG-Types-WeightInit}

|            **RNG Type / Package** | **Returned Array Type** |                                                                                                                                                                **Unsupported Functions** |
| ---------------------------------:| -----------------------:| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                       `Random.jl` |                 `Array` |                                                                                                                                                                                          |
|                   `StableRNGs.jl` |                 `Array` |                                                                                                                                                                                          |
|       `CUDA.CURAND.default_rng()` |               `CuArray` |                                                                                                                                                                                          |
|              `CUDA.default_rng()` |               `CuArray` |                                                                                                                                                                                          |
|  `GPUArrays.default_rng(CuArray)` |               `CuArray` |                                                                                                                                                                                          |
|            `AMDGPU.rocrand_rng()` |              `ROCArray` |                                                                                                                                                                                          |
|          `AMDGPU.gpuarrays_rng()` |              `ROCArray` |                                                                                                                                                                                          |
| `GPUArrays.default_rng(ROCArray)` |              `ROCArray` |                                                                                                                                                                                          |
|           `Metal.gpuarrays_rng()` |              `MtlArray` |                                                                                                    [`orthogonal`](/api/Building_Blocks/WeightInitializers#WeightInitializers.orthogonal) |
| `GPUArrays.default_rng(MtlArray)` |              `MtlArray` |                                                                                                    [`orthogonal`](/api/Building_Blocks/WeightInitializers#WeightInitializers.orthogonal) |
|          `oneAPI.gpuarrays_rng()` |              `oneArray` | [`orthogonal`](/api/Building_Blocks/WeightInitializers#WeightInitializers.orthogonal), [`truncated_normal`](/api/Building_Blocks/WeightInitializers#WeightInitializers.truncated_normal) |
| `GPUArrays.default_rng(oneArray)` |              `oneArray` | [`orthogonal`](/api/Building_Blocks/WeightInitializers#WeightInitializers.orthogonal), [`truncated_normal`](/api/Building_Blocks/WeightInitializers#WeightInitializers.truncated_normal) |

## API Reference {#API-Reference}

### Main Functions {#Main-Functions}

```julia
glorot_normal([::AbstractRNG=Utils.default_rng()], [T=Float32], size...;
    gain = 1) -> AbstractArray{T, length(size)}
```

Return an `AbstractArray{T}` of the given `size` containing random numbers drawn from a normal distribution with standard deviation `gain * sqrt(2 / (fan_in + fan_out))`. This method is described in \[1] and also known as Xavier initialization.

**References**

\[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." *Proceedings of the thirteenth international conference on artificial intelligence and statistics*. 2010.

source

```julia
glorot_uniform([::AbstractRNG=Utils.default_rng()], [T=Float32], size...;
    gain = 1) -> AbstractArray{T, length(size)}
```

Return an `AbstractArray{T}` of the given `size` containing random numbers drawn from a uniform distribution on the interval $\[-x, x]$, where `x = gain * sqrt(6 / (fan_in + fan_out))`. This method is described in \[1] and also known as Xavier initialization.

**References**

\[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." *Proceedings of the thirteenth international conference on artificial intelligence and statistics*. 2010.

source

```julia
identity_init([::AbstractRNG=Utils.default_rng()], [T=Float32], size...; gain::Number=1,
    shift::Union{Integer, Tuple{Integer, Integer}}=0) -> AbstractArray{T}
```

Constructs an array that aims to provide an identity mapping when used as parameters in most layers of a neural network. The identity mapping is scaled by the `gain` parameter.

**Behavior**

* 1D: Returns a `Vector` of zeros (useful for biases in layers where `input_size == output_size`).

* 2D: Returns an identity matrix (useful for fully connected layers with equal input and output sizes).

* More than 2D: Returns a tensor where the central slice along the last two dimensions is an identity matrix, and the rest are zeros (useful for convolutional layers, simulating an identity convolution).

**Caveats**

* Not all layers will result in an identity mapping when using this initializer. Exceptions include recurrent and normalization layers.

* Layers must have `input_size == output_size` for a perfect identity mapping. In cases where this condition is not met, the function pads extra dimensions with zeros.

* For convolutional layers to achieve an identity mapping, kernel sizes must be odd, and appropriate padding must be applied to ensure the output feature maps are the same size as the input feature maps.

**Arguments**

* `rng::AbstractRNG`: An optional random number generator, included for consistency with other initializers but ignored since the output is deterministic.

* `T::Type{<:Number}`: The numeric type of the array elements.

* `size...`: The dimensions of the array to be initialized.

* `gain::Number=1`: A scaling factor applied to the identity mapping.

* `shift::Union{Integer, Tuple{Integer, Integer}}=0`: An integer or a tuple specifying the circular shift applied to the output array.

**Returns**

* `AbstractArray{T}`: An array initialized to represent an identity mapping, scaled by `gain` and optionally shifted by `shift`.

**Examples**

```julia
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

source

```julia
kaiming_normal([::AbstractRNG=Utils.default_rng()], [T=Float32], size...;
    gain = √T(2)) -> AbstractArray{T, length(size)}
```

Return an `AbstractArray{T}` of the given `size` containing random numbers taken from a normal distribution standard deviation `gain / sqrt(fan_in)`

**References**

\[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." *Proceedings of the IEEE international conference on computer vision*. 2015.

source

```julia
kaiming_uniform([::AbstractRNG=Utils.default_rng()], [T=Float32], size...;
    gain = √T(2)) -> AbstractArray{T, length(size)}
```

Return an `AbstractArray{T}` of the given `size` containing random numbers drawn from a uniform distribution on the interval `[-x, x]`, where `x = gain * sqrt(3/fan_in)`.

**References**

\[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." *Proceedings of the IEEE international conference on computer vision*. 2015.

source

```julia
sparse_init([::AbstractRNG=Utils.default_rng()], [T=Float32], dims::Integer...;
    sparsity::Number, std::Number=0.01) -> AbstractArray{T}
```

Creates a sparsely initialized weight matrix with a specified proportion of zeroed elements, using random numbers drawn from a normal distribution for the non-zero elements. This method was introduced in \[1].

::: tip Note

The sparsity parameter controls the proportion of the matrix that will be zeroed. For example, a sparsity of 0.3 means that approximately 30% of the elements will be set to zero. The non-zero elements are distributed according to a normal distribution, scaled by the std parameter.

:::

**Arguments**

* `rng::AbstractRNG`: The random number generator to use.

* `T::Type{<:Number}`: The numeric type of the elements in the returned array.

* `dims::Integer...`: The dimensions of the weight matrix to be generated.

* `sparsity::Number`: The proportion of elements to be zeroed. Must be between 0 and 1.

* `std::Number=0.01`: The standard deviation of the normal distribution before applying `gain`.

**Returns**

* `AbstractArray{T}`: A sparsely initialized weight matrix of dimensions `dims` and type `T`.

**Examples**

```julia
julia> y = sparse_init(Xoshiro(123), Float32, 5, 5; sparsity=0.3, std=0.01);

julia> y isa Matrix{Float32}
true

julia> size(y) == (5, 5)
true
```

**References**

\[1] Martens, J, "Deep learning via Hessian-free optimization" Proceedings of the 27th International Conference on International Conference on Machine Learning. 2010.

source

```julia
truncated_normal([::AbstractRNG=Utils.default_rng()], [T=Float32], size...; mean = 0,
    std = 1, lo = -2, hi = 2) -> AbstractArray{T, length(size)}
```

Return an `AbstractArray{T}` of the given `size` where each element is drawn from a truncated normal distribution. The numbers are distributed like `filter(x -> lo ≤ x ≤ hi, mean .+ std .* randn(100))`.

source

```julia
orthogonal([::AbstractRNG=Utils.default_rng()], [T=Float32], dims::Integer...;
    gain = 1)  -> AbstractArray{T, length(dims)}
```

Return an `AbstractArray{T}` of the given dimensions (`dims`) which is a (semi) orthogonal matrix, as described in \[1].

The function constructs an orthogonal or semi-orthogonal matrix depending on the specified dimensions. For two dimensions, it returns a matrix where `dims = (rows, cols)`. For more than two dimensions, it computes an orthogonal matrix of size `prod(dims[1:(end - 1)])` by `dims[end]` before reshaping it to the original dimensions.

Cannot construct a vector, i.e., `length(dims) == 1` is forbidden.

**Arguments**

* `rng::AbstractRNG`: Random number generator.

* `T::Type{<:Real}`: The type of the elements in the array.

* `dims::Integer...`: The dimensions of the array.

* `gain::Number`: Scaling factor for the elements of the orthogonal matrix.

**References**

\[1] Saxe, McClelland, Ganguli. "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks", ICLR 2014, https://arxiv.org/abs/1312.6120

source

### Other Convenience Functions {#Other-Convenience-Functions}

::: warning Beware

Unlike the other functions these ones don't take a type argument.

:::

```julia
zeros16([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{Float16, length(size)}
```

Return an `AbstractArray{Float16}` of the given `size` containing an AbstractArray of zeros.

source

```julia
ones16([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{Float16, length(size)}
```

Return an `AbstractArray{Float16}` of the given `size` containing an AbstractArray of ones.

source

```julia
rand16([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{Float16, length(size)}
```

Return an `AbstractArray{Float16}` of the given `size` containing random numbers from a uniform distribution.

source

```julia
randn16([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{Float16, length(size)}
```

Return an `AbstractArray{Float16}` of the given `size` containing random numbers from a standard normal distribution.

source

```julia
zeros32([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{Float32, length(size)}
```

Return an `AbstractArray{Float32}` of the given `size` containing an AbstractArray of zeros.

source

```julia
ones32([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{Float32, length(size)}
```

Return an `AbstractArray{Float32}` of the given `size` containing an AbstractArray of ones.

source

```julia
rand32([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{Float32, length(size)}
```

Return an `AbstractArray{Float32}` of the given `size` containing random numbers from a uniform distribution.

source

```julia
randn32([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{Float32, length(size)}
```

Return an `AbstractArray{Float32}` of the given `size` containing random numbers from a standard normal distribution.

source

```julia
zeros64([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{Float64, length(size)}
```

Return an `AbstractArray{Float64}` of the given `size` containing an AbstractArray of zeros.

source

```julia
ones64([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{Float64, length(size)}
```

Return an `AbstractArray{Float64}` of the given `size` containing an AbstractArray of ones.

source

```julia
rand64([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{Float64, length(size)}
```

Return an `AbstractArray{Float64}` of the given `size` containing random numbers from a uniform distribution.

source

```julia
randn64([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{Float64, length(size)}
```

Return an `AbstractArray{Float64}` of the given `size` containing random numbers from a standard normal distribution.

source

```julia
zerosC16([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{ComplexF16, length(size)}
```

Return an `AbstractArray{ComplexF16}` of the given `size` containing an AbstractArray of zeros.

source

```julia
onesC16([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{ComplexF16, length(size)}
```

Return an `AbstractArray{ComplexF16}` of the given `size` containing an AbstractArray of ones.

source

```julia
randC16([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{ComplexF16, length(size)}
```

Return an `AbstractArray{ComplexF16}` of the given `size` containing random numbers from a uniform distribution.

source

```julia
randnC16([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{ComplexF16, length(size)}
```

Return an `AbstractArray{ComplexF16}` of the given `size` containing random numbers from a standard normal distribution.

source

```julia
zerosC32([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{ComplexF32, length(size)}
```

Return an `AbstractArray{ComplexF32}` of the given `size` containing an AbstractArray of zeros.

source

```julia
onesC32([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{ComplexF32, length(size)}
```

Return an `AbstractArray{ComplexF32}` of the given `size` containing an AbstractArray of ones.

source

```julia
randC32([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{ComplexF32, length(size)}
```

Return an `AbstractArray{ComplexF32}` of the given `size` containing random numbers from a uniform distribution.

source

```julia
randnC32([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{ComplexF32, length(size)}
```

Return an `AbstractArray{ComplexF32}` of the given `size` containing random numbers from a standard normal distribution.

source

```julia
zerosC64([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{ComplexF64, length(size)}
```

Return an `AbstractArray{ComplexF64}` of the given `size` containing an AbstractArray of zeros.

source

```julia
onesC64([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{ComplexF64, length(size)}
```

Return an `AbstractArray{ComplexF64}` of the given `size` containing an AbstractArray of ones.

source

```julia
randC64([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{ComplexF64, length(size)}
```

Return an `AbstractArray{ComplexF64}` of the given `size` containing random numbers from a uniform distribution.

source

```julia
randnC64([::AbstractRNG=Utils.default_rng()], size...;
    kwargs...) -> AbstractArray{ComplexF64, length(size)}
```

Return an `AbstractArray{ComplexF64}` of the given `size` containing random numbers from a standard normal distribution.

source
