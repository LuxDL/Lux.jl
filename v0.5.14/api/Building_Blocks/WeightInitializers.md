


<a id='WeightInitializers'></a>

# WeightInitializers


This package is a light dependency providing common weight initialization schemes for deep learning models.


<a id='Index'></a>

## Index

- [`WeightInitializers.glorot_normal`](#WeightInitializers.glorot_normal)
- [`WeightInitializers.glorot_uniform`](#WeightInitializers.glorot_uniform)
- [`WeightInitializers.kaiming_normal`](#WeightInitializers.kaiming_normal)
- [`WeightInitializers.kaiming_uniform`](#WeightInitializers.kaiming_uniform)
- [`WeightInitializers.ones16`](#WeightInitializers.ones16)
- [`WeightInitializers.ones32`](#WeightInitializers.ones32)
- [`WeightInitializers.ones64`](#WeightInitializers.ones64)
- [`WeightInitializers.onesC16`](#WeightInitializers.onesC16)
- [`WeightInitializers.onesC32`](#WeightInitializers.onesC32)
- [`WeightInitializers.onesC64`](#WeightInitializers.onesC64)
- [`WeightInitializers.rand16`](#WeightInitializers.rand16)
- [`WeightInitializers.rand32`](#WeightInitializers.rand32)
- [`WeightInitializers.rand64`](#WeightInitializers.rand64)
- [`WeightInitializers.randC16`](#WeightInitializers.randC16)
- [`WeightInitializers.randC32`](#WeightInitializers.randC32)
- [`WeightInitializers.randC64`](#WeightInitializers.randC64)
- [`WeightInitializers.randn16`](#WeightInitializers.randn16)
- [`WeightInitializers.randn32`](#WeightInitializers.randn32)
- [`WeightInitializers.randn64`](#WeightInitializers.randn64)
- [`WeightInitializers.randnC16`](#WeightInitializers.randnC16)
- [`WeightInitializers.randnC32`](#WeightInitializers.randnC32)
- [`WeightInitializers.randnC64`](#WeightInitializers.randnC64)
- [`WeightInitializers.truncated_normal`](#WeightInitializers.truncated_normal)
- [`WeightInitializers.zeros16`](#WeightInitializers.zeros16)
- [`WeightInitializers.zeros32`](#WeightInitializers.zeros32)
- [`WeightInitializers.zeros64`](#WeightInitializers.zeros64)
- [`WeightInitializers.zerosC16`](#WeightInitializers.zerosC16)
- [`WeightInitializers.zerosC32`](#WeightInitializers.zerosC32)
- [`WeightInitializers.zerosC64`](#WeightInitializers.zerosC64)


<a id='API-Reference'></a>

## API Reference


<a id='Main-Functions'></a>

### Main Functions

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.glorot_normal' href='#WeightInitializers.glorot_normal'>#</a>&nbsp;<b><u>WeightInitializers.glorot_normal</u></b> &mdash; <i>Function</i>.



```julia
glorot_normal([::AbstractRNG=_default_rng()], [T=Float32], size...;
    gain = 1) -> AbstractArray{T, length(size)}
```

Return an `AbstractArray{T}` of the given `size` containing random numbers drawn from a normal distribution with standard deviation `gain * sqrt(2 / (fan_in + fan_out))`. This method is described in [1] and also known as Xavier initialization.

**References**

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." *Proceedings of the thirteenth international conference on artificial intelligence and statistics*. 2010.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.glorot_uniform' href='#WeightInitializers.glorot_uniform'>#</a>&nbsp;<b><u>WeightInitializers.glorot_uniform</u></b> &mdash; <i>Function</i>.



```julia
glorot_uniform([::AbstractRNG=_default_rng()], [T=Float32], size...;
    gain = 1) -> AbstractArray{T, length(size)}
```

Return an `AbstractArray{T}` of the given `size` containing random numbers drawn from a uniform distribution on the interval $[-x, x]$, where `x = gain * sqrt(6 / (fan_in + fan_out))`. This method is described in [1] and also known as Xavier initialization.

**References**

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." *Proceedings of the thirteenth international conference on artificial intelligence and statistics*. 2010.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.kaiming_normal' href='#WeightInitializers.kaiming_normal'>#</a>&nbsp;<b><u>WeightInitializers.kaiming_normal</u></b> &mdash; <i>Function</i>.



```julia
kaiming_normal([::AbstractRNG=_default_rng()], [T=Float32], size...;
    gain = √T(2)) -> AbstractArray{T, length(size)}
```

Return an `AbstractArray{T}` of the given `size` containing random numbers taken from a normal distribution standard deviation `gain / sqrt(fan_in)`

**References**

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." *Proceedings of the IEEE international conference on computer vision*. 2015.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.kaiming_uniform' href='#WeightInitializers.kaiming_uniform'>#</a>&nbsp;<b><u>WeightInitializers.kaiming_uniform</u></b> &mdash; <i>Function</i>.



```julia
kaiming_uniform([::AbstractRNG=_default_rng()], [T=Float32], size...;
    gain = √T(2)) -> AbstractArray{T, length(size)}
```

Return an `AbstractArray{T}` of the given `size` containing random numbers drawn from a uniform distribution on the interval `[-x, x]`, where `x = gain * sqrt(3/fan_in)`.

**References**

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." *Proceedings of the IEEE international conference on computer vision*. 2015.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.truncated_normal' href='#WeightInitializers.truncated_normal'>#</a>&nbsp;<b><u>WeightInitializers.truncated_normal</u></b> &mdash; <i>Function</i>.



```julia
truncated_normal([::AbstractRNG=_default_rng()], [T=Float32], size...; mean = 0,
    std = 1, lo = -2, hi = 2) -> AbstractArray{T, length(size)}
```

Return an `AbstractArray{T}` of the given `size` where each element is drawn from a truncated normal distribution. The numbers are distributed like `filter(x -> lo ≤ x ≤ hi, mean .+ std .* randn(100))`.

</div>
<br>

<a id='Commonly-Used-Wrappers'></a>

### Commonly Used Wrappers

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.zeros16' href='#WeightInitializers.zeros16'>#</a>&nbsp;<b><u>WeightInitializers.zeros16</u></b> &mdash; <i>Function</i>.



```julia
zeros16([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{Float16, length(size)}
```

Return an `AbstractArray{Float16}` of the given `size` containing an AbstractArray of zeros.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.ones16' href='#WeightInitializers.ones16'>#</a>&nbsp;<b><u>WeightInitializers.ones16</u></b> &mdash; <i>Function</i>.



```julia
ones16([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{Float16, length(size)}
```

Return an `AbstractArray{Float16}` of the given `size` containing an AbstractArray of ones.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.rand16' href='#WeightInitializers.rand16'>#</a>&nbsp;<b><u>WeightInitializers.rand16</u></b> &mdash; <i>Function</i>.



```julia
rand16([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{Float16, length(size)}
```

Return an `AbstractArray{Float16}` of the given `size` containing random numbers from a uniform distribution.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.randn16' href='#WeightInitializers.randn16'>#</a>&nbsp;<b><u>WeightInitializers.randn16</u></b> &mdash; <i>Function</i>.



```julia
randn16([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{Float16, length(size)}
```

Return an `AbstractArray{Float16}` of the given `size` containing random numbers from a standard normal distribution.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.zeros32' href='#WeightInitializers.zeros32'>#</a>&nbsp;<b><u>WeightInitializers.zeros32</u></b> &mdash; <i>Function</i>.



```julia
zeros32([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{Float32, length(size)}
```

Return an `AbstractArray{Float32}` of the given `size` containing an AbstractArray of zeros.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.ones32' href='#WeightInitializers.ones32'>#</a>&nbsp;<b><u>WeightInitializers.ones32</u></b> &mdash; <i>Function</i>.



```julia
ones32([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{Float32, length(size)}
```

Return an `AbstractArray{Float32}` of the given `size` containing an AbstractArray of ones.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.rand32' href='#WeightInitializers.rand32'>#</a>&nbsp;<b><u>WeightInitializers.rand32</u></b> &mdash; <i>Function</i>.



```julia
rand32([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{Float32, length(size)}
```

Return an `AbstractArray{Float32}` of the given `size` containing random numbers from a uniform distribution.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.randn32' href='#WeightInitializers.randn32'>#</a>&nbsp;<b><u>WeightInitializers.randn32</u></b> &mdash; <i>Function</i>.



```julia
randn32([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{Float32, length(size)}
```

Return an `AbstractArray{Float32}` of the given `size` containing random numbers from a standard normal distribution.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.zeros64' href='#WeightInitializers.zeros64'>#</a>&nbsp;<b><u>WeightInitializers.zeros64</u></b> &mdash; <i>Function</i>.



```julia
zeros64([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{Float64, length(size)}
```

Return an `AbstractArray{Float64}` of the given `size` containing an AbstractArray of zeros.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.ones64' href='#WeightInitializers.ones64'>#</a>&nbsp;<b><u>WeightInitializers.ones64</u></b> &mdash; <i>Function</i>.



```julia
ones64([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{Float64, length(size)}
```

Return an `AbstractArray{Float64}` of the given `size` containing an AbstractArray of ones.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.rand64' href='#WeightInitializers.rand64'>#</a>&nbsp;<b><u>WeightInitializers.rand64</u></b> &mdash; <i>Function</i>.



```julia
rand64([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{Float64, length(size)}
```

Return an `AbstractArray{Float64}` of the given `size` containing random numbers from a uniform distribution.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.randn64' href='#WeightInitializers.randn64'>#</a>&nbsp;<b><u>WeightInitializers.randn64</u></b> &mdash; <i>Function</i>.



```julia
randn64([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{Float64, length(size)}
```

Return an `AbstractArray{Float64}` of the given `size` containing random numbers from a standard normal distribution.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.zerosC16' href='#WeightInitializers.zerosC16'>#</a>&nbsp;<b><u>WeightInitializers.zerosC16</u></b> &mdash; <i>Function</i>.



```julia
zerosC16([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{ComplexF16, length(size)}
```

Return an `AbstractArray{ComplexF16}` of the given `size` containing an AbstractArray of zeros.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.onesC16' href='#WeightInitializers.onesC16'>#</a>&nbsp;<b><u>WeightInitializers.onesC16</u></b> &mdash; <i>Function</i>.



```julia
onesC16([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{ComplexF16, length(size)}
```

Return an `AbstractArray{ComplexF16}` of the given `size` containing an AbstractArray of ones.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.randC16' href='#WeightInitializers.randC16'>#</a>&nbsp;<b><u>WeightInitializers.randC16</u></b> &mdash; <i>Function</i>.



```julia
randC16([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{ComplexF16, length(size)}
```

Return an `AbstractArray{ComplexF16}` of the given `size` containing random numbers from a uniform distribution.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.randnC16' href='#WeightInitializers.randnC16'>#</a>&nbsp;<b><u>WeightInitializers.randnC16</u></b> &mdash; <i>Function</i>.



```julia
randnC16([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{ComplexF16, length(size)}
```

Return an `AbstractArray{ComplexF16}` of the given `size` containing random numbers from a standard normal distribution.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.zerosC32' href='#WeightInitializers.zerosC32'>#</a>&nbsp;<b><u>WeightInitializers.zerosC32</u></b> &mdash; <i>Function</i>.



```julia
zerosC32([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{ComplexF32, length(size)}
```

Return an `AbstractArray{ComplexF32}` of the given `size` containing an AbstractArray of zeros.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.onesC32' href='#WeightInitializers.onesC32'>#</a>&nbsp;<b><u>WeightInitializers.onesC32</u></b> &mdash; <i>Function</i>.



```julia
onesC32([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{ComplexF32, length(size)}
```

Return an `AbstractArray{ComplexF32}` of the given `size` containing an AbstractArray of ones.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.randC32' href='#WeightInitializers.randC32'>#</a>&nbsp;<b><u>WeightInitializers.randC32</u></b> &mdash; <i>Function</i>.



```julia
randC32([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{ComplexF32, length(size)}
```

Return an `AbstractArray{ComplexF32}` of the given `size` containing random numbers from a uniform distribution.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.randnC32' href='#WeightInitializers.randnC32'>#</a>&nbsp;<b><u>WeightInitializers.randnC32</u></b> &mdash; <i>Function</i>.



```julia
randnC32([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{ComplexF32, length(size)}
```

Return an `AbstractArray{ComplexF32}` of the given `size` containing random numbers from a standard normal distribution.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.zerosC64' href='#WeightInitializers.zerosC64'>#</a>&nbsp;<b><u>WeightInitializers.zerosC64</u></b> &mdash; <i>Function</i>.



```julia
zerosC64([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{ComplexF64, length(size)}
```

Return an `AbstractArray{ComplexF64}` of the given `size` containing an AbstractArray of zeros.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.onesC64' href='#WeightInitializers.onesC64'>#</a>&nbsp;<b><u>WeightInitializers.onesC64</u></b> &mdash; <i>Function</i>.



```julia
onesC64([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{ComplexF64, length(size)}
```

Return an `AbstractArray{ComplexF64}` of the given `size` containing an AbstractArray of ones.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.randC64' href='#WeightInitializers.randC64'>#</a>&nbsp;<b><u>WeightInitializers.randC64</u></b> &mdash; <i>Function</i>.



```julia
randC64([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{ComplexF64, length(size)}
```

Return an `AbstractArray{ComplexF64}` of the given `size` containing random numbers from a uniform distribution.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.randnC64' href='#WeightInitializers.randnC64'>#</a>&nbsp;<b><u>WeightInitializers.randnC64</u></b> &mdash; <i>Function</i>.



```julia
randnC64([::AbstractRNG=_default_rng()], size...; kwargs...) -> AbstractArray{ComplexF64, length(size)}
```

Return an `AbstractArray{ComplexF64}` of the given `size` containing random numbers from a standard normal distribution.

</div>
<br>