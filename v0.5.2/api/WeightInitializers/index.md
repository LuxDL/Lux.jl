


<a id='WeightInitializers'></a>

# WeightInitializers


This package is a light dependency providing common weight initialization schemes for deep learning models.


<a id='Index'></a>

## Index

- [`WeightInitializers.glorot_normal`](#WeightInitializers.glorot_normal)
- [`WeightInitializers.glorot_uniform`](#WeightInitializers.glorot_uniform)
- [`WeightInitializers.kaiming_normal`](#WeightInitializers.kaiming_normal)
- [`WeightInitializers.kaiming_uniform`](#WeightInitializers.kaiming_uniform)
- [`WeightInitializers.ones32`](#WeightInitializers.ones32)
- [`WeightInitializers.rand32`](#WeightInitializers.rand32)
- [`WeightInitializers.randn32`](#WeightInitializers.randn32)
- [`WeightInitializers.truncated_normal`](#WeightInitializers.truncated_normal)
- [`WeightInitializers.zeros32`](#WeightInitializers.zeros32)


<a id='API Reference'></a>

## API Reference

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.zeros32' href='#WeightInitializers.zeros32'>#</a>&nbsp;<b><u>WeightInitializers.zeros32</u></b> &mdash; <i>Function</i>.



```julia
zeros32(::AbstractRNG, size...) = zeros(Float32, size...)
```

Return an `Array{Float32}` of zeros of the given `size`. (`rng` is ignored)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.ones32' href='#WeightInitializers.ones32'>#</a>&nbsp;<b><u>WeightInitializers.ones32</u></b> &mdash; <i>Function</i>.



```julia
ones32(::AbstractRNG, size...) = ones(Float32, size...)
```

Return an `Array{Float32}` of ones of the given `size`. (`rng` is ignored)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.rand32' href='#WeightInitializers.rand32'>#</a>&nbsp;<b><u>WeightInitializers.rand32</u></b> &mdash; <i>Function</i>.



```julia
rand32(rng::AbstractRNG, size...) = rand(rng, Float32, size...)
```

Return an `Array{Float32}` of random numbers from a uniform distribution of the given `size`.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.randn32' href='#WeightInitializers.randn32'>#</a>&nbsp;<b><u>WeightInitializers.randn32</u></b> &mdash; <i>Function</i>.



```julia
randn32(rng::AbstractRNG, size...) = randn(rng, Float32, size...)
```

Return an `Array{Float32}` of random numbers from a standard normal distribution of the given `size`.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.glorot_normal' href='#WeightInitializers.glorot_normal'>#</a>&nbsp;<b><u>WeightInitializers.glorot_normal</u></b> &mdash; <i>Function</i>.



```julia
glorot_normal(rng::AbstractRNG, size...; gain = 1)
```

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a normal distribution with standard deviation `gain * sqrt(2 / (fan_in + fan_out))`. This method is described in [1] and also known as Xavier initialization.

**References**

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." *Proceedings of the thirteenth international conference on artificial intelligence and statistics*. 2010.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.glorot_uniform' href='#WeightInitializers.glorot_uniform'>#</a>&nbsp;<b><u>WeightInitializers.glorot_uniform</u></b> &mdash; <i>Function</i>.



```julia
glorot_uniform(rng::AbstractRNG, size...; gain = 1)
```

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a uniform distribution on the interval $[-x, x]$, where `x = gain * sqrt(6 / (fan_in + fan_out))`. This method is described in [1] and also known as Xavier initialization.

**References**

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." *Proceedings of the thirteenth international conference on artificial intelligence and statistics*. 2010.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.kaiming_normal' href='#WeightInitializers.kaiming_normal'>#</a>&nbsp;<b><u>WeightInitializers.kaiming_normal</u></b> &mdash; <i>Function</i>.



```julia
kaiming_normal(rng::AbstractRNG, size...; gain = √2f0)
```

Return an `Array{Float32}` of the given `size` containing random numbers taken from a normal distribution standard deviation `gain / sqrt(fan_in)`

**References**

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." *Proceedings of the IEEE international conference on computer vision*. 2015.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.kaiming_uniform' href='#WeightInitializers.kaiming_uniform'>#</a>&nbsp;<b><u>WeightInitializers.kaiming_uniform</u></b> &mdash; <i>Function</i>.



```julia
kaiming_uniform(rng::AbstractRNG, size...; gain = √2f0)
```

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a uniform distribution on the interval `[-x, x]`, where `x = gain * sqrt(3/fan_in)`.

**References**

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." *Proceedings of the IEEE international conference on computer vision*. 2015.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='WeightInitializers.truncated_normal' href='#WeightInitializers.truncated_normal'>#</a>&nbsp;<b><u>WeightInitializers.truncated_normal</u></b> &mdash; <i>Function</i>.



```julia
truncated_normal([rng = default_rng_value()], size...; mean = 0, std = 1, lo = -2, hi = 2)
```

Return an `Array{Float32}` of the given `size` where each element is drawn from a truncated normal distribution. The numbers are distributed like `filter(x -> lo<=x<=hi, mean .+ std .* randn(100))`.

</div>
<br>
