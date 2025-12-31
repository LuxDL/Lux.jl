
<a id='LuxLib'></a>

# LuxLib


Backend for Lux.jl




<a id='Index'></a>

## Index

- [`LuxLib.alpha_dropout`](#LuxLib.alpha_dropout)
- [`LuxLib.batchnorm`](#LuxLib.batchnorm)
- [`LuxLib.dropout`](#LuxLib.dropout)
- [`LuxLib.groupnorm`](#LuxLib.groupnorm)
- [`LuxLib.instancenorm`](#LuxLib.instancenorm)
- [`LuxLib.layernorm`](#LuxLib.layernorm)


<a id='Dropout'></a>

## Dropout

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxLib.alpha_dropout' href='#LuxLib.alpha_dropout'>#</a>&nbsp;<b><u>LuxLib.alpha_dropout</u></b> &mdash; <i>Function</i>.



```julia
alpha_dropout(rng::AbstractRNG, x, p, ::Val{training})
alpha_dropout(rng::AbstractRNG, x, p, ::Val{training}, α, A, B)
```

Alpha Dropout: Dropout ensuring that the mean and variance of the output remains same as the input. For details see [1]. Use the second call signature to avoid recomputing the constants for a fixed dropout probability.

**Arguments**

  * `rng`: Random number generator
  * `x`: Input Array
  * `p`: Probability of an element to be dropped out
  * `Val(training)`: If `true` then dropout is applied on `x` with probability `p`. Else, `x` is returned
  * `α`: -1.7580993408473766. Computed at limit x tends to infinity, `selu(x) = -λβ = α`
  * `A`: Scaling factor for the mean
  * `B`: Scaling factor for the variance

**Returns**

  * Output Array after applying alpha dropout
  * Updated state for the random number generator

**References**

[1] Klambauer, Günter, et al. "Self-normalizing neural networks." Advances in neural     information processing systems 30 (2017).

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxLib.dropout' href='#LuxLib.dropout'>#</a>&nbsp;<b><u>LuxLib.dropout</u></b> &mdash; <i>Function</i>.



```julia
dropout(rng::AbstractRNG, x, p, ::Val{training}, invp; dims)
dropout(rng::AbstractRNG, x, mask, p, ::Val{training}, ::Val{update_mask}, invp;
        dims)
```

Dropout: Simple Way to prevent Neural Networks for Overfitting. For details see [1].

**Arguments**

  * `rng`: Random number generator
  * `x`: Input Array
  * `mask`: Dropout Mask. If not used then it is constructed automatically
  * `p`: Probability of an element to be dropped out
  * `Val(training)`: If `true` then dropout is applied on `x` with probability `p` along `dims`. Else, `x` is returned
  * `Val(update_mask)`: If `true` then the mask is generated and used. Else, the `mask` provided is directly used
  * `invp`: Inverse of the probability

**Keyword Arguments**

  * `dims`: Dimensions along which dropout is applied
  * `invp`: Inverse of the probability ($\frac{1}{p}$)

**Returns**

  * Output Array after applying dropout
  * Dropout Mask (if `training == false`, the returned value is meaningless)
  * Updated state for the random number generator

**References**

[1] Srivastava, Nitish, et al. "Dropout: a simple way to prevent neural networks from     overfitting." The journal of machine learning research 15.1 (2014): 1929-1958.

</div>
<br>

<a id='Normalization'></a>

## Normalization

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxLib.batchnorm' href='#LuxLib.batchnorm'>#</a>&nbsp;<b><u>LuxLib.batchnorm</u></b> &mdash; <i>Function</i>.



```julia
batchnorm(x, scale, bias, running_mean, running_var; momentum, epsilon, training)
```

Batch Normalization. For details see [1].

Batch Normalization computes the mean and variance for each $D_1 \times ... \times D_{N - 2} \times 1 \times D_N$ input slice and normalises the input accordingly.

**Arguments**

  * `x`: Input to be Normalized
  * `scale`: Scale factor ($\gamma$) (can be `nothing`)
  * `bias`: Bias factor ($\beta$) (can be `nothing`)
  * `running_mean`: Running mean (can be `nothing`)
  * `running_var`: Running variance (can be `nothing`)

**Keyword Arguments**

  * `momentum`: Momentum for updating running mean and variance
  * `epsilon`: Value added to the denominator for numerical stability
  * `training`: Set to `Val(true)` if running in training mode

**Returns**

Normalized Array of same size as `x`. And a Named Tuple containing the updated running mean and variance.

**Performance Considerations**

If the input array is `2D`, `4D`, or `5D` `CuArray` with element types `Float16`, `Float32` and `Float64`, then the CUDNN code path will be used. In all other cases, a broadcasting fallback is used which is not highly optimized.

**References**

[1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network     training by reducing internal covariate shift." International conference on machine     learning. PMLR, 2015.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxLib.groupnorm' href='#LuxLib.groupnorm'>#</a>&nbsp;<b><u>LuxLib.groupnorm</u></b> &mdash; <i>Function</i>.



```julia
groupnorm(x, scale, bias; groups, epsilon)
```

Group Normalization. For details see [1].

This op is similar to batch normalization, but statistics are shared across equally-sized groups of channels and not shared across batch dimension. Thus, group normalization does not depend on the batch composition and does not require maintaining internal state for storing statistics.

**Arguments**

  * `x`: Input to be Normalized
  * `scale`: Scale factor ($\gamma$) (can be `nothing`)
  * `bias`: Bias factor ($\beta$) (can be `nothing`)

**Keyword Arguments**

  * `groups`: Number of groups
  * `epsilon`: Value added to the denominator for numerical stability

**Returns**

The normalized array is returned.

**Performance Considerations**

The most common case of this Op – `x` is a 4D array – is optimized using KernelAbstractions and has a fast custom backwards pass implemented. All other cases have a fallback implementation which is not especially optimized.

We have tested the code path for `Float16` and it works, but gradient accumulation is extremely fragile. Hence, for `Float16` inputs, it uses the fallback implementation.

If the batch size is small (< 16), then the fallback implementation will be faster than the KA version. However, this customization is not possible using the direct `groupnorm` interface.

**References**

[1] Wu, Yuxin, and Kaiming He. "Group normalization." Proceedings of the European conference     on computer vision (ECCV). 2018.

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxLib.instancenorm' href='#LuxLib.instancenorm'>#</a>&nbsp;<b><u>LuxLib.instancenorm</u></b> &mdash; <i>Function</i>.



```julia
instancenorm(x, scale, bias; epsilon, training)
```

Instance Normalization. For details see [1].

Instance Normalization computes the mean and variance for each $D_1 \times ... \times D_{N - 2} \times 1 \times 1$` input slice and normalises the input accordingly.

**Arguments**

  * `x`: Input to be Normalized (must be atleast 3D)
  * `scale`: Scale factor ($\gamma$) (can be `nothing`)
  * `bias`: Bias factor ($\beta$) (can be `nothing`)

**Keyword Arguments**

  * `epsilon`: Value added to the denominator for numerical stability
  * `training`: Set to `Val(true)` if running in training mode

**Returns**

Normalized Array of same size as `x`. And a Named Tuple containing the updated running mean and variance.

**References**

[1] Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance normalization: The     missing ingredient for fast stylization." arXiv preprint arXiv:1607.08022 (2016).

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='LuxLib.layernorm' href='#LuxLib.layernorm'>#</a>&nbsp;<b><u>LuxLib.layernorm</u></b> &mdash; <i>Function</i>.



```julia
layernorm(x, scale, bias; dims, epsilon)
```

Layer Normalization. For details see [1].

Given an input array $x$, this layer computes

$$
y = \frac{x - \mathbb{E}[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta
$$

**Arguments**

  * `x`: Input to be Normalized
  * `scale`: Scale factor ($\gamma$) (can be `nothing`)
  * `bias`: Bias factor ($\beta$) (can be `nothing`)

**Keyword Arguments**

  * `dims`: Dimensions along which the mean and std of `x` is computed
  * `epsilon`: Value added to the denominator for numerical stability

**Returns**

Normalized Array of same size as `x`.

**References**

[1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." arXiv     preprint arXiv:1607.06450 (2016).

</div>
<br>
