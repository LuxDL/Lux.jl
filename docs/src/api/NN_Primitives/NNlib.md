```@meta
CollapsedDocStrings = true
```

# [NNlib](@id NNlib-API)

Neural Network Primitives with custom bindings for different accelerator backends in Julia.

!!! note "Reexport of `NNlib`"

    Lux doesn't re-export all of `NNlib` for now. Directly loading `NNlib` is the
    recommended appraoch for accessing these functions.

## Attention 

```@docs
dot_product_attention
dot_product_attention_scores
make_causal_mask
```

## Softmax

```@docs
softmax
logsoftmax
```

## Pooling

```@docs
PoolDims
maxpool
meanpool
lpnormpool
```

## Padding

```@docs
pad_reflect
pad_symmetric
pad_circular
pad_repeat
pad_constant
pad_zeros
```

## Convolution

```@docs
conv
ConvDims
depthwiseconv
DepthwiseConvDims
DenseConvDims
Lux.NNlib.unfold
Lux.NNlib.fold
```

## Upsampling

```@docs
upsample_nearest
∇upsample_nearest
upsample_linear
∇upsample_linear
upsample_bilinear
∇upsample_bilinear
upsample_trilinear
∇upsample_trilinear
pixel_shuffle
```

## Rotation
Rotate images in the first two dimensions of an array.

```@docs
imrotate
∇imrotate
```

## Batched Operations

```@docs
batched_mul
batched_mul!
batched_adjoint
batched_transpose
batched_vec
```

## Gather and Scatter

```@docs
Lux.NNlib.gather
Lux.NNlib.gather!
Lux.NNlib.scatter
Lux.NNlib.scatter!
```

## Sampling

```@docs
grid_sample
∇grid_sample
```

## Losses

```@docs
ctc_loss
```

## Miscellaneous

```@docs
logsumexp
glu
```

!!! tip

    `within_gradient` function currently doesn't work for Enzyme. Prefer to use
    `LuxLib.Utils.within_autodiff` if needed. Though pay heed that this function is
    not part of the public API.

```@docs
Lux.NNlib.within_gradient
```

!!! tip

    Use [`LuxLib.bias_activation!!`](@ref) or [`LuxLib.bias_activation`](@ref) instead of
    `NNlib.bias_act!`.

```@docs
bias_act!
```
