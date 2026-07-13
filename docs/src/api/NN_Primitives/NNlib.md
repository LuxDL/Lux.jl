```@meta
CollapsedDocStrings = true
CurrentModule = NNlib
```

# [NNlib](@id NNlib-API)

Neural Network Primitives with custom bindings for different accelerator backends in Julia.

!!! note "Reexport of `NNlib`"

    Lux doesn't re-export all of `NNlib` for now. Directly loading `NNlib` is the
    recommended approach for accessing these functions.

## Attention

```@docs
dot_product_attention
dot_product_attention_scores
make_causal_mask
```

## Softmax

```@docs
softmax
softmax!
logsoftmax
logsoftmax!
∇softmax
∇softmax!
∇logsoftmax
∇logsoftmax!
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
NNlib.unfold
NNlib.fold
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
NNlib.gather
NNlib.gather!
NNlib.scatter
NNlib.scatter!
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
NNlib.@disallow_spawns
```

!!! tip

    `within_gradient` function currently doesn't work for Enzyme. Prefer to use
    `LuxLib.Utils.within_autodiff` if needed. Though pay heed that this function is
    not part of the public API.

```@docs
NNlib.within_gradient
```

!!! tip

    Use `LuxLib.API.bias_activation!!` or `LuxLib.API.bias_activation` instead of
    `NNlib.bias_act!`.

```@docs
bias_act!
```

## Dropout

!!! tip

    Use `LuxLib.API.dropout` instead of `NNlib.dropout`.

```@docs
NNlib.dropout
NNlib.dropout!
```

## Internal NNlib Functions

These functions are not part of the public API and are subject to change without notice.

```@docs
NNlib.BatchedAdjoint
NNlib.∇conv_filter_direct!
NNlib._check_trivial_rotations!
NNlib.fast_act
NNlib.spectrogram
NNlib.is_strided
NNlib.conv_direct!
NNlib.gemm!
NNlib.calc_padding_regions
NNlib._prepare_imrotate
NNlib.insert_singleton_spatial_dimension
NNlib._fast_broadcast!
NNlib.hann_window
NNlib._rng_from_array
NNlib.istft
NNlib.transpose_swapbatch
NNlib.transpose_pad
NNlib.power_to_db
NNlib.col2im!
NNlib.storage_type
NNlib.im2col_dims
NNlib.reverse_indices
NNlib.∇conv_filter_im2col!
NNlib.conv_im2col!
NNlib.∇conv_data_direct!
NNlib.scatter_dims
NNlib.∇conv_data_im2col!
NNlib.storage_typejoin
NNlib.add_blanks
NNlib.∇filter_im2col_dims
NNlib._bilinear_helper
NNlib._triangular_filterbanks
NNlib.db_to_power
NNlib.predilated_size
NNlib.stft
NNlib.hamming_window
NNlib.maximum_dims
NNlib.BatchedTranspose
NNlib._rotate_coordinates
NNlib.melscale_filterbanks
NNlib.logaddexp
NNlib.im2col!
NNlib.predilate
NNlib.safe_div
```
