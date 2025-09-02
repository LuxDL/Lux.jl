```@meta
CollapsedDocStrings = true
CurrentModule = LuxLib
```

# [LuxLib](@id LuxLib-API)

Backend for Lux.jl

## Apply Activation

```@docs
fast_activation
fast_activation!!
```

## Attention

```@docs
scaled_dot_product_attention
```

## Batched Operations

```@docs
batched_matmul
```

## Bias Activation

```@docs
bias_activation
bias_activation!!
```

## Convolutional Layers

```@docs
fused_conv_bias_activation
```

## Dropout

```@docs
alpha_dropout
dropout
```

## Fully Connected Layers

```@docs
fused_dense_bias_activation
```

## Normalization

```@docs
batchnorm
groupnorm
instancenorm
layernorm
```

## Helper Functions

```@docs
LuxLib.internal_operation_mode
```
