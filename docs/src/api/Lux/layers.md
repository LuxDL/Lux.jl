```@meta
CollapsedDocStrings = true
```

# Built-In Layers

## Containers

```@docs
BranchLayer
Chain
PairwiseFusion
Parallel
SkipConnection
RepeatedLayer
AlternatePrecision
```

## Convolutional Layers

```@docs
Conv
ConvTranspose
```

## Dropout Layers

```@docs
AlphaDropout
Dropout
VariationalHiddenDropout
```

## Pooling Layers

```@docs
AdaptiveLPPool
AdaptiveMaxPool
AdaptiveMeanPool
GlobalLPPool
GlobalMaxPool
GlobalMeanPool
LPPool
MaxPool
MeanPool
```

## Recurrent Layers

```@docs
GRUCell
LSTMCell
RNNCell
Recurrence
StatefulRecurrentCell
BidirectionalRNN
```

## Linear Layers

```@docs
Bilinear
Dense
Scale
```

## Attention Layers

```@docs
MultiHeadAttention
```

## Embedding Layers

```@docs
Embedding
RotaryPositionalEmbedding
SinusoidalPositionalEmbedding
```

### Functional API

```@docs
apply_rotary_embedding
compute_rotary_embedding_params
```

## Misc. Helper Layers

```@docs
FlattenLayer
Maxout
NoOpLayer
ReshapeLayer
SelectDim
WrappedFunction
ReverseSequence
```

## Normalization Layers

```@docs
BatchNorm
GroupNorm
InstanceNorm
LayerNorm
WeightNorm
RMSNorm
```

## Upsampling

```@docs
PixelShuffle
Upsample
```
