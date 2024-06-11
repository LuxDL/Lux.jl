```@meta
CurrentModule = Boltz
```

# `Boltz.Layers` and `Boltz.Basis` API Reference

## `Layers` API

```@docs
Layers.ConvBatchNormActivation
Layers.ConvNormActivation
Layers.ClassTokens
Layers.HamiltonianNN
Layers.MultiHeadSelfAttention
Layers.MLP
Layers.TensorProductLayer
Layers.ViPosEmbedding
Layers.VisionTransformerEncoder
```

## Basis Functions

!!! warning

    The function calls for these basis functions should be considered experimental and are
    subject to change without deprecation. However, the functions themselves are stable
    and can be freely used in combination with the other Layers and Models.

```@docs
Basis.Cos
Basis.Chebychev
Basis.Fourier
Basis.Legendre
Basis.Polynomial
Basis.Sin
```
