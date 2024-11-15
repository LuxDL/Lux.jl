```@meta
CollapsedDocStrings = true
```

# [Activation Functions](@id NNlib-ActivationFunctions-API)

Non-linearities that go between layers of your model. Note that, unless otherwise stated,
activation functions operate on scalars. To apply them to an array you can call `σ.(xs)`,
`relu.(xs)` and so on.

```@docs
celu
elu
gelu
hardsigmoid
sigmoid_fast
hardtanh
tanh_fast
leakyrelu
lisht
logcosh
logsigmoid
mish
relu
relu6
rrelu
selu
sigmoid
softplus
softshrink
softsign
swish
hardswish
tanhshrink
trelu
```
