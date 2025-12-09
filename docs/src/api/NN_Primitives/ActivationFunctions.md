```@meta
CollapsedDocStrings = true
CurrentModule = NNlib
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
NNlib.hardσ
sigmoid_fast
hardtanh
tanh_fast
leakyrelu
lisht
logcosh
logsigmoid
NNlib.logσ
mish
relu
relu6
rrelu
selu
sigmoid
NNlib.σ
softplus
softshrink
softsign
swish
hardswish
tanhshrink
trelu
gelu_tanh
gelu_sigmoid
gelu_erf
```
