# Boltz ⚡

Accelerate ⚡ your ML research using pre-built Deep Learning Models with Lux

## Installation

```julia
using Pkg
Pkg.add("Boltz")
```

## Getting Started

```julia
using Boltz, Lux

model, ps, st = resnet(:resnet18; pretrained=true)
```


## Classification Models

| MODEL NAME | FUNCTION | NAME | PRETRAINED | TOP 1 ACCURACY (%) | TOP 5 ACCURACY (%) |
| ---------- | -------- | ---- | ---------- | ------------------ | ------------------ |
| AlexNet | `alexnet` | `:alexnet` | ✅ | 54.48 | 77.72 |
| ResNet | `resnet` | `:resnet18` | ✅ | 68.08 | 88.44 |
| ResNet | `resnet` | `:resnet34` | ✅ | 72.13 | 90.91 |
| ResNet | `resnet` | `:resnet50` | ✅ | 74.55 | 92.36 |
| ResNet | `resnet` | `:resnet101` | ✅ | 74.81 | 92.36 |
| ResNet | `resnet` | `:resnet152` | ✅ | 77.63 | 93.84 |
| VGG | `vgg` | `:vgg11` | ✅ | 67.35 | 87.91 |
| VGG | `vgg` | `:vgg13` | ✅ | 68.40 | 88.48 |
| VGG | `vgg` | `:vgg16` | ✅ | 70.24 | 89.80 |
| VGG | `vgg` | `:vgg19` | ✅ | 71.09 | 90.27 |
| VGG | `vgg` | `:vgg11_bn` | ✅ | 69.09 | 88.94 |
| VGG | `vgg` | `:vgg13_bn` | ✅ | 69.66 | 89.49 |
| VGG | `vgg` | `:vgg16_bn` | ✅ | 72.11 | 91.02 |
| VGG | `vgg` | `:vgg19_bn` | ✅ | 72.95 | 91.32 |
| ConvMixer | `convmixer` | `:small` | 🚫 | | |
| ConvMixer | `convmixer` | `:base` | 🚫 | | |
| ConvMixer | `convmixer` | `:large` | 🚫 | | |
| DenseNet | `densenet` | `:densenet121` | 🚫 | | |
| DenseNet | `densenet` | `:densenet161` | 🚫 | | |
| DenseNet | `densenet` | `:densenet169` | 🚫 | | |
| DenseNet | `densenet` | `:densenet201` | 🚫 | | |
| GoogleNet | `googlenet` | `:googlenet` | 🚫 | | |
| MobileNet | `mobilenet` | `:mobilenet_v1` | 🚫 | | |
| MobileNet | `mobilenet` | `:mobilenet_v2` | 🚫 | | |
| MobileNet | `mobilenet` | `:mobilenet_v3_small` | 🚫 | | |
| MobileNet | `mobilenet` | `:mobilenet_v3_large` | 🚫 | | |
| ResNeXT | `resnext` | `:resnext50` | 🚫 | | |
| ResNeXT | `resnext` | `:resnext101` | 🚫 | | |
| ResNeXT | `resnext` | `:resnext152` | 🚫 | | |
| Vision Transformer | `vision_transformer` | `:tiny` | 🚫 | | |
| Vision Transformer | `vision_transformer` | `:small` | 🚫 | | |
| Vision Transformer | `vision_transformer` | `:base` | 🚫 | | |
| Vision Transformer | `vision_transformer` | `:large` | 🚫 | | |
| Vision Transformer | `vision_transformer` | `:huge` | 🚫 | | |
| Vision Transformer | `vision_transformer` | `:giant` | 🚫 | | |
| Vision Transformer | `vision_transformer` | `:gigantic` | 🚫 | | |

These models can be created using `<FUNCTION>(<NAME>; pretrained = <PRETRAINED>)`

### Preprocessing

All the pretrained models require that the images be normalized with the parameters
`mean = [0.485f0, 0.456f0, 0.406f0]` and `std = [0.229f0, 0.224f0, 0.225f0]`.
