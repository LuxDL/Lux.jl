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

| MODEL NAME | FUNCTION | NAME | PRETRAINED | TOP 5 ACCURACY (%) | TOP 1 ACCURACY (%) |
| - | - | - | :-: | :-: | :-: |
| AlexNet | `alexnet` | `:alexnet` | ✅ | 54.48 | 77.72 |
| ResNet | `resnet` | `:resnet18` | ✅ | 68.08 | 88.44 |
| ResNet | `resnet` | `:resnet34` | ✅ | 72.13 | 90.91 |
| ResNet | `resnet` | `:resnet50` | ✅ | 74.55 | 92.36 |
| ResNet | `resnet` | `:resnet101` | ✅ | 74.81 | 92.36 |
| ResNet | `resnet` | `:resnet152` | ✅ | 77.63 | 93.84 |
| VGG | `vgg` | `:vgg11` | ✅ | | |
| VGG | `vgg` | `:vgg11_bn` | ✅ | | |
| VGG | `vgg` | `:vgg13` | ✅ | | |
| VGG | `vgg` | `:vgg13_bn` | ✅ | | |
| VGG | `vgg` | `:vgg16` | ✅ | | |
| VGG | `vgg` | `:vgg16_bn` | ✅ | | |
| VGG | `vgg` | `:vgg19` | ✅ | | |
| VGG | `vgg` | `:vgg19_bn` | ✅ | | |
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

### Preprocessing

All the pretrained models require that the images be normalized with the parameters `mean = [0.485f0, 0.456f0, 0.406f0]` and `std = [0.229f0, 0.224f0, 0.225f0]`.