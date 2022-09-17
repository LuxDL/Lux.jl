# Boltz ⚡

[![Join the chat at https://julialang.zulipchat.com #machine-learning](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/machine-learning)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](http://lux.csail.mit.edu/dev/lib/Boltz)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](http://lux.csail.mit.edu/stable/lib/Boltz)

[![CI](https://github.com/avik-pal/Lux.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/avik-pal/Lux.jl/actions/workflows/CI.yml)
[![CI Nightly](https://github.com/avik-pal/Lux.jl/actions/workflows/CINightly.yml/badge.svg)](https://github.com/avik-pal/Lux.jl/actions/workflows/CINightly.yml)
[![codecov](https://codecov.io/gh/avik-pal/Lux.jl/branch/main/graph/badge.svg?token=IMqBM1e3hz)](https://codecov.io/gh/avik-pal/Lux.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/Boltz)](https://pkgs.genieframework.com?packages=Boltz)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

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
| - | - | - | :-: | :-: | :-: |
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
