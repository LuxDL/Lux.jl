```@meta
CurrentModule = Boltz
```

# Boltz

Accelerate ⚡ your ML research using pre-built Deep Learning Models with Lux.


## Index

```@index
Pages = ["Boltz.md"]
```

# Computer Vision Models

## Classification Models: Native Lux Models

| MODEL NAME         | FUNCTION             | NAME        | PRETRAINED | TOP 1 ACCURACY (%) | TOP 5 ACCURACY (%) |
| ------------------ | -------------------- | ----------- | :--------: | :----------------: | :----------------: |
| VGG                | `vgg`                | `:vgg11`    |     Y      |       67.35        |       87.91        |
| VGG                | `vgg`                | `:vgg13`    |     Y      |       68.40        |       88.48        |
| VGG                | `vgg`                | `:vgg16`    |     Y      |       70.24        |       89.80        |
| VGG                | `vgg`                | `:vgg19`    |     Y      |       71.09        |       90.27        |
| VGG                | `vgg`                | `:vgg11_bn` |     Y      |       69.09        |       88.94        |
| VGG                | `vgg`                | `:vgg13_bn` |     Y      |       69.66        |       89.49        |
| VGG                | `vgg`                | `:vgg16_bn` |     Y      |       72.11        |       91.02        |
| VGG                | `vgg`                | `:vgg19_bn` |     Y      |       72.95        |       91.32        |
| Vision Transformer | `vision_transformer` | `:tiny`     |     N      |                    |                    |
| Vision Transformer | `vision_transformer` | `:small`    |     N      |                    |                    |
| Vision Transformer | `vision_transformer` | `:base`     |     N      |                    |                    |
| Vision Transformer | `vision_transformer` | `:large`    |     N      |                    |                    |
| Vision Transformer | `vision_transformer` | `:huge`     |     N      |                    |                    |
| Vision Transformer | `vision_transformer` | `:giant`    |     N      |                    |                    |
| Vision Transformer | `vision_transformer` | `:gigantic` |     N      |                    |                    |

## Building Blocks

```@docs
Boltz.ClassTokens
Boltz.MultiHeadAttention
Boltz.ViPosEmbedding
Boltz.transformer_encoder
Boltz.vgg
```

### Non-Public API

```@docs
Boltz._seconddimmean
Boltz._fast_chunk
Boltz._flatten_spatial
Boltz._vgg_block
Boltz._vgg_classifier_layers
Boltz._vgg_convolutional_layers
```

## Classification Models: Imported from Metalhead.jl

!!! tip

    You need to load `Flux` and `Metalhead` before using these models.


| MODEL NAME | FUNCTION    | NAME                  | PRETRAINED | TOP 1 ACCURACY (%) | TOP 5 ACCURACY (%) |
| ---------- | ----------- | --------------------- | :--------: | :----------------: | :----------------: |
| AlexNet    | `alexnet`   | `:alexnet`            |     N      |       54.48        |       77.72        |
| ResNet     | `resnet`    | `:resnet18`           |     N      |       68.08        |       88.44        |
| ResNet     | `resnet`    | `:resnet34`           |     N      |       72.13        |       90.91        |
| ResNet     | `resnet`    | `:resnet50`           |     N      |       74.55        |       92.36        |
| ResNet     | `resnet`    | `:resnet101`          |     N      |       74.81        |       92.36        |
| ResNet     | `resnet`    | `:resnet152`          |     N      |       77.63        |       93.84        |
| ConvMixer  | `convmixer` | `:small`              |     N      |                    |                    |
| ConvMixer  | `convmixer` | `:base`               |     N      |                    |                    |
| ConvMixer  | `convmixer` | `:large`              |     N      |                    |                    |
| DenseNet   | `densenet`  | `:densenet121`        |     N      |                    |                    |
| DenseNet   | `densenet`  | `:densenet161`        |     N      |                    |                    |
| DenseNet   | `densenet`  | `:densenet169`        |     N      |                    |                    |
| DenseNet   | `densenet`  | `:densenet201`        |     N      |                    |                    |
| GoogleNet  | `googlenet` | `:googlenet`          |     N      |                    |                    |
| MobileNet  | `mobilenet` | `:mobilenet_v1`       |     N      |                    |                    |
| MobileNet  | `mobilenet` | `:mobilenet_v2`       |     N      |                    |                    |
| MobileNet  | `mobilenet` | `:mobilenet_v3_small` |     N      |                    |                    |
| MobileNet  | `mobilenet` | `:mobilenet_v3_large` |     N      |                    |                    |
| ResNeXT    | `resnext`   | `:resnext50`          |     N      |                    |                    |
| ResNeXT    | `resnext`   | `:resnext101`         |     N      |                    |                    |
| ResNeXT    | `resnext`   | `:resnext152`         |     N      |                    |                    |

These models can be created using `<FUNCTION>(<NAME>; pretrained = <PRETRAINED>)`

### Preprocessing

All the pretrained models require that the images be normalized with the parameters
`mean = [0.485f0, 0.456f0, 0.406f0]` and `std = [0.229f0, 0.224f0, 0.225f0]`.