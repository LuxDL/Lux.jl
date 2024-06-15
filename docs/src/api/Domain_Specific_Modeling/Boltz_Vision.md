# Computer Vision Models (`Vision` API)

## Native Lux Models

```@docs
Vision.VGG
Vision.VisionTransformer
```

## Imported from Metalhead.jl

!!! tip

    You need to load `Flux` and `Metalhead` before using these models.

```@docs
Vision.AlexNet
Vision.ConvMixer
Vision.DenseNet
Vision.GoogLeNet
Vision.MobileNet
Vision.ResNet
Vision.ResNeXt
```

## Pretrained Models

!!! tip

    Pass `pretrained=true` to the model constructor to load the pretrained weights.


| MODEL                     | TOP 1 ACCURACY (%) | TOP 5 ACCURACY (%) |
| :------------------------ | :----------------: | :----------------: |
| `AlexNet()`               |       54.48        |       77.72        |
| `VGG(11)`                 |       67.35        |       87.91        |
| `VGG(13)`                 |       68.40        |       88.48        |
| `VGG(16)`                 |       70.24        |       89.80        |
| `VGG(19)`                 |       71.09        |       90.27        |
| `VGG(11; batchnorm=true)` |       69.09        |       88.94        |
| `VGG(13; batchnorm=true)` |       69.66        |       89.49        |
| `VGG(16; batchnorm=true)` |       72.11        |       91.02        |
| `VGG(19; batchnorm=true)` |       72.95        |       91.32        |

### Preprocessing

All the pretrained models require that the images be normalized with the parameters
`mean = [0.485f0, 0.456f0, 0.406f0]` and `std = [0.229f0, 0.224f0, 0.225f0]`.
