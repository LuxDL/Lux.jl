# Imagenet Training using ExplicitFluxLayers

This implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset.

## Requirements

* Install [julia](https://julialang.org/)
* In the Julia REPL instantiate the `Project.toml` in the parent directory
* Download the ImageNet dataset from http://www.image-net.org/
  - Then, move and extract the training and validation images to labeled subfolders, using [the following shell script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh)

## Training

To train a model, run `main.jl` with the desired model architecture and the path to the ImageNet dataset:

```bash
julia main.py -a ResNet18 [imagenet-folder with train and val folders]
```

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. This is appropriate for ResNet and models with batch normalization, but too high for AlexNet and VGG. Use 0.01 as the initial learning rate for AlexNet or VGG:

```bash
julia main.jl -a AlexNet --learning-rate 0.01 [imagenet-folder with train and val folders]
```

## Distributed Data Parallel Training

Ensure that you have a CUDA-Aware MPI Installed (else communication might severely bottleneck training) and [MPI.jl](https://juliaparallel.org/MPI.jl/stable/usage/#CUDA-aware-MPI-support) is aware of this build. Apart from this run the script using `mpiexecjl`.


## Usage

```bash
usage: main.jl [--arch ARCH] [--epochs EPOCHS]
               [--start-epoch START-EPOCH] [--batch-size BATCH-SIZE]
               [--learning-rate LEARNING-RATE] [--momentum MOMENTUM]
               [--weight-decay WEIGHT-DECAY] [--print-freq PRINT-FREQ]
               [--resume RESUME] [--evaluate] [--pretrained]
               [--seed SEED] [--distributed] [-h] data

ExplicitFluxLayers ImageNet Training

positional arguments:
  data                  path to dataset

optional arguments:
  --arch ARCH           model architectures: VGG19, ResNet50,
                        GoogLeNet, ResNeXt152, DenseNet201,
                        MobileNetv3_small, ResNet34, ResNet18,
                        DenseNet121, ResNet101, VGG13_BN, DenseNet169,
                        MobileNetv1, VGG11_BN, DenseNet161,
                        MobileNetv3_large, VGG11, VGG19_BN, VGG16_BN,
                        VGG16, ResNeXt50, AlexNet, VGG13, ResNeXt101,
                        MobileNetv2, ConvMixer or ResNet152 (default:
                        "ResNet18")
  --epochs EPOCHS       number of total epochs to run (type: Int64,
                        default: 90)
  --start-epoch START-EPOCH
                        manual epoch number (useful on restarts)
                        (type: Int64, default: 0)
  --batch-size BATCH-SIZE
                        mini-batch size, this is the total batch size
                        across all GPUs (type: Int64, default: 256)
  --learning-rate LEARNING-RATE
                        initial learning rate (type: Float32, default:
                        0.1)
  --momentum MOMENTUM   momentum (type: Float32, default: 0.9)
  --weight-decay WEIGHT-DECAY
                        weight decay (type: Float32, default: 0.0001)
  --print-freq PRINT-FREQ
                        print frequency (type: Int64, default: 10)
  --resume RESUME       resume from checkpoint (default: "")
  --evaluate            evaluate model on validation set
  --pretrained          use pre-trained model
  --seed SEED           seed for initializing training.  (type: Int64,
                        default: 0)
  -h, --help            show this help message and exit
```