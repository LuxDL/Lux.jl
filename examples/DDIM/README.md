# Denoising Diffusion Implicit Model (DDIM)

[Lux.jl](https://github.com/LuxDL/Lux.jl) implementation of Denoising Diffusion Implicit
Models ([arXiv:2010.02502](https://arxiv.org/abs/2010.02502)). The implementation follows
[the Keras example](https://keras.io/examples/generative/ddim/).

The model generates images from Gaussian noises by denoising iteratively.

![generated flowers](./assets/flowers_generated.png)

## Usage

Install Julia and instantiate `Project.toml`.

Following scripts are tested on a single NVIDIA RTX 5090 with 32GB of GPU. You
may need to adjust the image size, batch size and learning rate for your environment.

### Dataset

We use the dataset from [102 Category Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
The user is prompted to download the dataset when running the code for the first time.
The dataset is cached for subsequent runs.

### Training

```bash
julia --startup-file=no \
    --project=examples/DDIM \
    --threads=auto \
    examples/DDIM/main.jl \
    --expt-dir output
```

This code runs in about 20 mins on a single NVIDIA RTX 5090 with 32GB of GPU
memory. We recommend running the code for atleast 80 epochs to get good results.

### Image generation

```bash
julia --startup-file=no \
    --project=examples/DDIM \
    --threads=auto \
    examples/DDIM/main.jl \
    --inference-mode \
    --saved-model-path output/checkpoints/model_100.jld2 \
    --generate-n-images 24 \
    --expt-dir output
```

## Extended CLI options

```bash
usage: main.jl [--epochs EPOCHS] [--image-size IMAGE-SIZE]
               [--batchsize BATCHSIZE]
               [--learning-rate-start LEARNING-RATE-START]
               [--learning-rate-end LEARNING-RATE-END]
               [--weight-decay WEIGHT-DECAY]
               [--checkpoint-interval CHECKPOINT-INTERVAL]
               [--expt-dir EXPT-DIR]
               [--diffusion-steps DIFFUSION-STEPS]
               [--generate-image-interval GENERATE-IMAGE-INTERVAL]
               [--channels CHANNELS [CHANNELS...]]
               [--block-depth BLOCK-DEPTH] [--min-freq MIN-FREQ]
               [--max-freq MAX-FREQ] [--embedding-dims EMBEDDING-DIMS]
               [--min-signal-rate MIN-SIGNAL-RATE]
               [--max-signal-rate MAX-SIGNAL-RATE] [--inference]
               [--saved-model-path SAVED-MODEL-PATH]
               [--generate-n-images GENERATE-N-IMAGES] [-h]

optional arguments:
  --epochs EPOCHS       Number of epochs to train (type: Int64,
                        default: 100)
  --image-size IMAGE-SIZE
                        Input image size (square) (type: Int64,
                        default: 128)
  --batchsize BATCHSIZE
                        Training batch size (type: Int64, default:
                        128)
  --learning-rate-start LEARNING-RATE-START
                        Starting learning rate (type: Float32,
                        default: 0.001)
  --learning-rate-end LEARNING-RATE-END
                        Final learning rate (type: Float32, default:
                        1.0f-5)
  --weight-decay WEIGHT-DECAY
                        Weight decay (AdamW lambda) (type: Float32,
                        default: 1.0f-6)
  --checkpoint-interval CHECKPOINT-INTERVAL
                        Save checkpoint every N epochs (type: Int64,
                        default: 25)
  --expt-dir EXPT-DIR   Experiment output directory (default: "")
  --diffusion-steps DIFFUSION-STEPS
                        Number of DDIM reverse diffusion steps (type:
                        Int64, default: 80)
  --generate-image-interval GENERATE-IMAGE-INTERVAL
                        Generate and log images every N epochs (type:
                        Int64, default: 5)
  --channels CHANNELS [CHANNELS...]
                        UNet channels per stage (type: Int64, default:
                        [32, 64, 96, 128])
  --block-depth BLOCK-DEPTH
                        Number of residual blocks per stage (type:
                        Int64, default: 2)
  --min-freq MIN-FREQ   Sinusoidal embedding min frequency (type:
                        Float32, default: 1.0)
  --max-freq MAX-FREQ   Sinusoidal embedding max frequency (type:
                        Float32, default: 1000.0)
  --embedding-dims EMBEDDING-DIMS
                        Sinusoidal embedding dimension (type: Int64,
                        default: 32)
  --min-signal-rate MIN-SIGNAL-RATE
                        Minimum signal rate (type: Float32, default:
                        0.02)
  --max-signal-rate MAX-SIGNAL-RATE
                        Maximum signal rate (type: Float32, default:
                        0.95)
  --inference           Run in inference-only mode
  --saved-model-path SAVED-MODEL-PATH
                        Path to JLD2 checkpoint (required with
                        --inference)
  --generate-n-images GENERATE-N-IMAGES
                        Number of images to generate during inference
                        or periodic logging (type: Int64, default: 12)
  -h, --help            show this help message and exit
```
