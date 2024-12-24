# Train Vision Models on CIFAR-10

✈️ 🚗 🐦 🐈 🦌 🐕 🐸 🐎 🚢 🚚

We have the following scripts to train vision models on CIFAR-10:

1. `simple_cnn.jl`: Simple CNN model with a sequence of convolutional layers.
2. `mlp_mixer.jl`: MLP-Mixer model.
3. `conv_mixer.jl`: ConvMixer model.

To get the options for each script, run the script with the `--help` flag.

> [!NOTE]
> To train the model using Reactant.jl pass in `--backend=reactant` to the script. This is
> the recommended approch to train the models present in this directory.

## Simple CNN

```bash
julia --startup-file=no \
    --project=. \
    --threads=auto \
    simple_cnn.jl \
    --backend=reactant
```

On a RTX 4050 6GB Laptop GPU the training takes approximately 3 mins and the final training
and test accuracies are 97% and 65%, respectively.

## MLP-Mixer

## ConvMixer

> [!NOTE]
> This code has been adapted from https://github.com/locuslab/convmixer-cifar10

This is a simple ConvMixer training script for CIFAR-10. It's probably a good starting point
for new experiments on small datasets.

You can get around **90.0%** accuracy in just **25 epochs** by running the script with the
following arguments, which trains a ConvMixer-256/8 with kernel size 5 and patch size 2.

```bash
julia --startup-file=no \
    --project=. \
    --threads=auto \
    conv_mixer.jl \
    --backend=reactant
```

### Notes

  1. To match the results from the original repo, we need more augmentation strategies, that
     are currently not implemented in DataAugmentation.jl.
