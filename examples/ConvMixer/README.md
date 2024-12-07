# Train ConvMixer on CIFAR-10

 ✈️ 🚗 🐦 🐈 🦌 🐕 🐸 🐎 🚢 🚚

> [!NOTE]
> This code has been adapted from https://github.com/locuslab/convmixer-cifar10

This is a simple ConvMixer training script for CIFAR-10. It's probably a good starting point
for new experiments on small datasets.

You can get around **90.0%** accuracy in just **25 epochs** by running the script with the
following arguments, which trains a ConvMixer-256/8 with kernel size 5 and patch size 2.

> [!NOTE]
> To train the model using Reactant.jl pass in `--backend=reactant` to the script.

```bash
julia --startup-file=no \
    --project=. \
    --threads=auto \
    main.jl \
    --lr-max=0.05 \
    --weight-decay=0.0001
```

Here's an example of the output of the above command (on a V100 32GB GPU):

```
Epoch  1: Learning Rate 5.05e-03, Train Acc: 56.91%, Test Acc: 56.49%, Time: 129.84
Epoch  2: Learning Rate 1.01e-02, Train Acc: 69.75%, Test Acc: 68.40%, Time: 21.22
Epoch  3: Learning Rate 1.51e-02, Train Acc: 76.86%, Test Acc: 74.73%, Time: 21.33
Epoch  4: Learning Rate 2.01e-02, Train Acc: 81.03%, Test Acc: 78.14%, Time: 21.40
Epoch  5: Learning Rate 2.51e-02, Train Acc: 72.71%, Test Acc: 70.29%, Time: 21.34
Epoch  6: Learning Rate 3.01e-02, Train Acc: 83.12%, Test Acc: 80.20%, Time: 21.38
Epoch  7: Learning Rate 3.51e-02, Train Acc: 82.38%, Test Acc: 78.66%, Time: 21.39
Epoch  8: Learning Rate 4.01e-02, Train Acc: 84.24%, Test Acc: 79.97%, Time: 21.49
Epoch  9: Learning Rate 4.51e-02, Train Acc: 84.93%, Test Acc: 80.18%, Time: 21.40
Epoch 10: Learning Rate 5.00e-02, Train Acc: 84.97%, Test Acc: 80.26%, Time: 21.37
Epoch 11: Learning Rate 4.52e-02, Train Acc: 89.09%, Test Acc: 83.53%, Time: 21.31
Epoch 12: Learning Rate 4.05e-02, Train Acc: 91.62%, Test Acc: 85.10%, Time: 21.39
Epoch 13: Learning Rate 3.57e-02, Train Acc: 93.71%, Test Acc: 86.78%, Time: 21.29
Epoch 14: Learning Rate 3.10e-02, Train Acc: 95.14%, Test Acc: 87.23%, Time: 21.37
Epoch 15: Learning Rate 2.62e-02, Train Acc: 95.36%, Test Acc: 87.08%, Time: 21.34
Epoch 16: Learning Rate 2.15e-02, Train Acc: 97.07%, Test Acc: 87.91%, Time: 21.26
Epoch 17: Learning Rate 1.67e-02, Train Acc: 98.67%, Test Acc: 89.57%, Time: 21.40
Epoch 18: Learning Rate 1.20e-02, Train Acc: 99.41%, Test Acc: 89.77%, Time: 21.28
Epoch 19: Learning Rate 7.20e-03, Train Acc: 99.81%, Test Acc: 90.31%, Time: 21.39
Epoch 20: Learning Rate 2.50e-03, Train Acc: 99.94%, Test Acc: 90.83%, Time: 21.44
Epoch 21: Learning Rate 2.08e-03, Train Acc: 99.96%, Test Acc: 90.83%, Time: 21.23
Epoch 22: Learning Rate 1.66e-03, Train Acc: 99.97%, Test Acc: 90.91%, Time: 21.29
Epoch 23: Learning Rate 1.25e-03, Train Acc: 99.99%, Test Acc: 90.82%, Time: 21.29
Epoch 24: Learning Rate 8.29e-04, Train Acc: 99.99%, Test Acc: 90.79%, Time: 21.32
Epoch 25: Learning Rate 4.12e-04, Train Acc: 100.00%, Test Acc: 90.83%, Time: 21.32
```

## Usage

```bash
  main [options] [flags]

Options

  --batchsize <512::Int>
  --hidden-dim <256::Int>
  --depth <8::Int>
  --patch-size <2::Int>
  --kernel-size <5::Int>
  --weight-decay <0.01::Float64>
  --seed <42::Int>
  --epochs <25::Int>
  --lr-max <0.01::Float64>
  --backend <reactant::String>

Flags
  --clip-norm

  -h, --help                                                Print this help message.
  --version                                                 Print version.
```

## Notes

  1. To match the results from the original repo, we need more augmentation strategies, that
     are currently not implemented in DataAugmentation.jl.
  2. Don't compare the reported timings in that repo against the numbers here. They time the
     entire loop. We only time the training part of the loop.
