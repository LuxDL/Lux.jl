# Imagenet Training using Lux

This implements training of popular model architectures, such as ResNet, AlexNet, and VGG on
the ImageNet dataset.

## Requirements

* Install [julia](https://julialang.org/)
* In the Julia REPL instantiate the `Project.toml` in the parent directory
* Download the ImageNet dataset from http://www.image-net.org/
  - Then, move and extract the training and validation images to labeled subfolders, using
    [this shell script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh)

## Training

To train a model, run `main.jl` with the necessary parameters. See
[Boltz documentation](https://lux.csail.mit.edu/dev/api/Domain_Specific_Modeling/Boltz) for
the model configuration.

```bash
julia --project=examples/ImageNet -t auto examples/ImageNet/main.jl\
  --cfg.dataset.data_root=/home/avik-pal/data/ImageNet/\
  --cfg.dataset.train_batchsize=256 --cfg.dataset.eval_batchsize=256\
  --cfg.optimizer.learning_rate=0.5

julia --project=examples/ImageNet -t auto examples/ImageNet/main.jl\
  --cfg.model.name=alexnet --cfg.model.arch=alexnet\
  --cfg.dataset.data_root=/home/avik-pal/data/ImageNet/\
  --cfg.dataset.train_batchsize=256 --cfg.dataset.eval_batchsize=256\
  --cfg.optimizer.learning_rate=0.01
```

## Distributed Data Parallel Training

Setup [MPI.jl](https://juliaparallel.org/MPI.jl/).
If your system has functional NCCL we will use it for all CUDA communications. Otherwise, we
will use MPI for all communications.

```bash
mpiexecjl -np 4 julia --project=examples/ImageNet -t auto examples/ImageNet/main.jl\
  --cfg.dataset.data_root=/home/avik-pal/data/ImageNet/\
  --cfg.dataset.train_batchsize=256 --cfg.dataset.eval_batchsize=256\
  --cfg.optimizer.learning_rate=0.01
```

## Usage

```bash
usage: main.jl [--cfg.seed CFG.SEED] [--cfg.model.name CFG.MODEL.NAME]
               [--cfg.model.arch CFG.MODEL.ARCH]
               [--cfg.model.pretrained CFG.MODEL.PRETRAINED]
               [--cfg.optimizer.name CFG.OPTIMIZER.NAME]
               [--cfg.optimizer.learning_rate CFG.OPTIMIZER.LEARNING_RATE]
               [--cfg.optimizer.nesterov CFG.OPTIMIZER.NESTEROV]
               [--cfg.optimizer.momentum CFG.OPTIMIZER.MOMENTUM]
               [--cfg.optimizer.weight_decay CFG.OPTIMIZER.WEIGHT_DECAY]
               [--cfg.optimizer.scheduler.name CFG.OPTIMIZER.SCHEDULER.NAME]
               [--cfg.optimizer.scheduler.cycle_length CFG.OPTIMIZER.SCHEDULER.CYCLE_LENGTH]
               [--cfg.optimizer.scheduler.damp_factor CFG.OPTIMIZER.SCHEDULER.DAMP_FACTOR]
               [--cfg.optimizer.scheduler.lr_step CFG.OPTIMIZER.SCHEDULER.LR_STEP]                                                                             
               [--cfg.optimizer.scheduler.lr_step_decay CFG.OPTIMIZER.SCHEDULER.LR_STEP_DECAY]
               [--cfg.train.total_steps CFG.TRAIN.TOTAL_STEPS]
               [--cfg.train.evaluate_every CFG.TRAIN.EVALUATE_EVERY]
               [--cfg.train.resume CFG.TRAIN.RESUME]
               [--cfg.train.evaluate CFG.TRAIN.EVALUATE]
               [--cfg.train.checkpoint_dir CFG.TRAIN.CHECKPOINT_DIR]
               [--cfg.train.log_dir CFG.TRAIN.LOG_DIR]
               [--cfg.train.expt_subdir CFG.TRAIN.EXPT_SUBDIR]
               [--cfg.train.expt_id CFG.TRAIN.EXPT_ID]
               [--cfg.train.print_frequency CFG.TRAIN.PRINT_FREQUENCY]
               [--cfg.dataset.data_root CFG.DATASET.DATA_ROOT]
               [--cfg.dataset.eval_batchsize CFG.DATASET.EVAL_BATCHSIZE]
               [--cfg.dataset.train_batchsize CFG.DATASET.TRAIN_BATCHSIZE]                                                                                     
               [-h]

optional arguments:
  --cfg.seed CFG.SEED   (type: Int64, default: 12345)
  --cfg.model.name CFG.MODEL.NAME
                        (default: "resnet")
  --cfg.model.arch CFG.MODEL.ARCH
                        (default: "resnet18")
  --cfg.model.pretrained CFG.MODEL.PRETRAINED
                        (type: Bool, default: false)
  --cfg.optimizer.name CFG.OPTIMIZER.NAME
                        (default: "adam")
  --cfg.optimizer.learning_rate CFG.OPTIMIZER.LEARNING_RATE
                        (type: Float32, default: 0.01)
  --cfg.optimizer.nesterov CFG.OPTIMIZER.NESTEROV
                        (type: Bool, default: false)
  --cfg.optimizer.momentum CFG.OPTIMIZER.MOMENTUM
                        (type: Float32, default: 0.0)
  --cfg.optimizer.weight_decay CFG.OPTIMIZER.WEIGHT_DECAY
                        (type: Float32, default: 0.0)
  --cfg.optimizer.scheduler.name CFG.OPTIMIZER.SCHEDULER.NAME
                        (default: "step")
  --cfg.optimizer.scheduler.cycle_length CFG.OPTIMIZER.SCHEDULER.CYCLE_LENGTH
                        (type: Int64, default: 50000)
  --cfg.optimizer.scheduler.damp_factor CFG.OPTIMIZER.SCHEDULER.DAMP_FACTOR
                        (type: Float32, default: 1.2)
  --cfg.optimizer.scheduler.lr_step CFG.OPTIMIZER.SCHEDULER.LR_STEP
                        (type: Vector{Int64}, default: [100000, 250000, 500000])
  --cfg.optimizer.scheduler.lr_step_decay CFG.OPTIMIZER.SCHEDULER.LR_STEP_DECAY
                        (type: Float32, default: 0.1)
  --cfg.train.total_steps CFG.TRAIN.TOTAL_STEPS
                        (type: Int64, default: 800000)
  --cfg.train.evaluate_every CFG.TRAIN.EVALUATE_EVERY
                        (type: Int64, default: 10000)
  --cfg.train.resume CFG.TRAIN.RESUME
                        (default: "")
  --cfg.train.evaluate CFG.TRAIN.EVALUATE
                        (type: Bool, default: false)
  --cfg.train.checkpoint_dir CFG.TRAIN.CHECKPOINT_DIR
                        (default: "checkpoints")
  --cfg.train.log_dir CFG.TRAIN.LOG_DIR
                        (default: "logs")
  --cfg.train.expt_subdir CFG.TRAIN.EXPT_SUBDIR
                        (default: "")
  --cfg.train.expt_id CFG.TRAIN.EXPT_ID
                        (default: "")
  --cfg.train.print_frequency CFG.TRAIN.PRINT_FREQUENCY
                        (type: Int64, default: 100)
  --cfg.dataset.data_root CFG.DATASET.DATA_ROOT
                        (default: "")
  --cfg.dataset.eval_batchsize CFG.DATASET.EVAL_BATCHSIZE
                        (type: Int64, default: 64)
  --cfg.dataset.train_batchsize CFG.DATASET.TRAIN_BATCHSIZE
                        (type: Int64, default: 64)
  -h, --help            show this help message and exit
```
