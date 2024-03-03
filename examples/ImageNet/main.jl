# Imagenet training script based on https://github.com/pytorch/examples/blob/main/imagenet/main.py
using Boltz, Lux, Zygote
using Augmentor, Configurations, Dates, FileIO, Functors, Images, MLUtils, OneHotArrays,
      Optimisers, Random, Setfield, SimpleConfig, Statistics
import FLoops: ThreadedEx
import Metalhead
import MPI, NCCL
using LuxAMDGPU, LuxCUDA

using Formatting # TODO: Get rid of this

# Distributed Training: NCCL for NVIDIA GPUs and MPI for anything else
if LuxCUDA.functional()
    DistributedUtils.initialize(Val(:NCCL))
    backend = DistributedUtils.get_distributed_backend(Val(:NCCL))
else
    DistributedUtils.initialize(Val(:MPI))
    backend = DistributedUtils.get_distributed_backend(Val(:MPI))
end

const local_rank = DistributedUtils.local_rank(backend)
const total_workers = DistributedUtils.total_workers(backend)

is_distributed() = total_workers > 1
should_log() = !is_distributed() || (local_rank == 0)

# Experiment Configuration
include("config.jl")
# Utility Functions
include("utils.jl")
# DataLoading
include("data.jl")

function construct(rng::AbstractRNG, cfg::ModelConfig, ecfg::ExperimentConfig)
    model, ps, st = getfield(Boltz, Symbol(cfg.name))(Symbol(cfg.arch); cfg.pretrained)
    dev = gpu_device()
    ps, st = (ps, st) |> dev

    # Warmup for compilation
    x__ = randn(rng, Float32, 224, 224, 3, 1) |> dev
    y__ = onehotbatch([1], 1:1000) |> dev
    should_log() && println("$(now()) ==> staring `$(cfg.arch)` warmup...")
    model(x__, ps, st)
    should_log() && println("$(now()) ==> forward pass warmup completed")

    if !ecfg.train.evaluate
        (l, _, _), back = Zygote.pullback(
            p -> logitcrossentropyloss(x__, y__, model, p, st), ps)
        back((one(l), nothing, nothing))
        should_log() && println("$(now()) ==> backward pass warmup completed")
    end

    if is_distributed()
        ps = DistributedUtils.synchronize!!(backend, ps)
        st = DistributedUtils.synchronize!!(backend, st)
        should_log() && println("$(now()) ===> models synced across all ranks")
    end

    return (model, ps, st)
end

function construct(cfg::OptimizerConfig)
    if cfg.name == "adam"
        opt = Adam(cfg.learning_rate)
    elseif cfg.name == "sgd"
        if cfg.nesterov
            opt = Nesterov(cfg.learning_rate, cfg.momentum)
        elseif cfg.momentum == 0
            opt = Descent(cfg.learning_rate)
        else
            opt = Momentum(cfg.learning_rate, cfg.momentum)
        end
    else
        throw(ArgumentError("unknown value for `optimizer` = $(cfg.optimizer). Supported \
                             options are: `adam` and `sgd`."))
    end

    if cfg.weight_decay != 0
        opt = OptimiserChain(opt, WeightDecay(cfg.weight_decay))
    end

    if cfg.scheduler.name == "cosine"
        scheduler = CosineAnnealSchedule(cfg.learning_rate, cfg.learning_rate / 100,
            cfg.scheduler.cycle_length; dampen=cfg.scheduler.damp_factor)
    elseif cfg.scheduler.name == "constant"
        scheduler = ConstantSchedule(cfg.learning_rate)
    elseif cfg.scheduler.name == "step"
        scheduler = Step(
            cfg.learning_rate, cfg.scheduler.lr_step_decay, cfg.scheduler.lr_step)
    else
        throw(ArgumentError("unknown value for `lr_scheduler` = $(cfg.scheduler.name). \
                             Supported options are: `constant`, `step` and `cosine`."))
    end

    if is_distributed()
        opt = DistributedUtils.DistributedOptimizer(backend, opt)
    end

    return opt, scheduler
end

function loss_function(model, ps, st, (x, y))
    y_pred, st_ = model(x, ps, st)
    loss = logitcrossentropy(y_pred, y)
    return (loss, st_, (; y_pred))
end

# Validation
function validate(val_loader, model, ps, st, step, total_steps)
    batch_time = AverageMeter("Batch Time", "6.3f")
    data_time = AverageMeter("Data Time", "6.3f")
    forward_time = AverageMeter("Forward Pass Time", "6.3f")
    losses = AverageMeter("Loss", ".4f")
    top1 = AverageMeter("Acc@1", "6.2f")
    top5 = AverageMeter("Acc@5", "6.2f")

    dev = gpu_device()
    cpu_dev = cpu_device()

    progress = ProgressMeter(
        total_steps, (batch_time, data_time, forward_time, losses, top1, top5), "Val:")

    st_ = Lux.testmode(st)
    t = time()
    for (i, (x, y)) in enumerate(val_loader)
        x = x |> dev
        y = y |> dev
        t_data, t = time() - t, time()

        bsize = size(x, ndims(x))

        # Compute Output
        y_pred, st_ = model(x, ps, st_)
        loss = logitcrossentropyloss(y_pred, y)
        t_forward = time() - t

        # Metrics
        acc1, acc5 = accuracy(cpu_dev(y_pred), cpu_dev(y), (1, 5))
        top1(acc1, bsize)
        top5(acc5, bsize)
        losses(loss, bsize)

        # Measure Elapsed Time
        data_time(t_data, bsize)
        forward_time(t_forward, bsize)
        batch_time(t_data + t_forward, bsize)

        t = time()
    end

    should_log() && print_meter(progress, step)

    return top1.average
end

# Main Function
function main(cfg::ExperimentConfig)
    best_acc1 = 0
    dev = gpu_device()
    cpu_dev = cpu_device()

    # Seeding
    rng = get_prng(cfg.seed)

    # Model Construction
    if should_log()
        if cfg.model.pretrained
            println("$(now()) => using pre-trained model `$(cfg.model.arch)`")
        else
            println("$(now()) => creating model `$(cfg.model.arch)`")
        end
    end
    model, ps, st = construct(rng, cfg.model, cfg)

    # DataLoader
    should_log() && println("$(now()) => creating dataloaders")
    ds_train, ds_val = construct(cfg.dataset)
    _, ds_train_state = iterate(ds_train)

    # Optimizer and Scheduler
    should_log() && println("$(now()) => creating optimizer")
    opt, scheduler = construct(cfg.optimizer)
    opt_state = Optimisers.setup(opt, ps)
    if is_distributed()
        opt_state = DistributedUtils.synchronize!!(backend, opt_state)
        should_log() && println("$(now()) ==> synced optimiser state across all ranks")
    end

    expt_name = ("name-$(cfg.model.name)_arch-$(cfg.model.arch)_id-$(cfg.train.expt_id)")
    ckpt_dir = joinpath(cfg.train.expt_subdir, cfg.train.checkpoint_dir, expt_name)
    log_dir = joinpath(cfg.train.expt_subdir, cfg.train.log_dir, expt_name)
    if cfg.train.resume == ""
        rpath = joinpath(ckpt_dir, "model_current.jld2")
    else
        rpath = cfg.train.resume
    end

    ckpt = load_checkpoint(rpath)
    if !isnothing(ckpt)
        ps = ckpt.ps |> dev
        st = ckpt.st |> dev
        opt_state = fmap(dev, ckpt.opt_state)
        initial_step = ckpt.step
        should_log() && println("$(now()) ==> training started from $initial_step")
    else
        initial_step = 1
    end

    validate(ds_val, model, ps, st, 0, cfg.train.total_steps)
    cfg.train.evaluate && return

    reclaim_all()

    batch_time = AverageMeter("Batch Time", "6.3f")
    data_time = AverageMeter("Data Time", "6.3f")
    forward_time = AverageMeter("Forward Pass Time", "6.3f")
    backward_time = AverageMeter("Backward Pass Time", "6.3f")
    optimize_time = AverageMeter("Optimize Time", "6.3f")
    losses = AverageMeter("Loss", ".4e")
    top1 = AverageMeter("Acc@1", "6.2f")
    top5 = AverageMeter("Acc@5", "6.2f")

    progress = ProgressMeter(cfg.train.total_steps,
        (batch_time, data_time, forward_time,
            backward_time, optimize_time, losses, top1, top5),
        "Train: ")

    st = Lux.trainmode(st)

    for step in initial_step:(cfg.train.total_steps)
        # Train Step
        t = time()
        (x, y), ds_train_state = iterate(ds_train, ds_train_state)
        x = x |> dev
        y = y |> dev
        t_data = time() - t

        bsize = size(x, ndims(x))

        # Gradients and Update
        (loss, st, stats), back = Zygote.pullback(
            p -> loss_function(model, p, st, (x, y)), ps)
        t_forward, t = time() - t, time()
        gs = back((one(loss) / total_workers, nothing, nothing))[1]
        t_backward, t = time() - t, time()
        opt_state, ps = Optimisers.update!(opt_state, ps, gs)
        t_opt = time() - t

        # Metrics
        acc1, acc5 = accuracy(cpu_dev(stats.y_pred), cpu_dev(y), (1, 5))
        top1(acc1, bsize)
        top5(acc5, bsize)
        losses(loss, bsize)

        # Measure Elapsed Time
        data_time(t_data, bsize)
        forward_time(t_forward, bsize)
        backward_time(t_backward, bsize)
        optimize_time(t_opt, bsize)
        batch_time(t_data + t_forward + t_backward + t_opt, bsize)

        # Print Progress
        if step % cfg.train.print_frequency == 1 || step == cfg.train.total_steps
            should_log() && print_meter(progress, step)
            reset_meter!(progress)
        end

        if step % cfg.train.evaluate_every == 0
            acc1 = validate(ds_val, model, ps, st, step, cfg.train.total_steps)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            save_state = (ps=ps |> cpu_dev, st=st |> cpu_dev,
                opt_state=fmap(cpu_dev, opt_state), step=step)
            if should_log()
                save_checkpoint(
                    save_state; is_best, filename=joinpath(ckpt_dir, "model_$(step).jld2"))
            end
        end

        # LR Update
        opt_state = Optimisers.adjust(opt_state, scheduler(step + 1))

        t = time()
    end

    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(define_configuration(ARGS, ExperimentConfig, Dict{String, Any}()))
end
