# # ImageNet Classification using Distributed Data Parallel Training

# This implements training of popular model architectures, such as ResNet, AlexNet, and VGG
# on the ImageNet dataset.

# For distributed data-parallel training we need to launch this script using `mpiexecjl`

# Setup [MPI.jl](https://juliaparallel.org/MPI.jl/).
# If your system has functional NCCL we will use it for all CUDA communications.
# Otherwise, we will use MPI for all communications.

# ```bash
# mpiexecjl -np 4 julia --startup=no --project=examples/ImageNet -t auto\
#   examples/ImageNet/main.jl \
#   --model-name="ViT" \
#   --model-kind="tiny" \
#   --train-batchsize=256 \
#   --val-batchsize=256 \
#   --optimizer-kind="sgd" \
#   --learning-rate=0.01 \
#   --base-path="/home/avik-pal/data/ImageNet/"
# ```

# For single-node training, we can simply launch the script using `julia`

# ```bash
# julia --startup=no --project=examples/ImageNet -t auto examples/ImageNet/main.jl \
#   --model-name="ViT" \
#   --model-kind="tiny" \
#   --train-batchsize=256 \
#   --val-batchsize=256 \
#   --optimizer-kind="sgd" \
#   --learning-rate=0.01 \
#   --base-path="/home/avik-pal/data/ImageNet/"
# ```


# ## Setup Distributed Training

# We will use NCCL for NVIDIA GPUs and MPI for anything else

const distributed_backend = try
    if gdev isa CUDADevice
        DistributedUtils.initialize(NCCLBackend)
        DistributedUtils.get_distributed_backend(NCCLBackend)
    else
        DistributedUtils.initialize(MPIBackend)
        DistributedUtils.get_distributed_backend(MPIBackend)
    end
catch err
    @error "Could not initialize distributed training. Error: $err"
    nothing
end

const local_rank =
    distributed_backend === nothing ? 0 : DistributedUtils.local_rank(distributed_backend)
const total_workers = if distributed_backend === nothing
    1
else
    DistributedUtils.total_workers(distributed_backend)
end
const is_distributed = total_workers > 1
const should_log = !is_distributed || local_rank == 0


# ## Optimizer Configuration


# ## Utility Functions

const logitcrossentropy = CrossEntropyLoss(; logits=Val(true))

function loss_function(model, ps, st, (img, y))
    ŷ, stₙ = model(img, ps, st)
    return logitcrossentropy(ŷ, y), stₙ, (; prediction=ŷ)
end

sensible_println(msg) = should_log && println("[$(now())] ", msg)
sensible_print(msg) = should_log && print("[$(now())] ", msg)

function accuracy(ŷ::AbstractMatrix, y::AbstractMatrix, topk=(1,))
    pred_labels = partialsortperm.(eachcol(cdev(ŷ)), Ref(1:maximum(topk)); rev=true)
    true_labels = onecold(cdev(y))
    accuracies = Vector{Float64}(undef, length(topk))
    for (i, k) in enumerate(topk)
        accuracies[i] = sum(
            map((a, b) -> sum(view(a, 1:k) .== b), pred_labels, true_labels)
        )
    end
    accuracies .= accuracies .* 100 ./ size(y, 2)
    return accuracies
end

function save_checkpoint(state::NamedTuple; is_best::Bool, filename::String)
    should_log || return nothing
    @assert last(splitext(filename)) == ".jld2" "Filename should have a .jld2 extension."
    isdir(dirname(filename)) || mkpath(dirname(filename))
    save(filename; state)
    sensible_println("=> saved checkpoint `$(filename)`.")
    if is_best
        symlink_safe(filename, joinpath(dirname(filename), "model_best.jld2"))
        sensible_println("=> best model updated to `$(filename)`!")
    end
    symlink_safe(filename, joinpath(dirname(filename), "model_current.jld2"))
    return nothing
end

function symlink_safe(src, dest)
    rm(dest; force=true)
    symlink(src, dest)
    return nothing
end

function load_checkpoint(filename::String)
    try ## NOTE(@avik-pal): ispath is failing for symlinks?
        return JLD2.load(filename)[:state]
    catch
        sensible_println("$(filename) could not be loaded. This might be because the file \
                          is absent or is corrupt. Proceeding by returning `nothing`.")
        return nothing
    end
end

function full_gc_and_reclaim()
    GC.gc(true)
    MLDataDevices.functional(CUDADevice) && CUDA.reclaim()
    MLDataDevices.functional(AMDGPUDevice) && AMDGPU.reclaim()
    return nothing
end

@kwdef mutable struct AverageMeter
    fmtstr
    val::Float64 = 0.0
    sum::Float64 = 0.0
    count::Int = 0
    average::Float64 = 0
end

function AverageMeter(name::String, fmt::String)
    return AverageMeter(; fmtstr=FormatExpr("$(name) {1:$(fmt)} ({2:$(fmt)})"))
end

function (meter::AverageMeter)(val, n::Int)
    meter.val = val
    s = val * n
    if is_distributed
        v = [s, typeof(val)(n)]
        DistributedUtils.allreduce!(backend, v, +)
        s, n = v[1], Int(v[2])
    end
    meter.sum += s
    meter.count += n
    meter.average = meter.sum / meter.count
    return meter.average
end

function reset_meter!(meter::AverageMeter)
    meter.val = 0.0
    meter.sum = 0.0
    meter.count = 0
    meter.average = 0.0
    return meter
end

function print_meter(meter::AverageMeter)
    return should_log && printfmt(meter.fmtstr, meter.val, meter.average)
end

struct ProgressMeter
    batch_fmtstr
    meters
end

function ProgressMeter(num_batches::Int, meters, prefix::String="")
    fmt = "%" * string(length(string(num_batches))) * "d"
    fmt2 = "{1:" * string(length(string(num_batches))) * "d}"
    prefix = prefix != "" ? endswith(prefix, " ") ? prefix : prefix * " " : ""
    batch_fmtstr = FormatExpr("$prefix[$fmt2/" * cfmt(fmt, num_batches) * "]")
    return ProgressMeter(batch_fmtstr, meters)
end

reset_meter!(meter::ProgressMeter) = foreach(reset_meter!, meter.meters)

function print_meter(meter::ProgressMeter, batch::Int)
    should_log || return nothing
    printfmt(meter.batch_fmtstr, batch)
    foreach(meter.meters) do x
        print("\t")
        print_meter(x)
        return nothing
    end
    println()
    return nothing
end

get_loggable_values(meter::ProgressMeter) = getproperty.(meter.meters, :average)

# ## Training and Validation Loops

function validate(val_loader, model, ps, st, step, total_steps)
    batch_time = AverageMeter("Batch Time", "6.5f")
    data_time = AverageMeter("Data Time", "6.5f")
    forward_time = AverageMeter("Forward Pass Time", "6.5f")
    losses = AverageMeter("Loss", ".6f")
    top1 = AverageMeter("Acc@1", "6.4f")
    top5 = AverageMeter("Acc@5", "6.4f")

    progress = ProgressMeter(
        total_steps, (batch_time, data_time, forward_time, losses, top1, top5), "Val:"
    )

    st = Lux.testmode(st)
    t = time()
    for (img, y) in val_loader
        t_data, t = time() - t, time()

        bsize = size(img, ndims(img))

        loss, st, stats = loss_function(model, ps, st, (img, y))
        t_forward = time() - t

        acc1, acc5 = accuracy(stats.prediction, y, (1, 5))

        top1(acc1, bsize)
        top5(acc5, bsize)
        losses(loss, bsize)
        data_time(t_data, bsize)
        forward_time(t_forward, bsize)
        batch_time(t_data + t_forward, bsize)

        t = time()
    end

    print_meter(progress, step)
    return top1.average
end

# ## Entry Point

Comonicon.@main function main(;
    seed::Int=0,
    model_name::String,
    model_kind::String="nokind",
    depth::Int=-1,
    pretrained::Bool=false,
    base_path::String="",
    train_batchsize::Int=64,
    val_batchsize::Int=64,
    image_size::Int=-1,
    optimizer_kind::String="sgd",
    learning_rate::Float32=0.01f0,
    nesterov::Bool=false,
    momentum::Float32=0.0f0,
    weight_decay::Float32=0.0f0,
    scheduler_kind::String="step",
    cycle_length::Int=50000,
    damp_factor::Float32=1.2f0,
    lr_step_decay::Float32=0.1f0,
    lr_step::Vector{Int}=[100000, 250000, 500000],
    expt_id::String="",
    expt_subdir::String=@__DIR__,
    resume::String="",
    evaluate::Bool=false,
    total_steps::Int=800000,
    evaluate_every::Int=10000,
    print_frequency::Int=100,
)
    best_acc1 = 0

    rng = Random.default_rng()
    Random.seed!(rng, seed)

    model_type = getproperty(Vision, Symbol(model_name))
    image_size = default_image_size(model_type, image_size == -1 ? nothing : image_size)

    depth = depth == -1 ? nothing : depth
    model_kind = model_kind == "nokind" ? nothing : Symbol(model_kind)
    model_args = if model_kind === nothing && depth === nothing
        ()
    elseif model_kind !== nothing
        (model_kind,)
    else
        (depth,)
    end
    model, ps, st = construct_model(; rng, model_name, model_args, pretrained)

    ds_train, ds_val = construct_dataloaders(;
        base_path, train_batchsize, val_batchsize, image_size
    )

    opt, scheduler = construct_optimizer_and_scheduler(;
        kind=optimizer_kind,
        learning_rate,
        nesterov,
        momentum,
        weight_decay,
        scheduler_kind,
        cycle_length,
        damp_factor,
        lr_step_decay,
        lr_step,
    )

    expt_name = "name-$(model_name)_seed-$(seed)_id-$(expt_id)"
    ckpt_dir = joinpath(expt_subdir, "checkpoints", expt_name)

    rpath = resume == "" ? joinpath(ckpt_dir, "model_current.jld2") : resume

    ckpt = load_checkpoint(rpath)
    if !isnothing(ckpt)
        ps, st = (ckpt.ps, ckpt.st) |> gdev
        initial_step = ckpt.step
        sensible_println("=> training started from $(initial_step)")
    else
        initial_step = 1
    end

    validate(ds_val, model, ps, st, 0, total_steps)
    evaluate && return nothing

    full_gc_and_reclaim()

    batch_time = AverageMeter("Batch Time", "6.5f")
    data_time = AverageMeter("Data Time", "6.5f")
    training_time = AverageMeter("Training Time", "6.5f")
    losses = AverageMeter("Loss", ".6f")
    top1 = AverageMeter("Acc@1", "6.4f")
    top5 = AverageMeter("Acc@5", "6.4f")

    progress = ProgressMeter(
        total_steps, (batch_time, data_time, training_time, losses, top1, top5), "Train:"
    )

    st = Lux.trainmode(st)
    train_state = Training.TrainState(model, ps, st, opt)
    if is_distributed
        @set! train_state.optimizer_state = DistributedUtils.synchronize!!(
            distributed_backend, train_state.optimizer_state
        )
    end

    train_loader = Iterators.cycle(ds_train)
    _, train_loader_state = iterate(train_loader)
    for step in initial_step:total_steps
        t = time()
        (img, y), train_loader_state = iterate(train_loader, train_loader_state)
        t_data = time() - t

        bsize = size(img, ndims(img))

        t = time()
        _, loss, stats, train_state = Training.single_train_step!(
            AutoZygote(), loss_function, (img, y), train_state
        )
        t_training = time() - t

        isnan(loss) && throw(ArgumentError("NaN loss encountered."))

        acc1, acc5 = accuracy(stats.prediction, y, (1, 5))

        top1(acc1, bsize)
        top5(acc5, bsize)
        losses(loss, bsize)
        data_time(t_data, bsize)
        training_time(t_training, bsize)
        batch_time(t_data + t_training, bsize)

        if step % print_frequency == 1 || step == total_steps
            print_meter(progress, step)
            reset_meter!(progress)
        end

        if step % evaluate_every == 0
            acc1 = validate(ds_val, model, ps, st, step, total_steps)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            save_state = (; ps=cdev(ps), st=cdev(st), step)
            if should_log()
                save_checkpoint(
                    save_state; is_best, filename=joinpath(ckpt_dir, "model_$(step).jld2")
                )
            end
        end

        Optimisers.adjust!(train_state.optimizer_state, scheduler(step + 1))
    end

    return nothing
end
