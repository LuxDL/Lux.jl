using Boltz, Lux, MLDataDevices
# import Metalhead # Install and load this package to use the Metalhead models with Lux

using Dates, Random
using DataAugmentation, FileIO, MLUtils, OneHotArrays, Optimisers, ParameterSchedulers,
      Setfield
using Comonicon, Format
using JLD2
using Zygote

using LuxCUDA
# using AMDGPU # Install and load AMDGPU to train models on AMD GPUs with ROCm
import MPI, NCCL # Enables distributed training in Lux. NCCL is needed for CUDA GPUs

const gdev = gpu_device()
const cdev = cpu_device()

# Distributed Training: NCCL for NVIDIA GPUs and MPI for anything else
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

const local_rank = distributed_backend === nothing ? 0 :
                   DistributedUtils.local_rank(distributed_backend)
const total_workers = distributed_backend === nothing ? 1 :
                      DistributedUtils.total_workers(distributed_backend)
const is_distributed = total_workers > 1
const should_log = !is_distributed || local_rank == 0

# Data Loading for ImageNet
## We need the data to be in a specific format. See the README for more details.
const IMAGENET_CORRUPTED_FILES = [
    "n01739381_1309.JPEG", "n02077923_14822.JPEG", "n02447366_23489.JPEG",
    "n02492035_15739.JPEG", "n02747177_10752.JPEG", "n03018349_4028.JPEG",
    "n03062245_4620.JPEG", "n03347037_9675.JPEG", "n03467068_12171.JPEG",
    "n03529860_11437.JPEG", "n03544143_17228.JPEG", "n03633091_5218.JPEG",
    "n03710637_5125.JPEG", "n03961711_5286.JPEG", "n04033995_2932.JPEG",
    "n04258138_17003.JPEG", "n04264628_27969.JPEG", "n04336792_7448.JPEG",
    "n04371774_5854.JPEG", "n04596742_4225.JPEG", "n07583066_647.JPEG",
    "n13037406_4650.JPEG", "n02105855_2933.JPEG", "ILSVRC2012_val_00019877.JPEG"
]

function load_imagenet1k(base_path::String, split::Symbol)
    @assert split in (:train, :val)
    full_path = joinpath(base_path, string(split))
    synsets = sort(readdir(full_path))
    @assert length(synsets)==1000 "There should be 1000 subdirectories in $(full_path)."

    image_files = String[]
    labels = Int[]
    for (i, synset) in enumerate(synsets)
        filenames = readdir(joinpath(full_path, synset))
        filter!(x -> x ∉ IMAGENET_CORRUPTED_FILES, filenames)
        paths = joinpath.((full_path,), (synset,), filenames)
        append!(image_files, paths)
        append!(labels, repeat([i - 1], length(paths)))
    end

    return image_files, labels
end

default_image_size(::Type{Vision.VisionTransformer}, ::Nothing) = 256
default_image_size(::Type{Vision.VisionTransformer}, size::Int) = size
default_image_size(_, ::Nothing) = 224
default_image_size(_, size::Int) = size

struct MakeColoredImage <: DataAugmentation.Transform end

function DataAugmentation.apply(
        ::MakeColoredImage, item::DataAugmentation.AbstractArrayItem; randstate=nothing)
    data = itemdata(item)
    (ndims(data) == 2 || size(data, 3) == 1) && (data = cat(data, data, data; dims=Val(3)))
    return DataAugmentation.setdata(item, data)
end

struct FileDataset
    files
    labels
    augment
end

Base.length(dataset::FileDataset) = length(dataset.files)

function Base.getindex(dataset::FileDataset, i::Int)
    img = Image(FileIO.load(dataset.files[i]))
    aug_img = itemdata(DataAugmentation.apply(dataset.augment, img))
    return aug_img, OneHotArrays.onehot(dataset.labels[i], 0:999)
end

function construct_dataloaders(;
        base_path::String, train_batchsize, val_batchsize, image_size::Int)
    sensible_println("=> creating dataloaders.")

    train_augment = ScaleFixed((256, 256)) |> Maybe(FlipX(), 0.5) |>
                    RandomResizeCrop((image_size, image_size)) |> PinOrigin() |>
                    ImageToTensor() |> MakeColoredImage() |>
                    Normalize((0.485f0, 0.456f0, 0.406f0), (0.229f0, 0.224f0, 0.225f0)) |>
                    ToEltype(Float32)
    train_files, train_labels = load_imagenet1k(base_path, :train)

    train_dataset = FileDataset(train_files, train_labels, train_augment)

    val_augment = ScaleFixed((image_size, image_size)) |> PinOrigin() |>
                  ImageToTensor() |> MakeColoredImage() |>
                  Normalize((0.485f0, 0.456f0, 0.406f0), (0.229f0, 0.224f0, 0.225f0)) |>
                  ToEltype(Float32)
    val_files, val_labels = load_imagenet1k(base_path, :val)

    val_dataset = FileDataset(val_files, val_labels, val_augment)

    if is_distributed
        train_dataset = DistributedUtils.DistributedDataContainer(distributed_backend,
            train_dataset)
        val_dataset = DistributedUtils.DistributedDataContainer(distributed_backend,
            val_dataset)
    end

    train_dataloader = DataLoader(train_dataset; batchsize=train_batchsize ÷ total_workers,
        partial=false, collate=true, shuffle=true, parallel=true)
    val_dataloader = DataLoader(val_dataset; batchsize=val_batchsize ÷ total_workers,
        partial=true, collate=true, shuffle=false, parallel=true)

    return gdev(train_dataloader), gdev(val_dataloader)
end

# Model Construction
function construct_model(; rng::AbstractRNG, model_name::String, model_args,
        pretrained::Bool=false)
    model = getproperty(Vision, Symbol(model_name))(model_args...; pretrained)
    ps, st = Lux.setup(rng, model) |> gdev

    sensible_println("=> model `$(model_name)` created.")
    pretrained && sensible_println("==> using pre-trained model`")
    sensible_println("==> number of trainable parameters: $(Lux.parameterlength(ps))")
    sensible_println("==> number of states: $(Lux.statelength(st))")

    if is_distributed
        ps = DistributedUtils.synchronize!!(distributed_backend, ps)
        st = DistributedUtils.synchronize!!(distributed_backend, st)
        sensible_println("==> synced model parameters and states across all ranks")
    end

    return model, ps, st
end

# Optimizer Configuration
function construct_optimizer_and_scheduler(; kind::String, learning_rate::AbstractFloat,
        nesterov::Bool, momentum::AbstractFloat, weight_decay::AbstractFloat,
        scheduler_kind::String, cycle_length::Int, damp_factor::AbstractFloat,
        lr_step_decay::AbstractFloat, lr_step::Vector{Int})
    sensible_println("=> creating optimizer.")

    kind = Symbol(kind)
    optimizer = if kind == :adam
        Adam(learning_rate)
    elseif kind == :sgd
        if nesterov
            Nesterov(learning_rate, momentum)
        elseif iszero(momentum)
            Descent(learning_rate)
        else
            Momentum(learning_rate, momentum)
        end
    else
        throw(ArgumentError("Unknown value for `optimizer` = $kind. Supported options are: \
                             `adam` and `sgd`."))
    end

    optimizer = iszero(weight_decay) ? optimizer :
                OptimiserChain(optimizer, WeightDecay(weight_decay))

    sensible_println("=> creating scheduler.")

    scheduler_kind = Symbol(scheduler_kind)
    scheduler = if scheduler_kind == :cosine
        l0 = learning_rate
        l1 = learning_rate / 100
        ComposedSchedule(
            CosAnneal(l0, l1, cycle_length), Step(l0, damp_factor, cycle_length))
    elseif scheduler_kind == :constant
        Constant(learning_rate)
    elseif scheduler_kind == :step
        Step(learning_rate, lr_step_decay, lr_step)
    else
        throw(ArgumentError("Unknown value for `lr_scheduler` = $(scheduler_kind). \
                             Supported options are: `constant`, `step` and `cosine`."))
    end

    optimizer = is_distributed ?
                DistributedUtils.DistributedOptimizer(distributed_backend, optimizer) :
                optimizer

    return optimizer, scheduler
end

# Utility Functions
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
        accuracies[i] = sum(map(
            (a, b) -> sum(view(a, 1:k) .== b), pred_labels, true_labels))
    end
    accuracies .= accuracies .* 100 ./ size(y, 2)
    return accuracies
end

function save_checkpoint(state::NamedTuple; is_best::Bool, filename::String)
    should_log || return
    @assert last(splitext(filename))==".jld2" "Filename should have a .jld2 extension."
    isdir(dirname(filename)) || mkpath(dirname(filename))
    save(filename; state)
    sensible_println("=> saved checkpoint `$(filename)`.")
    if is_best
        symlink_safe(filename, joinpath(dirname(filename), "model_best.jld2"))
        sensible_println("=> best model updated to `$(filename)`!")
    end
    symlink_safe(filename, joinpath(dirname(filename), "model_current.jld2"))
    return
end

function symlink_safe(src, dest)
    rm(dest; force=true)
    symlink(src, dest)
    return
end

function load_checkpoint(filename::String)
    try # NOTE(@avik-pal): ispath is failing for symlinks?
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
    return
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
    should_log && printfmt(meter.fmtstr, meter.val, meter.average)
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
    should_log || return
    printfmt(meter.batch_fmtstr, batch)
    foreach(meter.meters) do x
        print("\t")
        print_meter(x)
        return
    end
    println()
    return
end

get_loggable_values(meter::ProgressMeter) = getproperty.(meter.meters, :average)

# Training and Validation Loops
function validate(val_loader, model, ps, st, step, total_steps)
    batch_time = AverageMeter("Batch Time", "6.5f")
    data_time = AverageMeter("Data Time", "6.5f")
    forward_time = AverageMeter("Forward Pass Time", "6.5f")
    losses = AverageMeter("Loss", ".6f")
    top1 = AverageMeter("Acc@1", "6.4f")
    top5 = AverageMeter("Acc@5", "6.4f")

    progress = ProgressMeter(
        total_steps, (batch_time, data_time, forward_time, losses, top1, top5), "Val:")

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

# Entry Point
Comonicon.@main function main(; seed::Int=0, model_name::String,
        model_kind::String="nokind", depth::Int=-1, pretrained::Bool=false,
        base_path::String="", train_batchsize::Int=64, val_batchsize::Int=64,
        image_size::Int=-1, optimizer_kind::String="sgd", learning_rate::Float32=0.01f0,
        nesterov::Bool=false, momentum::Float32=0.0f0, weight_decay::Float32=0.0f0,
        scheduler_kind::String="step", cycle_length::Int=50000, damp_factor::Float32=1.2f0,
        lr_step_decay::Float32=0.1f0, lr_step::Vector{Int}=[100000, 250000, 500000],
        expt_id::String="", expt_subdir::String=@__DIR__, resume::String="",
        evaluate::Bool=false, total_steps::Int=800000, evaluate_every::Int=10000,
        print_frequency::Int=100)
    best_acc1 = 0

    rng = Random.default_rng()
    Random.seed!(rng, seed)

    model_type = getproperty(Vision, Symbol(model_name))
    image_size = default_image_size(model_type, image_size == -1 ? nothing : image_size)

    depth = depth == -1 ? nothing : depth
    model_kind = model_kind == "nokind" ? nothing : Symbol(model_kind)
    model_args = model_kind === nothing && depth === nothing ? () :
                 model_kind !== nothing ? (model_kind,) : (depth,)
    model, ps, st = construct_model(; rng, model_name, model_args, pretrained)

    ds_train, ds_val = construct_dataloaders(;
        base_path, train_batchsize, val_batchsize, image_size)

    opt, scheduler = construct_optimizer_and_scheduler(;
        kind=optimizer_kind, learning_rate, nesterov, momentum, weight_decay,
        scheduler_kind, cycle_length, damp_factor, lr_step_decay, lr_step)

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
    evaluate && return

    full_gc_and_reclaim()

    batch_time = AverageMeter("Batch Time", "6.5f")
    data_time = AverageMeter("Data Time", "6.5f")
    training_time = AverageMeter("Training Time", "6.5f")
    losses = AverageMeter("Loss", ".6f")
    top1 = AverageMeter("Acc@1", "6.4f")
    top5 = AverageMeter("Acc@5", "6.4f")

    progress = ProgressMeter(
        total_steps, (batch_time, data_time, training_time, losses, top1, top5), "Train:")

    st = Lux.trainmode(st)
    train_state = Training.TrainState(model, ps, st, opt)
    if is_distributed
        @set! train_state.optimizer_state = DistributedUtils.synchronize!!(
            distributed_backend, train_state.optimizer_state)
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
            AutoZygote(), loss_function, (img, y), train_state)
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
                    save_state; is_best, filename=joinpath(ckpt_dir, "model_$(step).jld2"))
            end
        end

        Optimisers.adjust!(train_state.optimizer_state, scheduler(step + 1))
    end

    return
end
