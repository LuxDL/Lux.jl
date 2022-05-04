# Imagenet training script based on https://github.com/pytorch/examples/blob/main/imagenet/main.py

using ArgParse                                          # Parse Arguments from Commandline
using Augmentor                                         # Image Augmentation
using CUDA                                              # GPUs <3
using DataLoaders                                       # Pytorch like DataLoaders
using Dates                                             # Printing current time
using Lux                                               # Neural Network Framework
using FluxMPI                                           # Distibuted Training
using Formatting                                        # Pretty Printing
using Functors                                          # Parameter Manipulation
using Images                                            # Image Processing
using Metalhead                                         # Image Classification Models
using MLDataUtils                                       # Shuffling and Splitting Data
using NNlib                                             # Neural Network Backend
using Optimisers                                        # Collection of Gradient Based Optimisers
using ParameterSchedulers                               # Collection of Schedulers for Parameter Updates
using Random                                            # Make things less Random
using Serialization                                     # Serialize Models
using Setfield                                          # Easy Parameter Manipulation
using Zygote                                            # Our AD Engine

import Flux: OneHotArray, onecold, onehot, onehotbatch  # Only being used for OneHotArrays
import DataLoaders: LearnBase                           # Extending Datasets
import MLUtils

# Distributed Training
FluxMPI.Init(;verbose=true)
CUDA.allowscalar(false)

# unsafe_free OneHotArrays
CUDA.unsafe_free!(x::OneHotArray) = CUDA.unsafe_free!(x.indices)

# Image Classification Models
VGG11_BN(args...; kwargs...) = VGG11(args...; batchnorm=true, kwargs...)
VGG13_BN(args...; kwargs...) = VGG13(args...; batchnorm=true, kwargs...)
VGG16_BN(args...; kwargs...) = VGG16(args...; batchnorm=true, kwargs...)
VGG19_BN(args...; kwargs...) = VGG19(args...; batchnorm=true, kwargs...)
MobileNetv3_small(args...; kwargs...) = MobileNetv3(:small, args...; kwargs...)
MobileNetv3_large(args...; kwargs...) = MobileNetv3(:large, args...; kwargs...)
ResNeXt50(args...; kwargs...) = ResNeXt(50, args...; kwargs...)
ResNeXt101(args...; kwargs...) = ResNeXt(101, args...; kwargs...)
ResNeXt152(args...; kwargs...) = ResNeXt(152, args...; kwargs...)

AVAILABLE_IMAGENET_MODELS = [
    AlexNet,
    VGG11,
    VGG13,
    VGG16,
    VGG19,
    VGG11_BN,
    VGG13_BN,
    VGG16_BN,
    VGG19_BN,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNeXt50,
    ResNeXt101,
    ResNeXt152,
    GoogLeNet,
    DenseNet121,
    DenseNet161,
    DenseNet169,
    DenseNet201,
    MobileNetv1,
    MobileNetv2,
    MobileNetv3_small,
    MobileNetv3_large,
    ConvMixer,
]

IMAGENET_MODELS_DICT = Dict(string(model) => model for model in AVAILABLE_IMAGENET_MODELS)

function get_model(model_name::String, models_dict::Dict, rng, args...; warmup=true, kwargs...)
    model = Lux.transform(models_dict[model_name](args...; kwargs...).layers)
    ps, st = Lux.setup(rng, model) .|> gpu
    if warmup
        # Warmup for compilation
        x__ = randn(rng, Float32, 224, 224, 3, 1) |> gpu
        y__ = onehotbatch([1], 1:1000) |> gpu
        should_log() && println("$(now()) ==> staring `$model_name` warmup...")
        model(x__, ps, st)
        should_log() && println("$(now()) ==> forward pass warmup completed")
        (l, _, _), back = Zygote.pullback(p -> logitcrossentropyloss(x__, y__, model, p, st), ps)
        back((one(l), nothing, nothing))
        should_log() && println("$(now()) ==> backward pass warmup completed")
    end

    if is_distributed()
        ps = FluxMPI.synchronize!(ps; root_rank=0)
        st = FluxMPI.synchronize!(st; root_rank=0)
        should_log() && println("$(now()) ==> models synced across all ranks")
    end

    return model, ps, st
end

# Parse Training Arguments
function parse_commandline_arguments()
    parse_settings = ArgParseSettings("Lux ImageNet Training")
    @add_arg_table! parse_settings begin
        "--arch"
            default = "ResNet18"
            range_tester = x -> x ∈ keys(IMAGENET_MODELS_DICT)
            help = "model architectures: " * join(keys(IMAGENET_MODELS_DICT), ", ", " or ")
        "--epochs"
            help = "number of total epochs to run"
            arg_type = Int
            default = 90
        "--start-epoch"
            help = "manual epoch number (useful on restarts)"
            arg_type = Int
            default = 0
        "--batch-size"
            help = "mini-batch size, this is the total batch size across all GPUs"
            arg_type = Int
            default = 256
        "--learning-rate"
            help = "initial learning rate"
            arg_type = Float32
            default = 0.1f0
        "--momentum"
            help = "momentum"
            arg_type = Float32
            default = 0.9f0
        "--weight-decay"
            help = "weight decay"
            arg_type = Float32
            default = 1.0f-4
        "--print-freq"
            help = "print frequency"
            arg_type = Int
            default = 10
        "--resume"
            help = "resume from checkpoint"
            arg_type = String
            default = ""
        "--evaluate"
            help = "evaluate model on validation set"
            action = :store_true
        "--pretrained"
            help = "use pre-trained model"
            action = :store_true
        "--seed"
            help = "seed for initializing training. "
            arg_type = Int
            default = 0
        "data"
            help = "path to dataset"
            required = true
    end

    return parse_args(parse_settings)
end

# Loss Function
logitcrossentropyloss(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims=1); dims=1))

function logitcrossentropyloss(x, y, model, ps, st)
    ŷ, st_ = model(x, ps, st)
    return logitcrossentropyloss(ŷ, y), ŷ, st_
end

# Optimisers / Parameter Schedulers
function update_lr(st::ST, eta) where {ST}
    if hasfield(ST, :eta)
        @set! st.eta = eta
    end
    return st
end
update_lr(st::Optimisers.OptimiserChain, eta) = update_lr.(st.opts, eta)
function update_lr(st::Optimisers.Leaf, eta)
    @set! st.rule = update_lr(st.rule, eta)
end
update_lr(st_opt::NamedTuple, eta) = fmap(l -> update_lr(l, eta), st_opt)

# Accuracy
function accuracy(ŷ, y, topk=(1,))
    maxk = maximum(topk)

    pred_labels = partialsortperm.(eachcol(ŷ), (1:maxk,), rev=true)
    true_labels = onecold(y)

    accuracies = Vector{Float32}(undef, length(topk))

    for (i, k) in enumerate(topk)
        accuracies[i] = sum(map((a, b) -> sum(view(a, 1:k) .== b), pred_labels, true_labels))
    end

    return accuracies .* 100 ./ size(y, ndims(y))
end

# Distributed Utils
is_distributed() = FluxMPI.Initialized() && total_workers() > 1
should_log() = !FluxMPI.Initialized() || local_rank() == 0

# Checkpointing
function save_checkpoint(state, is_best, filename="checkpoint.pth.tar")
    if should_log()
        serialize(filename, state)
        if is_best
            cp(filename, "model_best.pth.tar")
        end
    end
end

# DataLoading
struct ImageDataset
    image_files
    labels
    mapping
    augmentation_pipeline
    normalization_parameters
end

function ImageDataset(folder::String, augmentation_pipeline, normalization_parameters)
    ulabels = readdir(folder)
    label_dirs = joinpath.((folder,), ulabels)
    @assert length(label_dirs) == 1000 "There should be 1000 subdirectories in $folder"

    classes = readlines(joinpath(@__DIR__, "synsets.txt"))
    mapping = Dict(z => i for (i, z) in enumerate(ulabels))

    istrain = endswith(folder, r"train|train/")

    if istrain
        image_files = vcat(map((x, y) -> joinpath.((x,), y), label_dirs, readdir.(label_dirs))...)

        remove_files = [
            "n01739381_1309.JPEG",
            "n02077923_14822.JPEG",
            "n02447366_23489.JPEG",
            "n02492035_15739.JPEG",
            "n02747177_10752.JPEG",
            "n03018349_4028.JPEG",
            "n03062245_4620.JPEG",
            "n03347037_9675.JPEG",
            "n03467068_12171.JPEG",
            "n03529860_11437.JPEG",
            "n03544143_17228.JPEG",
            "n03633091_5218.JPEG",
            "n03710637_5125.JPEG",
            "n03961711_5286.JPEG",
            "n04033995_2932.JPEG",
            "n04258138_17003.JPEG",
            "n04264628_27969.JPEG",
            "n04336792_7448.JPEG",
            "n04371774_5854.JPEG",
            "n04596742_4225.JPEG",
            "n07583066_647.JPEG",
            "n13037406_4650.JPEG",
            "n02105855_2933.JPEG"
        ]
        remove_files = joinpath.(
            (folder,), joinpath.(first.(rsplit.(remove_files, "_", limit=2)), remove_files)
        )
    
        image_files = [setdiff(Set(image_files), Set(remove_files))...]

        labels = [mapping[x] for x in map(x -> x[2], rsplit.(image_files, "/", limit=3))]
    else
        vallist = hcat(split.(readlines(joinpath(@__DIR__, "val_list.txt")))...)
        labels = parse.(Int, vallist[2, :]) .+ 1
        filenames = [joinpath(classes[l], vallist[1, i]) for (i, l) in enumerate(labels)]
        image_files = joinpath.((folder,), filenames)
        idxs = findall(isfile, image_files)
        image_files = image_files[idxs]
        labels = labels[idxs]
    end

    return ImageDataset(image_files, labels, mapping, augmentation_pipeline, normalization_parameters)
end

LearnBase.nobs(data::ImageDataset) = length(data.image_files)

function LearnBase.getobs(data::ImageDataset, i::Int)
    img = Images.load(data.image_files[i])
    img = augment(img, data.augmentation_pipeline)
    cimg = channelview(img)
    if ndims(cimg) == 2
        cimg = reshape(cimg, 1, size(cimg, 1), size(cimg, 2))
        cimg = vcat(cimg, cimg, cimg)
    end
    img = Float32.(permutedims(cimg, (3, 2, 1)))
    img = (img .- data.normalization_parameters.mean) ./ data.normalization_parameters.std
    return img, onehot(data.labels[i], 1:1000)
end

MLUtils.numobs(data::ImageDataset) = length(data.image_files)

MLUtils.getobs(data::ImageDataset, i::Int) = LearnBase.getobs(data, i)

## DataLoaders doesn't yet work with MLUtils
LearnBase.nobs(data::DistributedDataContainer) = MLUtil.numobs(data)

LearnBase.getobs(data::DistributedDataContainer, i::Int) = MLUtils.getobs(data, i)

# Tracking
Base.@kwdef mutable struct AverageMeter
    fmtstr
    val::Float64 = 0.0
    sum::Float64 = 0.0
    count::Int = 0
    average::Float64 = 0
end

function AverageMeter(name::String, fmt::String)
    fmtstr = FormatExpr("$name {1:$fmt} ({2:$fmt})")
    return AverageMeter(; fmtstr=fmtstr)
end

function update!(meter::AverageMeter, val, n::Int)
    meter.val = val
    meter.sum += val * n
    meter.count += n
    meter.average = meter.sum / meter.count
    return meter.average
end

print_meter(meter::AverageMeter) = printfmt(meter.fmtstr, meter.val, meter.average)

struct ProgressMeter{N}
    batch_fmtstr
    meters::NTuple{N,AverageMeter}
end

function ProgressMeter(num_batches::Int, meters::NTuple{N}, prefix::String="") where {N}
    fmt = "%" * string(length(string(num_batches))) * "d"
    prefix = prefix != "" ? endswith(prefix, " ") ? prefix : prefix * " " : ""
    batch_fmtstr = generate_formatter("$prefix[$fmt/" * sprintf1(fmt, num_batches) * "]")
    return ProgressMeter{N}(batch_fmtstr, meters)
end

function print_meter(meter::ProgressMeter, batch::Int)
    base_str = meter.batch_fmtstr(batch)
    print(base_str)
    foreach(x -> (print("\t"); print_meter(x)), meter.meters[1:end])
    return println()
end

# Validation
function validate(val_loader, model, ps, st, args)
    batch_time = AverageMeter("Batch Time", "6.3f")
    losses = AverageMeter("Loss", ".4f")
    top1 = AverageMeter("Acc@1", "6.2f")
    top5 = AverageMeter("Acc@5", "6.2f")

    progress = ProgressMeter(length(val_loader), (batch_time, losses, top1, top5), "Val:")

    st_ = Lux.testmode(st)
    t = time()
    for (i, (x, y)) in enumerate(CuIterator(val_loader))
        # Compute Output
        ŷ, st_ = model(x, ps, st_)
        loss = logitcrossentropyloss(ŷ, y)

        # Metrics
        acc1, acc5 = accuracy(cpu(ŷ), cpu(y), (1, 5))
        update!(top1, acc1, size(x, ndims(x)))
        update!(top5, acc5, size(x, ndims(x)))
        update!(losses, loss, size(x, ndims(x)))

        # Measure Elapsed Time
        bt = time() - t
        update!(batch_time, bt, 1)

        # Print Progress
        if i % args["print-freq"] == 0 || i == length(val_loader)
            print_meter(progress, i)
        end

        t = time()
    end

    return top1.average, top5.average, losses.average
end

# Training
function train(train_loader, model, ps, st, optimiser_state, epoch, args)
    batch_time = AverageMeter("Batch Time", "6.3f")
    data_time = AverageMeter("Data Time", "6.3f")
    losses = AverageMeter("Loss", ".4e")
    top1 = AverageMeter("Acc@1", "6.2f")
    top5 = AverageMeter("Acc@5", "6.2f")
    progress = ProgressMeter(length(train_loader), (batch_time, data_time, losses, top1, top5), "Epoch: [$epoch]")

    st = Lux.trainmode(st)

    t = time()
    for (i, (x, y)) in enumerate(CuIterator(train_loader))
        update!(data_time, time() - t, size(x, ndims(x)))

        # Gradients and Update
        (loss, ŷ, st), back = Zygote.pullback(p -> logitcrossentropyloss(x, y, model, p, st), ps)
        gs = back((one(loss), nothing, nothing))[1]
        optimiser_state, ps = Optimisers.update(optimiser_state, ps, gs)

        # Metrics
        acc1, acc5 = accuracy(cpu(ŷ), cpu(y), (1, 5))
        update!(top1, acc1, size(x, ndims(x)))
        update!(top5, acc5, size(x, ndims(x)))
        update!(losses, loss, size(x, ndims(x)))

        # Measure Elapsed Time
        bt = time() - t
        update!(batch_time, bt, 1)

        # Print Progress
        if i % args["print-freq"] == 0 || i == length(train_loader)
            print_meter(progress, i)
        end

        t = time()
    end

    return ps, st, optimiser_state, (top1.average, top5.average, losses.average)
end

# Main Function
function main(args)
    best_acc1 = 0

    # Seeding
    rng = Random.default_rng()
    Random.seed!(rng, args["seed"])

    # Model Construction
    if should_log()
        if args["pretrained"]
            println("$(now()) => using pre-trained model `$(args["arch"])`")
        else
            println("$(now()) => creating model `$(args["arch"])`")
        end
    end
    model, ps, st = get_model(args["arch"], IMAGENET_MODELS_DICT, rng; warmup=true, pretrain=args["pretrained"])

    normalization_parameters = (
        mean=reshape([0.485f0, 0.456f0, 0.406f0], 1, 1, 3),
        std=reshape([0.229f0, 0.224f0, 0.225f0], 1, 1, 3)
    )
    train_data_augmentation = Resize(256, 256) |> FlipX(0.5) |> RCropSize(224, 224)
    val_data_augmentation = Resize(256, 256) |> CropSize(224, 224)
    train_dataset = ImageDataset(
        joinpath(args["data"], "train"),
        train_data_augmentation,
        normalization_parameters
    )
    val_dataset = ImageDataset(
        joinpath(args["data"], "val"),
        val_data_augmentation,
        normalization_parameters
    )
    if is_distributed()
        train_dataset = DistributedDataContainer(train_dataset)
        val_dataset = DistributedDataContainer(val_dataset)
    end

    train_loader = DataLoader(shuffleobs(train_dataset), args["batch-size"])
    val_loader = DataLoader(val_dataset, args["batch-size"])

    # Optimizer and Scheduler
    should_log() && println("$(now()) => creating optimiser")
    optimiser = Optimisers.OptimiserChain(
        Optimisers.Momentum(args["learning-rate"], args["momentum"]),
        Optimisers.WeightDecay(args["weight-decay"])
    )
    if is_distributed()
        optimiser = DistributedOptimiser(optimiser)
    end
    optimiser_state = Optimisers.setup(optimiser, ps)
    if is_distributed()
        optimiser_state = FluxMPI.synchronize!(optimiser_state)
        should_log() && println("$(now()) ==> synced optimiser state across all ranks")
    end
    scheduler = Step(λ=args["learning-rate"], γ=0.1f0, step_sizes=30)

    if args["resume"] != ""
        if isfile(args["resume"])
            checkpoint = deserialize(args["resume"])
            args["start-epoch"] = checkpoint["epoch"]
            optimiser_state = checkpoint["optimiser_state"] |> gpu
            ps = checkpoint["model_parameters"] |> gpu
            st = checkpoint["model_states"] |> gpu
            should_log() && println("$(now()) => loaded checkpoint `$(args["resume"])` (epoch $(args["start-epoch"]))")
        else
            should_log() && println("$(now()) => no checkpoint found at `$(args["resume"])`")
        end
    end

    if args["evaluate"]
        @assert !is_distributed() "We are not syncing statistics. For evaluation run on 1 process"
        validate(val_loader, model, ps, st, args)
        return
    end

    GC.gc(true)
    CUDA.reclaim()

    for epoch in args["start-epoch"]:args["epochs"]
        # Train for 1 epoch
        ps, st, optimiser_state, _ = train(train_loader, model, ps, st, optimiser_state, epoch, args)

        # Some Housekeeping
        GC.gc(true)
        CUDA.reclaim()

        # Evaluate on validation set
        acc1, _, _ = validate(val_loader, model, ps, st, args)

        # ParameterSchedulers
        eta_new = scheduler(epoch)
        optimiser_state = update_lr(optimiser_state, eta_new)

        # Some Housekeeping
        GC.gc(true)
        CUDA.reclaim()

        # Remember Best Accuracy and Save Checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_state = Dict(
            "epoch" => epoch,
            "arch" => args["arch"],
            "model_states" => st |> cpu,
            "model_parameters" => ps |> cpu,
            "optimiser_state" => optimiser_state |> cpu,
        )
        save_checkpoint(save_state, is_best)
    end
end

main(parse_commandline_arguments())
