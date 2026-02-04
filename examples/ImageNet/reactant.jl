# ## Package Imports

using Boltz, Lux, MLDataDevices, Reactant
## import Metalhead # Install and load this package to use the Metalhead models with Lux

using DataAugmentation, FileIO, OneHotArrays, Optimisers, ParameterSchedulers
using Setfield, Format, JLD2, Dates, Random, OhMyThreads, ImageTransformations
using ImageTransformations: BSpline, Constant, Linear

# ## Setup Distributed Training

# TODO: distributed

# Reactant.Distributed.initialize(; single_gpu_per_process=true)

const cdev = cpu_device()

# TODO: device
# const mesh = Sharding.Mesh(Reactant.devices(), (:batch,))
# const batch_device = reactant_device(;
#     force=true, sharding=Sharding.DimsSharding(mesh, (-1,), (:batch,))
# )
const xdev = reactant_device(; force=true)

# ## Data Loading for ImageNet

## We need the data to be in a specific format. See the
## [README.md](@__REPO_ROOT_URL__/examples/ImageNet/README.md) for more details.

const IMAGENET_CORRUPTED_FILES = [
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
    "n02105855_2933.JPEG",
    "ILSVRC2012_val_00019877.JPEG",
]

# TODO: autodownload tiny-imagenet

function load_imagenet(base_path::String, split::Symbol, tinyimagenet::Bool=false)
    @assert split in (:train, :val)

    ## For TinyImageNet validation set, the structure is different:
    ## - val/images/ contains all validation images
    ## - val/val_annotations.txt contains the mapping: image_name \t synset \t ...
    ## We need to use synsets from train folder to create consistent label mapping
    if tinyimagenet && split == :val
        train_path = joinpath(base_path, "train")
        synsets = sort(readdir(train_path))
        nlabels = length(synsets)
        @assert nlabels == 200 "There should be 200 subdirectories in \
                                $(train_path) found $(nlabels) instead."

        synset_to_label = Dict(synset => i - 1 for (i, synset) in enumerate(synsets))

        # Read val_annotations.txt to get image -> synset mapping
        val_annotations_path = joinpath(base_path, "val", "val_annotations.txt")
        image_to_synset = Dict{String,String}()
        for line in eachline(val_annotations_path)
            parts = Base.split(line, '\t')
            image_name = parts[1]
            synset = parts[2]
            image_to_synset[image_name] = synset
        end

        # Load images from val/images/
        image_dir = joinpath(base_path, "val", "images")
        filenames = readdir(image_dir)
        filter!(x -> x ∉ IMAGENET_CORRUPTED_FILES, filenames)

        image_files = String[]
        labels = Int[]
        for filename in filenames
            push!(image_files, joinpath(image_dir, filename))
            push!(labels, synset_to_label[image_to_synset[filename]])
        end

        return image_files, labels, nlabels
    end

    full_path = joinpath(base_path, string(split))
    synsets = sort(readdir(full_path))

    nlabels = length(synsets)
    @assert nlabels in (200, 1000)

    image_files = String[]
    labels = Int[]
    for (i, synset) in enumerate(synsets)
        image_dir = joinpath(full_path, synset)
        if nlabels == 200
            image_dir = joinpath(image_dir, "images")
        end
        filenames = readdir(image_dir)
        filter!(x -> x ∉ IMAGENET_CORRUPTED_FILES, filenames)
        paths = joinpath.((image_dir,), filenames)
        append!(image_files, paths)
        append!(labels, repeat([i - 1], length(paths)))
    end

    return image_files, labels, nlabels
end

# ### Dataset

default_image_size(::Type{Vision.VisionTransformer}, ::Nothing) = 256
default_image_size(::Type{Vision.VisionTransformer}, size::Int) = size
default_image_size(_, ::Nothing) = 224
default_image_size(_, size::Int) = size

struct MakeColoredImage <: DataAugmentation.Transform end

DataAugmentation.makebuffer(::MakeColoredImage, items) = nothing

function DataAugmentation.apply(
    ::MakeColoredImage, item::DataAugmentation.AbstractArrayItem; randstate=nothing
)
    data = itemdata(item)
    (ndims(data) == 2 || size(data, 3) == 1) && (data = cat(data, data, data; dims=Val(3)))
    return DataAugmentation.setdata(item, data)
end

struct FileDataset
    files
    labels
    augment
    nlabels::Int
end

Base.length(dataset::FileDataset) = length(dataset.files)

function load_image(dataset::FileDataset, i::Int)
    return Image(FileIO.load(dataset.files[i]); interpolate=BSpline(Constant()))
end

function Base.getindex(dataset::FileDataset, i::Int; buffer=nothing)
    img = load_image(dataset, i)
    if buffer !== nothing
        aug_img = itemdata(DataAugmentation.apply!(buffer, dataset.augment, img))
    else
        aug_img = itemdata(DataAugmentation.apply(dataset.augment, img))
    end
    return aug_img, dataset.labels[i]
end

# ### DataLoader

struct DataLoader
    data::FileDataset
    batchsize::Int
    device::MLDataDevices.AbstractDevice
    shuffle::Bool
    imagesize::Int
    nlabels::Int
    cache::Tuple
end

function DataLoader(data, batchsize, device, shuffle, imagesize, nlabels)
    img_cache = Array{Float32,4}(undef, imagesize, imagesize, 3, batchsize)
    label_cache = Array{Float32,1}(undef, batchsize)

    dummy_img = load_image(data, 1)
    buf = DataAugmentation.makebuffer(data.augment, dummy_img)
    chnl = Channel{typeof(buf)}(Threads.nthreads())
    put!(chnl, buf)
    foreach(2:Threads.nthreads()) do _
        put!(chnl, DataAugmentation.makebuffer(data.augment, dummy_img))
    end

    return DataLoader(
        data, batchsize, device, shuffle, imagesize, nlabels, (img_cache, label_cache, chnl)
    )
end

function Base.iterate(loader::DataLoader)
    idxs = loader.shuffle ? Random.shuffle(1:length(loader.data)) : 1:length(loader.data)
    idxs = collect(Iterators.partition(idxs, loader.batchsize))
    length(idxs[end]) < loader.batchsize && (idxs = idxs[1:(end - 1)])
    return iterate(loader, (1, idxs))
end

function Base.iterate(loader::DataLoader, state)
    idx, idxs = state
    idx > length(idxs) && return nothing
    OhMyThreads.tforeach(1:(loader.batchsize); scheduler=:static) do batch_idx
        buffer = take!(loader.cache[3])

        d, l = getindex(loader.data, idxs[idx][batch_idx]; buffer=buffer)
        loader.cache[1][:, :, :, batch_idx] .= d
        loader.cache[2][batch_idx] = l

        put!(loader.cache[3], buffer)
    end
    # TODO: move to device
    return (
        loader.device((
            loader.cache[1], onehotbatch(loader.cache[2], 0:(loader.data.nlabels - 1))
        )),
        (idx + 1, idxs),
    )
end

function construct_dataloaders(;
    base_path::String, train_batchsize, val_batchsize, image_size::Int
)
    train_augment =
        ScaleFixed((256, 256)) |>
        Maybe(FlipX{2}(), 0.5) |>
        RandomResizeCrop((image_size, image_size)) |>
        PinOrigin() |>
        ImageToTensor() |>
        MakeColoredImage() |>
        ToEltype(Float32) |>
        Normalize((0.485f0, 0.456f0, 0.406f0), (0.229f0, 0.224f0, 0.225f0))
    train_files, train_labels, nlabels = load_imagenet(base_path, :train)

    train_dataset = FileDataset(train_files, train_labels, train_augment, nlabels)

    val_augment =
        ScaleFixed((image_size, image_size)) |>
        PinOrigin() |>
        ImageToTensor() |>
        MakeColoredImage() |>
        ToEltype(Float32) |>
        Normalize((0.485f0, 0.456f0, 0.406f0), (0.229f0, 0.224f0, 0.225f0))
    val_files, val_labels, nlabels = load_imagenet(base_path, :val, nlabels == 200)

    val_dataset = FileDataset(val_files, val_labels, val_augment, nlabels)

    train_dataloader = DataLoader(
        train_dataset, train_batchsize, xdev, true, image_size, nlabels
    )
    val_dataloader = DataLoader(
        val_dataset, val_batchsize, xdev, false, image_size, nlabels
    )

    return train_dataloader, val_dataloader
end

# ## Model Construction

function construct_model(;
    rng::AbstractRNG, model_name::String, model_args, pretrained::Bool=false
)
    model = getproperty(Vision, Symbol(model_name))(model_args...; pretrained)
    ps, st = Lux.setup(rng, model) |> xdev
    return model, ps, st
end

# ## Optimizer Configuration

function construct_optimizer_and_scheduler(;
    kind::String,
    learning_rate::AbstractFloat,
    nesterov::Bool,
    momentum::AbstractFloat,
    weight_decay::AbstractFloat,
    scheduler_kind::String,
    cycle_length::Int,
    damp_factor::AbstractFloat,
    lr_step_decay::AbstractFloat,
    lr_step::Vector{Int},
)
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

    optimizer = if iszero(weight_decay)
        optimizer
    else
        OptimiserChain(optimizer, WeightDecay(weight_decay))
    end

    scheduler_kind = Symbol(scheduler_kind)
    scheduler = if scheduler_kind == :cosine
        l0 = learning_rate
        l1 = learning_rate / 100
        ComposedSchedule(
            CosAnneal(l0, l1, cycle_length), Step(l0, damp_factor, cycle_length)
        )
    elseif scheduler_kind == :constant
        Constant(learning_rate)
    elseif scheduler_kind == :step
        Step(learning_rate, lr_step_decay, lr_step)
    else
        throw(ArgumentError("Unknown value for `lr_scheduler` = $(scheduler_kind). \
                             Supported options are: `constant`, `step` and `cosine`."))
    end

    return optimizer, scheduler
end
