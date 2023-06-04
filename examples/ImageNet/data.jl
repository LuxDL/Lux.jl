# DataLoading
struct ImageDataset
    image_files::Any
    labels::Any
    mapping::Any
    augmentation_pipeline::Any
    normalization_parameters::Any
end

function ImageDataset(folder::String, augmentation_pipeline, normalization_parameters)
    ulabels = readdir(folder)
    label_dirs = joinpath.((folder,), ulabels)
    @assert length(label_dirs)==1000 "There should be 1000 subdirectories in $folder"

    classes = readlines(joinpath(@__DIR__, "synsets.txt"))
    mapping = Dict(z => i for (i, z) in enumerate(ulabels))

    istrain = endswith(folder, r"train|train/")

    if istrain
        image_files = vcat(map((x, y) -> joinpath.((x,), y),
            label_dirs,
            readdir.(label_dirs))...)

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
            "n02105855_2933.JPEG",
        ]
        remove_files = joinpath.((folder,),
            joinpath.(first.(rsplit.(remove_files, "_", limit=2)), remove_files))

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

    return ImageDataset(image_files,
        labels,
        mapping,
        augmentation_pipeline,
        normalization_parameters)
end

function Base.getindex(data::ImageDataset, i::Int)
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

Base.length(data::ImageDataset) = length(data.image_files)

function construct(cfg::DatasetConfig)
    normalization_parameters = (mean=reshape([0.485f0, 0.456f0, 0.406f0], 1, 1, 3),
        std=reshape([0.229f0, 0.224f0, 0.225f0], 1, 1, 3))
    train_data_augmentation = Resize(256, 256) |> FlipX(0.5) |> RCropSize(224, 224)
    val_data_augmentation = Resize(256, 256) |> CropSize(224, 224)
    train_dataset = ImageDataset(joinpath(cfg.data_root, "train"),
        train_data_augmentation,
        normalization_parameters)
    val_dataset = ImageDataset(joinpath(cfg.data_root, "val"),
        val_data_augmentation,
        normalization_parameters)
    if is_distributed()
        train_dataset = DistributedDataContainer(train_dataset)
        val_dataset = DistributedDataContainer(val_dataset)
    end

    train_data = BatchView(shuffleobs(train_dataset);
        batchsize=cfg.train_batchsize ÷ total_workers(),
        partial=false,
        collate=true)

    val_data = BatchView(val_dataset;
        batchsize=cfg.eval_batchsize ÷ total_workers(),
        partial=true,
        collate=true)

    train_iter = Iterators.cycle(MLUtils.eachobsparallel(train_data;
        executor=ThreadedEx(),
        buffer=true))

    val_iter = MLUtils.eachobsparallel(val_data; executor=ThreadedEx(), buffer=true)

    return train_iter, val_iter
end
