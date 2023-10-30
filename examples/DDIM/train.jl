using Lux
using Random
using Images
using Augmentor
using MLUtils
using Optimisers
using Statistics
using ProgressBars
using Zygote
using CUDA
using BSON
using Comonicon
using Printf

include("./model.jl")

# Image dataset
# `image_files` are vector of image file path
# if `use_cache=true`, preprocessed image is cached on memory
struct ImageDataset
    image_files::Vector{AbstractString}
    preprocess::Function
    use_cache::Bool
    cache::Vector{Union{Nothing, AbstractArray{Float32, 3}}}
end

function ImageDataset(dirpath::AbstractString, preprocess::Function, use_cache::Bool)
    image_files = joinpath.(dirpath, readdir(dirpath))
    cache = map(x -> nothing, image_files)
    return ImageDataset(image_files, preprocess, use_cache, cache)
end

Base.length(ds::ImageDataset) = length(ds.image_files)

function Base.getindex(ds::ImageDataset, i::Int)
    if ds.use_cache && !isnothing(ds.cache[i])
        return ds.cache[i]
    else
        img = Images.load(ds.image_files[i])
        img = ds.preprocess(img)
        img = permutedims(channelview(img), (2, 3, 1))
        if ds.use_cache
            ds.cache[i] = img
        end
        return Float32.(img)
    end
end

function preprocess_image(image::Matrix{RGB{T}}, image_size::Int) where {T <: Real}
    sigma = min(size(image)...) / image_size
    k = round(Int, 2 * sigma) * 2 + 1 # kernel size of two sigma in each direction
    pl = CropRatio(1.0) |> GaussianBlur(k, sigma) |> Resize(image_size, image_size)
    return augment(image, pl)
end

#=
Training utilities
=#
function compute_loss(ddim::DenoisingDiffusionImplicitModel{T},
        images::AbstractArray{T, 4},
        rng::AbstractRNG,
        ps,
        st::NamedTuple) where {T <: AbstractFloat}
    (noises, images, pred_noises, pred_images), st = ddim((images, rng), ps, st)
    noise_loss = mean(abs.(pred_noises - noises))
    image_loss = mean(abs.(pred_images - images))
    loss = noise_loss + image_loss
    return loss, st
end

function train_step(ddim::DenoisingDiffusionImplicitModel{T},
        images::AbstractArray{T, 4},
        rng::AbstractRNG,
        ps,
        st::NamedTuple,
        opt_st::NamedTuple) where {T <: AbstractFloat}
    (loss, st), back = Zygote.pullback(p -> compute_loss(ddim, images, rng, p, st), ps)
    gs = back((one(loss), nothing))[1]
    opt_st, ps = Optimisers.update(opt_st, ps, gs)
    return loss, ps, st, opt_st
end

function save_checkpoint(ps, st, opt_st, output_dir, epoch)
    path = joinpath(output_dir, @sprintf("checkpoint_%.4d.bson", epoch))
    cpu_dev = cpu_device()
    return bson(path,
        Dict(:ps => cpu_dev(ps), :st => cpu_dev(st), :opt_st => cpu_dev(opt_st)))
end

function save_as_png(images::AbstractArray{T, 4},
        output_dir,
        epoch) where {T <: AbstractFloat}
    for i in axes(images, 4)
        img = @view images[:, :, :, i]
        img = colorview(RGB, permutedims(img, (3, 1, 2)))
        save(joinpath(output_dir, @sprintf("img_%.3d_epoch_%.4d.png", i, epoch)), img)
    end
end

@main function main(;
        dataset_dir::String,
        epochs::Int=1,
        image_size::Int=64,
        batchsize::Int=64,
        learning_rate::Float64=1e-3,
        weight_decay::Float64=1e-4,
        val_diffusion_steps::Int=3,
        checkpoint_interval::Int=5,
        output_dir::String="output/train",
        # model hyper params
        channels::Vector{Int}=[32, 64, 96, 128],
        block_depth::Int=2,
        min_freq::Float32=1.0f0,
        max_freq::Float32=1000.0f0,
        embedding_dims::Int=32,
        min_signal_rate::Float32=0.02f0,
        max_signal_rate::Float32=0.95f0)
    rng = Random.MersenneTwister()
    Random.seed!(rng, 1234)

    image_dir = joinpath(output_dir, "images")
    ckpt_dir = joinpath(output_dir, "ckpt")
    mkpath(image_dir)
    mkpath(ckpt_dir)

    if CUDA.functional()
        println("GPU is available.")
    else
        println("GPU is not available.")
    end

    println("Preparing dataset.")
    ds = ImageDataset(dataset_dir, x -> preprocess_image(x, image_size), true)
    data_loader = DataLoader(ds;
        batchsize=batchsize,
        partial=false,
        collate=true,
        parallel=true,
        rng=rng,
        shuffle=true)

    println("Preparing DDIM.")
    dev = gpu_device()

    ddim = DenoisingDiffusionImplicitModel((image_size, image_size);
        channels=channels,
        block_depth=block_depth,
        min_freq=min_freq,
        max_freq=max_freq,
        embedding_dims=embedding_dims,
        min_signal_rate=min_signal_rate,
        max_signal_rate=max_signal_rate)
    ps, st = Lux.setup(rng, ddim) .|> dev

    println("Set optimizer.")
    opt = AdamW(learning_rate, (9.0f-1, 9.99f-1), weight_decay)
    opt_st = Optimisers.setup(opt, ps) |> dev

    rng_gen = Random.MersenneTwister()
    Random.seed!(rng_gen, 0)

    println("Training.")
    for epoch in 1:epochs
        losses = []
        iter = ProgressBar(data_loader)

        st = Lux.trainmode(st)
        for images in iter
            images = images |> dev
            loss, ps, st, opt_st = train_step(ddim, images, rng, ps, st, opt_st)
            push!(losses, loss)
            set_description(iter, "Epoch: $(epoch) Loss: $(mean(losses))")
        end

        st = Lux.testmode(st)
        generated_images, _ = generate(ddim,
            Lux.replicate(rng_gen), # to get inference on the same noises
            (image_size, image_size, 3, 10),
            val_diffusion_steps,
            ps,
            st)
        generated_images = generated_images |> cpu_device()
        save_as_png(generated_images, image_dir, epoch)
        if epoch % checkpoint_interval == 0
            save_checkpoint(ps, st, opt_st, ckpt_dir, epoch)
        end
    end
end
