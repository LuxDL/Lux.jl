# # Denoising Diffusion Implicit Model (DDIM)

# [Lux.jl](https://github.com/LuxDL/Lux.jl) implementation of Denoising Diffusion Implicit
# Models ([arXiv:2010.02502](https://arxiv.org/abs/2010.02502)).
# The model generates images from Gaussian noises by <em>denoising</em> iteratively.

# ## Package Imports

using ArgCheck, CairoMakie, ConcreteStructs, Comonicon, DataAugmentation, DataDeps, FileIO,
      ImageCore, JLD2, Lux, LuxCUDA, MLUtils, Optimisers, ParameterSchedulers, ProgressBars,
      Random, Setfield, StableRNGs, Statistics, Zygote
using TensorBoardLogger: TBLogger, log_value, log_images

CUDA.allowscalar(false)

# ## Model Definition

# This DDIM implementation follows
# [the Keras example](https://keras.io/examples/generative/ddim/). Embed noise variances to
# embedding.

function sinusoidal_embedding(x::AbstractArray{T, 4}, min_freq::T, max_freq::T,
        embedding_dims::Int) where {T <: AbstractFloat}
    size(x)[1:3] != (1, 1, 1) &&
        throw(DimensionMismatch("Input shape must be (1, 1, 1, batch)"))

    lower, upper = log(min_freq), log(max_freq)
    n = embedding_dims ÷ 2
    d = (upper - lower) / (n - 1)
    freqs = reshape(exp.(lower:d:upper) |> get_device(x), 1, 1, n, 1)
    x_ = 2 .* x .* freqs
    return cat(sinpi.(x_), cospi.(x_); dims=Val(3))
end

function residual_block(in_channels::Int, out_channels::Int)
    return Parallel(+,
        in_channels == out_channels ? NoOpLayer() :
        Conv((1, 1), in_channels => out_channels; pad=SamePad()),
        Chain(BatchNorm(in_channels; affine=false),
            Conv((3, 3), in_channels => out_channels, swish; pad=SamePad()),
            Conv((3, 3), out_channels => out_channels; pad=SamePad()));
        name="ResidualBlock(in_chs=$in_channels, out_chs=$out_channels)")
end

function downsample_block(in_channels::Int, out_channels::Int, block_depth::Int)
    return @compact(;
        name="DownsampleBlock(in_chs=$in_channels, out_chs=$out_channels, block_depth=$block_depth)",
        residual_blocks=Tuple(residual_block(
                                  ifelse(i == 1, in_channels, out_channels), out_channels)
        for i in 1:block_depth),
        meanpool=MeanPool((2, 2)), block_depth) do x
        skips = (x,)
        for i in 1:block_depth
            skips = (skips..., residual_blocks[i](last(skips)))
        end
        y = meanpool(last(skips))
        @return y, skips
    end
end

function upsample_block(in_channels::Int, out_channels::Int, block_depth::Int)
    return @compact(;
        name="UpsampleBlock(in_chs=$in_channels, out_chs=$out_channels, block_depth=$block_depth)",
        residual_blocks=Tuple(residual_block(
                                  ifelse(
                                      i == 1, in_channels + out_channels, out_channels * 2),
                                  out_channels) for i in 1:block_depth),
        upsample=Upsample(:bilinear; scale=2), block_depth) do x_skips
        x, skips = x_skips
        x = upsample(x)
        for i in 1:block_depth
            x = residual_blocks[i](cat(x, skips[end - i + 1]; dims=Val(3)))
        end
        @return x
    end
end

function unet_model(image_size::Tuple{Int, Int}; channels=[32, 64, 96, 128],
        block_depth=2, min_freq=1.0f0, max_freq=1000.0f0, embedding_dims=32)
    upsample = Upsample(:nearest; size=image_size)
    conv_in = Conv((1, 1), 3 => channels[1])
    conv_out = Conv((1, 1), channels[1] => 3; init_weight=Lux.zeros32)

    channel_input = embedding_dims + channels[1]
    down_blocks = [downsample_block(
                       i == 1 ? channel_input : channels[i - 1], channels[i], block_depth)
                   for i in 1:(length(channels) - 1)]
    residual_blocks = Chain([residual_block(
                                 ifelse(i == 1, channels[end - 1], channels[end]),
                                 channels[end]) for i in 1:block_depth]...)

    reverse!(channels)
    up_blocks = [upsample_block(in_chs, out_chs, block_depth)
                 for (in_chs, out_chs) in zip(channels[1:(end - 1)], channels[2:end])]

    #! format: off
    return @compact(;
        upsample, conv_in, conv_out, down_blocks, residual_blocks, up_blocks,
        min_freq, max_freq, embedding_dims,
        num_blocks=(length(channels) - 1)) do x::Tuple{AbstractArray{<:Real, 4}, AbstractArray{<:Real, 4}}
    #! format: on
        noisy_images, noise_variances = x

        @argcheck size(noise_variances)[1:3] == (1, 1, 1)
        @argcheck size(noisy_images, 4) == size(noise_variances, 4)

        emb = upsample(sinusoidal_embedding(
            noise_variances, min_freq, max_freq, embedding_dims))
        x = cat(conv_in(noisy_images), emb; dims=Val(3))
        skips_at_each_stage = ()
        for i in 1:num_blocks
            x, skips = down_blocks[i](x)
            skips_at_each_stage = (skips_at_each_stage..., skips)
        end
        x = residual_blocks(x)
        for i in 1:num_blocks
            x = up_blocks[i]((x, skips_at_each_stage[end - i + 1]))
        end
        @return conv_out(x)
    end
end

function ddim(rng::AbstractRNG, args...; min_signal_rate=0.02f0,
        max_signal_rate=0.95f0, kwargs...)
    unet = unet_model(args...; kwargs...)
    bn = BatchNorm(3; affine=false, track_stats=true)

    return @compact(; unet, bn, rng, min_signal_rate,
        max_signal_rate, dispatch=:DDIM) do x::AbstractArray{<:Real, 4}
        images = bn(x)
        rng = Lux.replicate(rng)

        noises = rand_like(rng, images)
        diffusion_times = rand_like(rng, images, (1, 1, 1, size(images, 4)))

        noise_rates, signal_rates = diffusion_schedules(
            diffusion_times, min_signal_rate, max_signal_rate)

        noisy_images = @. signal_rates * images + noise_rates * noises

        pred_noises, pred_images = denoise(unet, noisy_images, noise_rates, signal_rates)

        @return noises, images, pred_noises, pred_images
    end
end

function diffusion_schedules(diffusion_times::AbstractArray{T, 4}, min_signal_rate::T,
        max_signal_rate::T) where {T <: Real}
    start_angle = acos(max_signal_rate)
    end_angle = acos(min_signal_rate)

    diffusion_angles = @. start_angle + (end_angle - start_angle) * diffusion_times

    signal_rates = @. cos(diffusion_angles)
    noise_rates = @. sin(diffusion_angles)

    return noise_rates, signal_rates
end

function denoise(unet, noisy_images::AbstractArray{T, 4}, noise_rates::AbstractArray{T, 4},
        signal_rates::AbstractArray{T, 4}) where {T <: Real}
    pred_noises = unet((noisy_images, noise_rates .^ 2))
    pred_images = @. (noisy_images - pred_noises * noise_rates) / signal_rates
    return pred_noises, pred_images
end

# ## Helper Functions for Image Generation

function reverse_diffusion(
        model, initial_noise::AbstractArray{T, 4}, diffusion_steps::Int) where {T <: Real}
    num_images = size(initial_noise, 4)
    step_size = one(T) / diffusion_steps
    dev = get_device(initial_noise)

    next_noisy_images = initial_noise
    pred_images = nothing

    min_signal_rate = model.model.value_storage.st_init_fns.min_signal_rate()
    max_signal_rate = model.model.value_storage.st_init_fns.max_signal_rate()

    for step in 1:diffusion_steps
        noisy_images = next_noisy_images

        # We start t = 1, and gradually decreases to t=0
        diffusion_times = (ones(T, 1, 1, 1, num_images) .- step_size * step) |> dev

        noise_rates, signal_rates = diffusion_schedules(
            diffusion_times, min_signal_rate, max_signal_rate)

        pred_noises, pred_images = denoise(
            StatefulLuxLayer{true}(model.model.layers.unet, model.ps.unet, model.st.unet),
            noisy_images, noise_rates, signal_rates)

        next_diffusion_times = diffusion_times .- step_size
        next_noisy_rates, next_signal_rates = diffusion_schedules(
            next_diffusion_times, min_signal_rate, max_signal_rate)

        next_noisy_images = next_signal_rates .* pred_images .+
                            next_noisy_rates .* pred_noises
    end

    return pred_images
end

function denormalize(model::StatefulLuxLayer, x::AbstractArray{<:Real, 4})
    mean = reshape(model.st.bn.running_mean, 1, 1, 3, 1)
    var = reshape(model.st.bn.running_var, 1, 1, 3, 1)
    std = sqrt.(var .+ model.model.layers.bn.epsilon)
    return std .* x .+ mean
end

function save_images(output_dir, images::AbstractArray{<:Real, 4})
    imgs = Vector{Array{RGB, 2}}(undef, size(images, 4))
    for i in axes(images, 4)
        img = @view images[:, :, :, i]
        img = colorview(RGB, permutedims(img, (3, 1, 2)))
        save(joinpath(output_dir, "img_$(i).png"), img)
        imgs[i] = img
    end
    return imgs
end

function generate_and_save_image_grid(output_dir, imgs::Vector{<:AbstractArray{<:RGB, 2}})
    fig = Figure()
    nrows, ncols = 3, 4
    for r in 1:nrows, c in 1:ncols
        i = (r - 1) * ncols + c
        i > length(imgs) && break
        ax = Axis(fig[r, c]; aspect=DataAspect())
        image!(ax, imgs[i])
        hidedecorations!(ax)
    end
    save(joinpath(output_dir, "flowers_generated.png"), fig)
    return
end

function generate(
        model::StatefulLuxLayer, rng, image_size::NTuple{4, Int}, diffusion_steps::Int, dev)
    initial_noise = randn(rng, Float32, image_size...) |> dev
    generated_images = reverse_diffusion(model, initial_noise, diffusion_steps)
    generated_images = denormalize(model, generated_images)
    return clamp01.(generated_images)
end

# ## Dataset

# We will register the dataset using the [DataDeps.jl](https://github.com/oxinabox/DataDeps.jl)
# package. The dataset is available at
# [https://www.robots.ox.ac.uk/~vgg/data/flowers/102/](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
# This allows us to automatically download the dataset when we run the code.

struct FlowersDataset
    image_files::Vector{AbstractString}
    preprocess::Function
    use_cache::Bool
    cache::Vector{Union{Nothing, AbstractArray{Float32, 3}}}
end

function FlowersDataset(preprocess::F, use_cache::Bool) where {F}
    dirpath = try
        joinpath(datadep"FlowersDataset", "jpg")
    catch KeyError
        register(DataDep("FlowersDataset", "102 Category Flowers Dataset",
            "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
            "2d01ecc807db462958cfe3d92f57a8c252b4abd240eb955770201e45f783b246";
            post_fetch_method=file -> run(`tar -xzf $file`)))
        joinpath(datadep"FlowersDataset", "jpg")
    end
    image_files = joinpath.(dirpath, readdir(dirpath))
    cache = map(x -> nothing, image_files)
    return FlowersDataset(image_files, preprocess, use_cache, cache)
end

Base.length(ds::FlowersDataset) = length(ds.image_files)

function Base.getindex(ds::FlowersDataset, i::Int)
    ds.use_cache && !isnothing(ds.cache[i]) && return ds.cache[i]
    img = load(ds.image_files[i])
    img = ds.preprocess(img)
    img = permutedims(channelview(img), (2, 3, 1))
    ds.use_cache && (ds.cache[i] = img)
    return convert(AbstractArray{Float32}, img)
end

function preprocess_image(image::Matrix{<:RGB}, image_size::Int)
    return apply(
        CenterResizeCrop((image_size, image_size)), DataAugmentation.Image(image)) |>
           itemdata
end

const maeloss = MAELoss()

function loss_function(model, ps, st, data)
    (noises, images, pred_noises, pred_images), st = Lux.apply(model, data, ps, st)
    noise_loss = maeloss(pred_noises, noises)
    image_loss = maeloss(pred_images, images)
    return noise_loss, st, (; image_loss, noise_loss)
end

# ## Entry Point for our code

Comonicon.@main function main(; epochs::Int=100, image_size::Int=128,
        batchsize::Int=128, learning_rate_start::Float32=1.0f-3,
        learning_rate_end::Float32=1.0f-5, weight_decay::Float32=1.0f-6,
        checkpoint_interval::Int=25, expt_dir=tempname(@__DIR__),
        diffusion_steps::Int=80, generate_image_interval::Int=5,
        # model hyper params
        channels::Vector{Int}=[32, 64, 96, 128], block_depth::Int=2, min_freq::Float32=1.0f0, max_freq::Float32=1000.0f0,
        embedding_dims::Int=32, min_signal_rate::Float32=0.02f0,
        max_signal_rate::Float32=0.95f0, generate_image_seed::Int=12,
        # inference specific
        inference_mode::Bool=false, saved_model_path=nothing, generate_n_images::Int=12)
    isdir(expt_dir) || mkpath(expt_dir)

    @info "Experiment directory: $(expt_dir)"

    rng = Random.default_rng()
    Random.seed!(rng, 1234)

    image_dir = joinpath(expt_dir, "images")
    isdir(image_dir) || mkpath(image_dir)

    ckpt_dir = joinpath(expt_dir, "checkpoints")
    isdir(ckpt_dir) || mkpath(ckpt_dir)

    gdev = gpu_device()
    @info "Using device: $gdev"

    @info "Building model"
    model = ddim(rng, (image_size, image_size); channels, block_depth, min_freq,
        max_freq, embedding_dims, min_signal_rate, max_signal_rate)
    ps, st = Lux.setup(rng, model) |> gdev

    if inference_mode
        @argcheck saved_model_path!==nothing "`saved_model_path` must be specified for inference"
        @load saved_model_path parameters states
        parameters = parameters |> gdev
        states = states |> gdev
        model = StatefulLuxLayer{true}(model, parameters, Lux.testmode(states))

        generated_images = generate(model, StableRNG(generate_image_seed),
            (image_size, image_size, 3, generate_n_images), diffusion_steps, gdev) |>
                           cpu_device()

        path = joinpath(image_dir, "inference")
        @info "Saving generated images to $(path)"
        imgs = save_images(path, generated_images)
        generate_and_save_image_grid(path, imgs)
        return
    end

    tb_dir = joinpath(expt_dir, "tb_logs")
    @info "Tensorboard logs being saved to $(tb_dir). Run tensorboard with \
           `tensorboard --logdir $(dirname(tb_dir))`"
    tb_logger = TBLogger(tb_dir)

    tstate = Training.TrainState(
        model, ps, st, AdamW(; eta=learning_rate_start, lambda=weight_decay))

    @info "Preparing dataset"
    ds = FlowersDataset(x -> preprocess_image(x, image_size), true)
    data_loader = DataLoader(ds; batchsize, collate=true, parallel=true) |> gdev

    scheduler = CosAnneal(learning_rate_start, learning_rate_end, epochs)

    image_losses = Vector{Float32}(undef, length(data_loader))
    noise_losses = Vector{Float32}(undef, length(data_loader))
    step = 1
    for epoch in 1:epochs
        pbar = ProgressBar(data_loader)

        eta = scheduler(epoch)
        tstate = Optimisers.adjust!(tstate, eta)

        log_value(tb_logger, "Learning Rate", eta; step)

        for (i, data) in enumerate(data_loader)
            step += 1
            (_, _, stats, tstate) = Training.single_train_step!(
                AutoZygote(), loss_function, data, tstate)
            image_losses[i] = stats.image_loss
            noise_losses[i] = stats.noise_loss

            log_value(tb_logger, "Image Loss", stats.image_loss; step)
            log_value(tb_logger, "Noise Loss", stats.noise_loss; step)

            ProgressBars.update(pbar)
            set_description(
                pbar, "Epoch: $(epoch) Image Loss: $(mean(view(image_losses, 1:i))) Noise \
                       Loss: $(mean(view(noise_losses, 1:i)))")
        end

        if epoch % generate_image_interval == 0 || epoch == epochs
            model_test = StatefulLuxLayer{true}(
                tstate.model, tstate.parameters, Lux.testmode(tstate.states))
            generated_images = generate(model_test, StableRNG(generate_image_seed),
                (image_size, image_size, 3, generate_n_images), diffusion_steps, gdev) |>
                               cpu_device()

            path = joinpath(image_dir, "epoch_$(epoch)")
            @info "Saving generated images to $(path)"
            imgs = save_images(path, generated_images)
            log_images(tb_logger, "Generated Images", imgs; step)
        end

        if epoch % checkpoint_interval == 0 || epoch == epochs
            path = joinpath(ckpt_dir, "model_$(epoch).jld2")
            @info "Saving checkpoint to $(path)"
            parameters = tstate.parameters |> cpu_device()
            states = tstate.states |> cpu_device()
            @save path parameters states
        end
    end

    return tstate
end
