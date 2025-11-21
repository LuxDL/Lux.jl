# # Denoising Diffusion Implicit Model (DDIM)

# [Lux.jl](https://github.com/LuxDL/Lux.jl) implementation of Denoising Diffusion Implicit
# Models ([arXiv:2010.02502](https://arxiv.org/abs/2010.02502)).
# The model generates images from Gaussian noises by <em>denoising</em> iteratively.

# ## Package Imports

ENV["XLA_REACTANT_GPU_MEM_FRACTION"] = get(ENV, "XLA_REACTANT_GPU_MEM_FRACTION", "0.98")

using ConcreteStructs,
    ArgParse,
    DataAugmentation,
    DataDeps,
    Dates,
    Enzyme,
    FileIO,
    ImageCore,
    ImageShow,
    JLD2,
    Lux,
    MLUtils,
    Optimisers,
    ParameterSchedulers,
    ProgressTables,
    Printf,
    Random,
    Reactant,
    Statistics,
    TensorBoardLogger,
    OhMyThreads

const IN_VSCODE = isdefined(Main, :VSCodeServer)

# ## Model Definition

# This DDIM implementation follows
# [the Keras example](https://keras.io/examples/generative/ddim/). Embed noise variances to
# embedding.

function sinusoidal_embedding(
    x::AbstractArray{T,4}, min_freq, max_freq, embedding_dims::Int
) where {T}
    if size(x)[1:3] != (1, 1, 1)
        throw(DimensionMismatch("Input shape must be (1, 1, 1, batch)"))
    end

    lower, upper = T(log(min_freq)), T(log(max_freq))
    n = embedding_dims ÷ 2
    freqs = exp.(reshape(range(lower, upper; length=n), 1, 1, n, 1))
    x_ = 2 .* x .* freqs
    return cat(sinpi.(x_), cospi.(x_); dims=Val(3))
end

@concrete struct ResidualBlock <: AbstractLuxWrapperLayer{:layer}
    layer
end

function ResidualBlock(in_channels::Int, out_channels::Int)
    return ResidualBlock(
        Parallel(
            +,
            if in_channels == out_channels
                NoOpLayer()
            else
                Conv((1, 1), in_channels => out_channels; pad=SamePad())
            end,
            Chain(
                BatchNorm(in_channels; affine=false),
                Conv((3, 3), in_channels => out_channels, swish; pad=SamePad()),
                Conv((3, 3), out_channels => out_channels; pad=SamePad()),
            ),
        ),
    )
end

@concrete struct DownsampleBlock <: AbstractLuxContainerLayer{(:residual_blocks, :pool)}
    residual_blocks
    pool
end

function DownsampleBlock(in_channels::Int, out_channels::Int, block_depth::Int)
    residual_blocks = Tuple([
        ResidualBlock(ifelse(i == 1, in_channels, out_channels), out_channels) for
        i in 1:block_depth
    ])
    return DownsampleBlock(residual_blocks, MeanPool((2, 2)))
end

function (d::DownsampleBlock)(x::AbstractArray, ps, st::NamedTuple)
    skips = (x,)
    st_residual_blocks = ()
    for i in eachindex(d.residual_blocks)
        y, st_new = d.residual_blocks[i](
            last(skips), ps.residual_blocks[i], st.residual_blocks[i]
        )
        skips = (skips..., y)
        st_residual_blocks = (st_residual_blocks..., st_new)
    end
    y, st_pool = d.pool(last(skips), ps.pool, st.pool)
    return (y, skips), (; residual_blocks=st_residual_blocks, pool=st_pool)
end

@concrete struct UpsampleBlock <: AbstractLuxContainerLayer{(:residual_blocks, :upsample)}
    residual_blocks
    upsample
end

function UpsampleBlock(in_channels::Int, out_channels::Int, block_depth::Int)
    residual_blocks = Tuple([
        ResidualBlock(
            ifelse(i == 1, in_channels + out_channels, out_channels * 2), out_channels
        ) for i in 1:block_depth
    ])
    return UpsampleBlock(residual_blocks, Upsample(:bilinear; scale=2))
end

function (u::UpsampleBlock)((x, skips), ps, st::NamedTuple)
    x, st_upsample = u.upsample(x, ps.upsample, st.upsample)
    y, st_residual_blocks = x, ()
    for i in eachindex(u.residual_blocks)
        y, st_new = u.residual_blocks[i](
            cat(y, skips[end - i + 1]; dims=Val(3)),
            ps.residual_blocks[i],
            st.residual_blocks[i],
        )
        st_residual_blocks = (st_residual_blocks..., st_new)
    end
    return y, (; residual_blocks=st_residual_blocks, upsample=st_upsample)
end

@concrete struct UNet <: AbstractLuxContainerLayer{(
    :conv_in, :conv_out, :down_blocks, :residual_blocks, :up_blocks, :upsample
)}
    upsample
    conv_in
    conv_out
    down_blocks
    residual_blocks
    up_blocks
    min_freq
    max_freq
    embedding_dims
end

function UNet(
    image_size::Dims{2};
    channels=[32, 64, 96, 128],
    block_depth=2,
    min_freq=1.0f0,
    max_freq=1000.0f0,
    embedding_dims=32,
)
    upsample = Upsample(:nearest; size=image_size)
    conv_in = Conv((1, 1), 3 => channels[1])
    conv_out = Conv((1, 1), channels[1] => 3; init_weight=zeros32)

    channel_input = embedding_dims + channels[1]
    down_blocks = Tuple([
        DownsampleBlock(i == 1 ? channel_input : channels[i - 1], channels[i], block_depth)
        for i in 1:(length(channels) - 1)
    ])
    residual_blocks = Chain(
        [
            ResidualBlock(ifelse(i == 1, channels[end - 1], channels[end]), channels[end])
            for i in 1:block_depth
        ]...,
    )

    reverse!(channels)
    up_blocks = Tuple([
        UpsampleBlock(in_chs, out_chs, block_depth) for
        (in_chs, out_chs) in zip(channels[1:(end - 1)], channels[2:end])
    ])

    return UNet(
        upsample,
        conv_in,
        conv_out,
        down_blocks,
        residual_blocks,
        up_blocks,
        min_freq,
        max_freq,
        embedding_dims,
    )
end

function (u::UNet)((noisy_images, noise_variances), ps, st::NamedTuple)
    @assert size(noise_variances)[1:3] == (1, 1, 1)
    @assert size(noisy_images, 4) == size(noise_variances, 4)

    emb, st_upsample = u.upsample(
        sinusoidal_embedding(noise_variances, u.min_freq, u.max_freq, u.embedding_dims),
        ps.upsample,
        st.upsample,
    )
    tmp, st_conv_in = u.conv_in(noisy_images, ps.conv_in, st.conv_in)
    x = cat(tmp, emb; dims=Val(3))

    skips_at_each_stage = ()
    st_down_blocks = ()
    for i in eachindex(u.down_blocks)
        (x, skips), st_new = u.down_blocks[i](x, ps.down_blocks[i], st.down_blocks[i])
        skips_at_each_stage = (skips_at_each_stage..., skips)
        st_down_blocks = (st_down_blocks..., st_new)
    end

    x, st_residual_blocks = u.residual_blocks(x, ps.residual_blocks, st.residual_blocks)

    st_up_blocks = ()
    for i in eachindex(u.up_blocks)
        x, st_new = u.up_blocks[i](
            (x, skips_at_each_stage[end - i + 1]), ps.up_blocks[i], st.up_blocks[i]
        )
        st_up_blocks = (st_up_blocks..., st_new)
    end

    x, st_conv_out = u.conv_out(x, ps.conv_out, st.conv_out)

    return (
        x,
        (;
            conv_in=st_conv_in,
            conv_out=st_conv_out,
            down_blocks=st_down_blocks,
            residual_blocks=st_residual_blocks,
            up_blocks=st_up_blocks,
            upsample=st_upsample,
        ),
    )
end

function diffusion_schedules(
    diffusion_times::AbstractArray{T,4}, min_signal_rate, max_signal_rate
) where {T}
    start_angle = T(acos(max_signal_rate))
    end_angle = T(acos(min_signal_rate))

    diffusion_angles = @. start_angle + (end_angle - start_angle) * diffusion_times

    signal_rates = @. cos(diffusion_angles)
    noise_rates = @. sin(diffusion_angles)

    return noise_rates, signal_rates
end

function denoise(
    unet,
    noisy_images::AbstractArray{T1,4},
    noise_rates::AbstractArray{T2,4},
    signal_rates::AbstractArray{T3,4},
) where {T1,T2,T3}
    T = promote_type(T1, T2, T3)
    noisy_images = T.(noisy_images)
    signal_rates = T.(signal_rates)

    pred_noises = unet((noisy_images, noise_rates .^ 2))
    pred_images = @. (noisy_images - pred_noises * noise_rates) / signal_rates
    return pred_noises, pred_images
end

function denoise!(
    pred_images,
    unet,
    noisy_images::AbstractArray{T1,4},
    noise_rates::AbstractArray{T2,4},
    signal_rates::AbstractArray{T3,4},
) where {T1,T2,T3}
    T = promote_type(T1, T2, T3)
    noisy_images = T.(noisy_images)
    noise_rates = T.(noise_rates)
    signal_rates = T.(signal_rates)

    pred_noises = unet((noisy_images, noise_rates .^ 2))
    @. pred_images = (noisy_images - pred_noises * noise_rates) / signal_rates
    return pred_noises
end

@concrete struct DDIM <: AbstractLuxContainerLayer{(:unet, :bn)}
    unet
    bn
    min_signal_rate
    max_signal_rate
    image_size::Dims{3}
end

function DDIM(
    image_size::Dims{2}, args...; min_signal_rate=0.02f0, max_signal_rate=0.95f0, kwargs...
)
    return DDIM(
        UNet(image_size, args...; kwargs...),
        BatchNorm(3; affine=false, track_stats=true),
        min_signal_rate,
        max_signal_rate,
        (image_size..., 3),
    )
end

function Lux.initialstates(rng::AbstractRNG, ddim::DDIM)
    rand(rng, 1)
    return (;
        rng, bn=Lux.initialstates(rng, ddim.bn), unet=Lux.initialstates(rng, ddim.unet)
    )
end

function (ddim::DDIM)(x::AbstractArray{T,4}, ps, st::NamedTuple) where {T}
    images, st_bn = ddim.bn(x, ps.bn, st.bn)

    rng = Lux.replicate(st.rng)
    diffusion_times = rand_like(rng, images, (1, 1, 1, size(images, 4)))

    noise_rates, signal_rates = diffusion_schedules(
        diffusion_times, ddim.min_signal_rate, ddim.max_signal_rate
    )

    noises = randn_like(rng, images)
    noisy_images = @. signal_rates * images + noise_rates * noises

    unet = StatefulLuxLayer{true}(ddim.unet, ps.unet, st.unet)
    pred_noises, pred_images = denoise(unet, noisy_images, noise_rates, signal_rates)

    return ((noises, images, pred_noises, pred_images), (; rng, bn=st_bn, unet=unet.st))
end

## Helper Functions for Image Generation

function generate(model::DDIM, ps, st::NamedTuple, diffusion_steps::Int, num_samples::Int)
    rng = Lux.replicate(st.rng)
    μ, σ² = st.bn.running_mean, st.bn.running_var
    initial_noise = randn_like(rng, μ, (model.image_size..., num_samples))
    generated_images = reverse_diffusion(model, initial_noise, ps, st, diffusion_steps)
    return clamp01.(denormalize(generated_images, μ, σ², model.bn.epsilon))
end

function reverse_diffusion_single_step!(
    pred_images,
    noisy_images,
    step,
    step_size,
    unet,
    ps,
    st,
    ones,
    min_signal_rate,
    max_signal_rate,
)
    diffusion_times = ones .- step_size * step

    noise_rates, signal_rates = diffusion_schedules(
        diffusion_times, min_signal_rate, max_signal_rate
    )

    sunet = StatefulLuxLayer{true}(unet, ps, st)
    pred_noises = denoise!(pred_images, sunet, noisy_images, noise_rates, signal_rates)

    next_diffusion_times = diffusion_times .- step_size
    next_noisy_rates, next_signal_rates = diffusion_schedules(
        next_diffusion_times, min_signal_rate, max_signal_rate
    )
    @. noisy_images = next_signal_rates .* pred_images .+ next_noisy_rates .* pred_noises

    return nothing
end

function reverse_diffusion(
    model::DDIM, initial_noise::AbstractArray{T,4}, ps, st::NamedTuple, diffusion_steps::Int
) where {T}
    step_size = one(T) / diffusion_steps
    ones_dev = ones_like(initial_noise, (1, 1, 1, size(initial_noise, 4)))

    noisy_images, pred_images = initial_noise, similar(initial_noise)

    @trace for step in 1:diffusion_steps
        reverse_diffusion_single_step!(
            pred_images,
            noisy_images,
            step,
            step_size,
            model.unet,
            ps.unet,
            st.unet,
            ones_dev,
            model.min_signal_rate,
            model.max_signal_rate,
        )
    end

    return pred_images
end

function denormalize(x::AbstractArray{T,4}, μ, σ², ε) where {T}
    μ = reshape(μ, 1, 1, 3, 1)
    σ = sqrt.(reshape(σ², 1, 1, 3, 1) .+ T(ε))
    return σ .* x .+ μ
end

function create_image_list(imgs::AbstractArray)
    return map(eachslice(imgs; dims=4)) do img
        cimg = if size(img, 3) == 1
            colorview(Gray, view(img, :, :, 1))
        else
            colorview(RGB, permutedims(img, (3, 1, 2)))
        end
        return cimg'
    end
end

function create_image_grid(
    images::AbstractArray, grid_rows::Int, grid_cols::Union{Int,Nothing}=nothing
)
    images = ndims(images) != 1 ? create_image_list(images) : images
    grid_cols = grid_cols === nothing ? length(images) ÷ grid_rows : grid_cols

    ## Check if the number of images matches the grid
    total_images = grid_rows * grid_cols
    @assert length(images) ≤ total_images

    ## Get the size of a single image (assuming all images are the same size)
    img_height, img_width = size(images[1])

    ## Create a blank grid canvas
    grid_height = img_height * grid_rows
    grid_width = img_width * grid_cols
    grid_canvas = similar(images[1], grid_height, grid_width)

    ## Place each image in the correct position on the canvas
    for idx in 1:total_images
        idx > length(images) && break

        row = div(idx - 1, grid_cols) + 1
        col = mod(idx - 1, grid_cols) + 1

        start_row = (row - 1) * img_height + 1
        start_col = (col - 1) * img_width + 1

        grid_canvas[start_row:(start_row + img_height - 1), start_col:(start_col + img_width - 1)] .= images[idx]
    end

    return grid_canvas
end

function save_images(output_dir, images::Vector{<:AbstractMatrix{<:RGB}})
    for (i, img) in enumerate(images)
        if any(isnan, img)
            @warn "NaN image found in the generated images. Skipping..."
            continue
        end
        save(joinpath(output_dir, "img_$(i).png"), img)
    end
end

# ## Dataset

# We will register the dataset using the
# [DataDeps.jl](https://github.com/oxinabox/DataDeps.jl)
# package. The dataset is available at
# [https://www.robots.ox.ac.uk/~vgg/data/flowers/102/](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
# This allows us to automatically download the dataset when we run the code.

@concrete struct FlowersDataset
    image_size
    image_files
    transform
end

FlowersDataset(image_size::Int) = FlowersDataset((image_size, image_size))

function FlowersDataset(image_size::Dims{2})
    dirpath = try
        joinpath(datadep"FlowersDataset", "jpg")
    catch err
        err isa KeyError || rethrow()
        register(
            DataDep(
                "FlowersDataset",
                "102 Category Flowers Dataset",
                "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
                "2d01ecc807db462958cfe3d92f57a8c252b4abd240eb955770201e45f783b246";
                post_fetch_method=file -> run(`tar -xzf $file`),
            ),
        )
        joinpath(datadep"FlowersDataset", "jpg")
    end
    image_files = joinpath.(dirpath, readdir(dirpath))
    transform =
        ScaleKeepAspect(image_size) |>
        CenterResizeCrop(image_size) |>
        Maybe(FlipX{2}()) |>
        ImageToTensor()
    return FlowersDataset(image_size, image_files, transform)
end

Base.length(ds::FlowersDataset) = length(ds.image_files)

function Base.getindex(ds::FlowersDataset, i::Int)
    return Float32.(itemdata(apply(ds.transform, Image(load(ds.image_files[i])))))
end

function Base.getindex(ds::FlowersDataset, idxs)
    imgs = Array{Float32,4}(undef, ds.image_size..., 3, length(idxs))
    tforeach(1:length(idxs)) do i
        img = Image(load(ds.image_files[idxs[i]]))
        return copyto!(view(imgs, :, :, :, i), itemdata(apply(ds.transform, img)))
    end
    return imgs
end

function loss_function(model, ps, st, data)
    (noises, images, pred_noises, pred_images), stₙ = Lux.apply(model, data, ps, st)
    noise_loss = MSELoss()(pred_noises, noises)
    image_loss = MSELoss()(pred_images, images)
    return noise_loss, stₙ, (; image_loss, noise_loss)
end

# ## Entry Point for our code

function main(;
    epochs::Int=100,
    image_size::Int=128,
    batchsize::Int=128,
    learning_rate_start::Float32=1.0f-3,
    learning_rate_end::Float32=1.0f-5,
    weight_decay::Float32=1.0f-6,
    checkpoint_interval::Int=25,
    expt_dir="",
    diffusion_steps::Int=80,
    generate_image_interval::Int=5,
    ## model hyper params
    channels::Vector{Int}=[32, 64, 96, 128],
    block_depth::Int=2,
    min_freq::Float32=1.0f0,
    max_freq::Float32=1000.0f0,
    embedding_dims::Int=32,
    min_signal_rate::Float32=0.02f0,
    max_signal_rate::Float32=0.95f0,
    ## inference specific
    inference_mode::Bool=false,
    saved_model_path=nothing,
    generate_n_images::Int=12,
)
    if isempty(expt_dir)
        expt_dir =
            joinpath(@__DIR__, string(now(UTC)) * "_" * uppercase(randstring(4))) * "_ddim"
    end
    isdir(expt_dir) || mkpath(expt_dir)

    @printf "[%s] [Info] Experiment directory: %s\n" now(UTC) expt_dir

    rng = Random.default_rng()
    Random.seed!(rng, 1234)

    image_dir = joinpath(expt_dir, "images")
    isdir(image_dir) || mkpath(image_dir)

    ckpt_dir = joinpath(expt_dir, "checkpoints")
    isdir(ckpt_dir) || mkpath(ckpt_dir)

    xdev = reactant_device(; force=true)
    cdev = cpu_device()

    @printf "[%s] [Info] Building model\n" now(UTC)
    model = DDIM(
        (image_size, image_size);
        channels,
        block_depth,
        min_freq,
        max_freq,
        embedding_dims,
        min_signal_rate,
        max_signal_rate,
    )

    if inference_mode
        @assert saved_model_path !== nothing "`saved_model_path` must be specified for \
                                              inference"
        @load saved_model_path parameters states
        ps, st = (parameters, states) |> xdev

        generated_images = @jit generate(
            model, ps, Lux.testmode(st), diffusion_steps, generate_n_images
        )
        generated_images = generated_images |> cdev

        path = joinpath(image_dir, "inference")
        @printf "[%s] [Info] Saving generated images to %s\n" now(UTC) path
        imgs = create_image_list(generated_images)
        save_images(path, imgs)
        if IN_VSCODE
            display(create_image_grid(generated_images, 8, cld(length(imgs), 8)))
        end
        return nothing
    end

    ps, st = Lux.setup(rng, model) |> xdev

    tb_dir = joinpath(expt_dir, "tb_logs")
    @printf "[%s] [Info] Tensorboard logs being saved to %s. Run tensorboard with \
             `tensorboard --logdir %s`\n" now(UTC) tb_dir dirname(tb_dir)
    tb_logger = TBLogger(tb_dir)

    opt = AdamW(; eta=learning_rate_start, lambda=weight_decay)
    scheduler = CosAnneal(learning_rate_start, learning_rate_end, epochs)
    tstate = Training.TrainState(model, ps, st, opt)

    @printf "[%s] [Info] Preparing dataset\n" now(UTC)
    ds = FlowersDataset(image_size)
    data_loader =
        DataLoader(ds; batchsize, shuffle=true, partial=false, parallel=false) |> xdev

    pt = ProgressTable(;
        header=["Epoch", "Image Loss", "Noise Loss", "Time (s)", "Throughput (img/s)"],
        widths=[10, 24, 24, 24, 24],
        format=["%3d", "%.6f", "%.6f", "%.6f", "%.6f"],
        color=[:normal, :normal, :normal, :normal, :normal],
        border=true,
        alignment=[:center, :center, :center, :center, :center],
    )

    @printf "[%s] [Info] Compiling generate function\n" now(UTC)
    time_start = time()
    generate_compiled = @compile generate(
        model, ps, Lux.testmode(st), diffusion_steps, generate_n_images
    )
    @printf "[%s] [Info] Compiled generate function in %.6f seconds\n" now(UTC) (
        time() - time_start
    )

    image_losses = Vector{Float32}(undef, length(data_loader))
    noise_losses = Vector{Float32}(undef, length(data_loader))
    step = 1

    @printf "[%s] [Info] Training model\n" now(UTC)
    initialize(pt)

    for epoch in 1:epochs
        total_time = 0.0
        total_samples = 0

        eta = Float32(scheduler(epoch))
        tstate = Optimisers.adjust!(tstate, eta)

        log_value(tb_logger, "Training/Learning Rate", eta; step)

        start_time = time()
        for (i, data) in enumerate(data_loader)
            (_, loss, stats, tstate) = Training.single_train_step!(
                AutoEnzyme(), loss_function, data, tstate; return_gradients=Val(false)
            )

            @assert !isnan(loss) "NaN loss ($(loss)) encountered!"

            total_samples += size(data, ndims(data))

            image_losses[i] = stats.image_loss
            noise_losses[i] = stats.noise_loss

            log_value(tb_logger, "Training/Image Loss", Float32(stats.image_loss); step)
            log_value(tb_logger, "Training/Noise Loss", Float32(stats.noise_loss); step)
            log_value(
                tb_logger,
                "Training/Throughput",
                total_samples / (time() - start_time);
                step,
            )

            step += 1
        end

        total_time = time() - start_time
        next(
            pt,
            [
                epoch,
                mean(image_losses),
                mean(noise_losses),
                total_time,
                total_samples / total_time,
            ],
        )

        if epoch % generate_image_interval == 0 || epoch == epochs
            generated_images = generate_compiled(
                tstate.model,
                tstate.parameters,
                Lux.testmode(tstate.states),
                diffusion_steps,
                generate_n_images,
            )
            generated_images = generated_images |> cdev

            path = joinpath(image_dir, "epoch_$(epoch)")
            imgs = create_image_list(generated_images)
            save_images(path, imgs)
            log_images(tb_logger, "Generated Images", imgs; step)
            if IN_VSCODE
                display(create_image_grid(generated_images, 8, cld(length(imgs), 8)))
            end
        end

        if epoch % checkpoint_interval == 0 || epoch == epochs
            path = joinpath(ckpt_dir, "model_$(epoch).jld2")
            @printf "[%s] [Info] Saving checkpoint to %s\n" now(UTC) path
            parameters = tstate.parameters |> cdev
            states = tstate.states |> cdev
            @save path parameters states
        end
    end

    finalize(pt)
    @printf "[%s] [Info] Saving final model\n" now(UTC)

    return tstate
end

function get_argparse_settings()
    s = ArgParseSettings(; autofix_names=true)
    #! format: off
    @add_arg_table s begin
        "--epochs"
            help = "Number of epochs to train"
            arg_type = Int
            default = 100
        "--image-size"
            help = "Input image size (square)"
            arg_type = Int
            default = 128
        "--batchsize"
            help = "Training batch size"
            arg_type = Int
            default = 128
        "--learning-rate-start"
            help = "Starting learning rate"
            arg_type = Float32
            default = 3.0f-3
        "--learning-rate-end"
            help = "Final learning rate"
            arg_type = Float32
            default = 1.0f-4
        "--weight-decay"
            help = "Weight decay (AdamW lambda)"
            arg_type = Float32
            default = 1.0f-6
        "--checkpoint-interval"
            help = "Save checkpoint every N epochs"
            arg_type = Int
            default = 25
        "--expt-dir"
            help = "Experiment output directory"
            arg_type = String
            default = ""
        "--diffusion-steps"
            help = "Number of DDIM reverse diffusion steps"
            arg_type = Int
            default = 80
        "--generate-image-interval"
            help = "Generate and log images every N epochs"
            arg_type = Int
            default = 5
        # model hyper params
        "--channels"
            help = "UNet channels per stage"
            arg_type = Int
            nargs = '+'
            default = [32, 64, 96, 128]
        "--block-depth"
            help = "Number of residual blocks per stage"
            arg_type = Int
            default = 2
        "--min-freq"
            help = "Sinusoidal embedding min frequency"
            arg_type = Float32
            default = 1.0f0
        "--max-freq"
            help = "Sinusoidal embedding max frequency"
            arg_type = Float32
            default = 1000.0f0
        "--embedding-dims"
            help = "Sinusoidal embedding dimension"
            arg_type = Int
            default = 32
        "--min-signal-rate"
            help = "Minimum signal rate"
            arg_type = Float32
            default = 0.02f0
        "--max-signal-rate"
            help = "Maximum signal rate"
            arg_type = Float32
            default = 0.95f0
        # inference specific
        "--inference"
            help = "Run in inference-only mode"
            action = :store_true
        "--saved-model-path"
            help = "Path to JLD2 checkpoint (required with --inference)"
            arg_type = String
        "--generate-n-images"
            help = "Number of images to generate during inference or periodic logging"
            arg_type = Int
            default = 12
    end
    #! format: on
    return s
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_args(ARGS, get_argparse_settings(); as_symbols=true)

    main(;
        epochs=args[:epochs],
        image_size=args[:image_size],
        batchsize=args[:batchsize],
        learning_rate_start=args[:learning_rate_start],
        learning_rate_end=args[:learning_rate_end],
        weight_decay=args[:weight_decay],
        checkpoint_interval=args[:checkpoint_interval],
        expt_dir=args[:expt_dir],
        diffusion_steps=args[:diffusion_steps],
        generate_image_interval=args[:generate_image_interval],
        channels=args[:channels],
        block_depth=args[:block_depth],
        min_freq=args[:min_freq],
        max_freq=args[:max_freq],
        embedding_dims=args[:embedding_dims],
        min_signal_rate=args[:min_signal_rate],
        max_signal_rate=args[:max_signal_rate],
        inference_mode=args[:inference],
        saved_model_path=args[:saved_model_path],
        generate_n_images=args[:generate_n_images],
    )
end
