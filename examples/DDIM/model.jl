# DDIM implementation follwoing https://keras.io/examples/generative/ddim/

using Lux
using Random
using CUDA
using NNlib
using Setfield
# Note: Julia/Lux assume image batch of WHCN ordering

# Embed noise variances to embedding
function sinusoidal_embedding(x::AbstractArray{T, 4}, min_freq::T, max_freq::T,
                              embedding_dims::Int) where {T <: AbstractFloat}
    if size(x)[1:3] != (1, 1, 1)
        throw(DimensionMismatch("Input shape must be (1, 1, 1, batch)"))
    end

    # define frequencies
    # LinRange requires @adjoint when used with Zygote
    # Instead we manually implement range.
    lower = log(min_freq)
    upper = log(max_freq)
    n = div(embedding_dims, 2)
    d = (upper - lower) / (n - 1)
    freqs = exp.(lower:d:upper) |> gpu
    @assert length(freqs) == div(embedding_dims, 2)
    @assert size(freqs) == (div(embedding_dims, 2),)

    angular_speeds = reshape(convert(T, 2) * π * freqs, (1, 1, length(freqs), 1))
    @assert size(angular_speeds) == (1, 1, div(embedding_dims, 2), 1)

    embeddings = cat(sin.(angular_speeds .* x), cos.(angular_speeds .* x); dims=3)
    @assert size(embeddings) == (1, 1, embedding_dims, size(x, 4))

    return embeddings
end

# Basic building block of UNet
function residual_block(in_channels::Int, out_channels::Int)
    if in_channels == out_channels
        first_layer = NoOpLayer()
    else
        first_layer = Conv((3, 3), in_channels => out_channels; pad=SamePad())
    end

    return Chain(first_layer,
                 SkipConnection(Chain(BatchNorm(out_channels; affine=false, momentum=0.99),
                                      Conv((3, 3), out_channels => out_channels; stride=1,
                                           pad=(1, 1)), swish,
                                      Conv((3, 3), out_channels => out_channels; stride=1,
                                           pad=(1, 1))), +))
end

# Downsampling block of UNet
# It narrows height and width while increasing channels.
struct DownBlock <: Lux.AbstractExplicitContainerLayer{(:residual_blocks, :maxpool)}
    residual_blocks::Lux.AbstractExplicitLayer
    maxpool::MaxPool
end

function DownBlock(in_channels::Int, out_channels::Int, block_depth::Int)
    layers = []
    push!(layers, residual_block(in_channels, out_channels))
    for _ in 2:block_depth
        push!(layers, residual_block(out_channels, out_channels))
    end
    # disable optimizations to keep block index
    residual_blocks = Chain(layers...; disable_optimizations=true)
    maxpool = MaxPool((2, 2); pad=0)
    return DownBlock(residual_blocks, maxpool)
end

function (db::DownBlock)(x::AbstractArray{T, 4}, ps,
                         st::NamedTuple) where {T <: AbstractFloat}
    skips = () # accumulate intermediate outputs
    for i in 1:length(db.residual_blocks)
        layer_name = Symbol(:layer_, i)
        x, new_st = db.residual_blocks[i](x, ps.residual_blocks[layer_name],
                                          st.residual_blocks[layer_name])
        # Don't use push! on vector because it invokes Zygote error
        skips = (skips..., x)
        @set! st.residual_blocks[layer_name] = new_st
    end
    x, _ = db.maxpool(x, ps.maxpool, st.maxpool)
    return (x, skips), st
end

# Upsampling block of UNet
# It doubles height and width while decreasing channels.
struct UpBlock <: Lux.AbstractExplicitContainerLayer{(:residual_blocks, :upsample)}
    residual_blocks::Lux.AbstractExplicitLayer
    upsample::Upsample
end

function UpBlock(in_channels::Int, out_channels::Int, block_depth::Int)
    layers = []
    push!(layers, residual_block(in_channels + out_channels, out_channels))
    for _ in 2:block_depth
        push!(layers, residual_block(out_channels * 2, out_channels))
    end
    residual_blocks = Chain(layers...; disable_optimizations=true)
    upsample = Upsample(:bilinear; scale=2)
    return UpBlock(residual_blocks, upsample)
end

function (up::UpBlock)(x::Tuple{AbstractArray{T, 4}, NTuple{N, AbstractArray{T, 4}}}, ps,
                       st::NamedTuple) where {T <: AbstractFloat, N}
    x, skips = x
    x, _ = up.upsample(x, ps.upsample, st.upsample)
    for i in 1:length(up.residual_blocks)
        layer_name = Symbol(:layer_, i)
        x = cat(x, skips[end - i + 1]; dims=3) # cat on channel
        x, new_st = up.residual_blocks[i](x, ps.residual_blocks[layer_name],
                                          st.residual_blocks[layer_name])
        @set! st.residual_blocks[layer_name] = new_st
    end

    return x, st
end

# UNet
# It takes as input images array and returns the array of the same size.
struct UNet <:
       Lux.AbstractExplicitContainerLayer{(:upsample, :conv_in, :conv_out, :down_blocks,
                                           :residual_blocks, :up_blocks)}
    upsample::Upsample
    conv_in::Conv
    conv_out::Conv
    down_blocks::Lux.AbstractExplicitLayer
    residual_blocks::Lux.AbstractExplicitLayer
    up_blocks::Lux.AbstractExplicitLayer
    noise_embedding::Function
end

function UNet(image_size::Tuple{Int, Int}; channels=[32, 64, 96, 128], block_depth=2,
              min_freq=1.0f0, max_freq=1000.0f0, embedding_dims=32)
    upsample = Upsample(:nearest; size=image_size)
    conv_in = Conv((1, 1), 3 => channels[1])
    conv_out = Conv((1, 1), channels[1] => 3; init_weight=Lux.zeros32)

    noise_embedding = x -> sinusoidal_embedding(x, min_freq, max_freq, embedding_dims)

    channel_input = embedding_dims + channels[1]

    down_blocks = []
    push!(down_blocks, DownBlock(channel_input, channels[1], block_depth))
    for i in 1:(length(channels) - 2)
        push!(down_blocks, DownBlock(channels[i], channels[i + 1], block_depth))
    end
    down_blocks = Chain(down_blocks...; disable_optimizations=true)

    residual_blocks = []
    push!(residual_blocks, residual_block(channels[end - 1], channels[end]))
    for _ in 2:block_depth
        push!(residual_blocks, residual_block(channels[end], channels[end]))
    end
    residual_blocks = Chain(residual_blocks...; disable_optimizations=true)

    reverse!(channels)
    up_blocks = [UpBlock(channels[i], channels[i + 1], block_depth)
                 for i in 1:(length(channels) - 1)]
    up_blocks = Chain(up_blocks...)

    return UNet(upsample, conv_in, conv_out, down_blocks, residual_blocks, up_blocks,
                noise_embedding)
end

function (unet::UNet)(x::Tuple{AbstractArray{T, 4}, AbstractArray{T, 4}}, ps,
                      st::NamedTuple) where {T <: AbstractFloat}
    noisy_images, noise_variances = x
    @assert size(noise_variances)[1:3] == (1, 1, 1)
    @assert size(noisy_images, 4) == size(noise_variances, 4)

    emb = unet.noise_embedding(noise_variances)
    @assert size(emb)[[1, 2, 4]] == (1, 1, size(noise_variances, 4))
    emb, _ = unet.upsample(emb, ps.upsample, st.upsample)
    @assert size(emb)[[1, 2, 4]] ==
            (size(noisy_images, 1), size(noisy_images, 2), size(noise_variances, 4))

    x, new_st = unet.conv_in(noisy_images, ps.conv_in, st.conv_in)
    @set! st.conv_in = new_st
    @assert size(x)[[1, 2, 4]] ==
            (size(noisy_images, 1), size(noisy_images, 2), size(noisy_images, 4))

    x = cat(x, emb; dims=3)
    @assert size(x)[[1, 2, 4]] ==
            (size(noisy_images, 1), size(noisy_images, 2), size(noisy_images, 4))

    skips_at_each_stage = ()
    for i in 1:length(unet.down_blocks)
        layer_name = Symbol(:layer_, i)
        (x, skips), new_st = unet.down_blocks[i](x, ps.down_blocks[layer_name],
                                                 st.down_blocks[layer_name])
        @set! st.down_blocks[layer_name] = new_st
        skips_at_each_stage = (skips_at_each_stage..., skips)
    end

    x, new_st = unet.residual_blocks(x, ps.residual_blocks, st.residual_blocks)
    @set! st.residual_blocks = new_st

    for i in 1:length(unet.up_blocks)
        layer_name = Symbol(:layer_, i)
        x, new_st = unet.up_blocks[i]((x, skips_at_each_stage[end - i + 1]),
                                      ps.up_blocks[layer_name], st.up_blocks[layer_name])
        @set! st.up_blocks[layer_name] = new_st
    end

    x, new_st = unet.conv_out(x, ps.conv_out, st.conv_out)
    @set! st.conv_out = new_st

    return x, st
end

# Define DDIM model
# This generates noise, adds it to images and calls UNet on it to denoise.
struct DenoisingDiffusionImplicitModel{T <: AbstractFloat} <:
       Lux.AbstractExplicitContainerLayer{(:unet, :batchnorm)}
    unet::UNet
    batchnorm::BatchNorm
    min_signal_rate::T
    max_signal_rate::T
end

function DenoisingDiffusionImplicitModel(image_size::Tuple{Int, Int};
                                         channels=[32, 64, 96, 128], block_depth=2,
                                         min_freq=1.0f0, max_freq=1000.0f0,
                                         embedding_dims=32, min_signal_rate=0.02f0,
                                         max_signal_rate=0.95f0)
    unet = UNet(image_size; channels=channels, block_depth=block_depth, min_freq=min_freq,
                max_freq=max_freq, embedding_dims=embedding_dims)
    batchnorm = BatchNorm(3; affine=false, momentum=0.99, track_stats=true)

    return DenoisingDiffusionImplicitModel(unet, batchnorm, min_signal_rate,
                                           max_signal_rate)
end

function (ddim::DenoisingDiffusionImplicitModel{T})(x::Tuple{AbstractArray{T, 4},
                                                             AbstractRNG}, ps,
                                                    st::NamedTuple) where {
                                                                           T <:
                                                                           AbstractFloat}
    images, rng = x
    images, new_st = ddim.batchnorm(images, ps.batchnorm, st.batchnorm)
    @set! st.batchnorm = new_st

    noises = randn(rng, eltype(images), size(images)...) |> gpu

    diffusion_times = rand(rng, eltype(images), 1, 1, 1, size(images, 4)) |> gpu
    noise_rates, signal_rates = diffusion_schedules(diffusion_times, ddim.min_signal_rate,
                                                    ddim.max_signal_rate)

    noisy_images = signal_rates .* images + noise_rates .* noises

    (pred_noises, pred_images), st = denoise(ddim, noisy_images, noise_rates, signal_rates,
                                             ps, st)

    return (noises, images, pred_noises, pred_images), st
end

# Generates noise with variable magnitude depending on time.
# Noise is at minimum at t=0, and maximum at t=1.
function diffusion_schedules(diffusion_times::AbstractArray{T, 4}, min_signal_rate::T,
                             max_signal_rate::T) where {T <: AbstractFloat}
    start_angle = acos(max_signal_rate)
    end_angle = acos(min_signal_rate)

    diffusion_angles = start_angle .+ (end_angle - start_angle) * diffusion_times

    # see Eq. (12) in 2010.02502 with sigma=0
    signal_rates = cos.(diffusion_angles) # sqrt{alpha_t}
    noise_rates = sin.(diffusion_angles) # sqrt{1-alpha_t}

    return noise_rates, signal_rates
end

function denoise(ddim::DenoisingDiffusionImplicitModel{T},
                 noisy_images::AbstractArray{T, 4}, noise_rates::AbstractArray{T, 4},
                 signal_rates::AbstractArray{T, 4}, ps,
                 st::NamedTuple) where {T <: AbstractFloat}
    pred_noises, new_st = ddim.unet((noisy_images, noise_rates .^ 2), ps.unet, st.unet)
    @set! st.unet = new_st

    pred_images = (noisy_images - pred_noises .* noise_rates) ./ signal_rates

    return (pred_noises, pred_images), st
end

function reverse_diffusion(ddim::DenoisingDiffusionImplicitModel{T},
                           initial_noise::AbstractArray{T, 4}, diffusion_steps::Int, ps,
                           st::NamedTuple; save_each_step=false) where {T <: AbstractFloat}
    num_images = size(initial_noise, 4)
    step_size = convert(T, 1.0) / diffusion_steps

    next_noisy_images = initial_noise
    pred_images = nothing

    # save intermediate images at each step for inference
    images_each_step = ifelse(save_each_step, [initial_noise], nothing)

    for step in 1:diffusion_steps
        noisy_images = next_noisy_images

        # We start t = 1, and gradually decreases to t=0
        diffusion_times = ones(T, 1, 1, 1, num_images) .- step_size * step |> gpu

        noise_rates, signal_rates = diffusion_schedules(diffusion_times,
                                                        ddim.min_signal_rate,
                                                        ddim.max_signal_rate)

        (pred_noises, pred_images), _ = denoise(ddim, noisy_images, noise_rates,
                                                signal_rates, ps, st)

        next_diffusion_times = diffusion_times .- step_size
        next_noise_rates, next_signal_rates = diffusion_schedules(next_diffusion_times,
                                                                  ddim.min_signal_rate,
                                                                  ddim.max_signal_rate)

        # see Eq. (12) in 2010.02502 with sigma=0
        next_noisy_images = next_signal_rates .* pred_images +
                            next_noise_rates .* pred_noises

        if save_each_step
            push!(images_each_step, pred_images)
        end
    end

    return pred_images, images_each_step
end

function denormalize(ddim::DenoisingDiffusionImplicitModel{T}, x::AbstractArray{T, 4},
                     st) where {T <: AbstractFloat}
    mean = reshape(st.running_mean, 1, 1, 3, 1)
    var = reshape(st.running_var, 1, 1, 3, 1)
    std = sqrt.(var .+ ddim.batchnorm.epsilon)
    return std .* x .+ mean
end

function generate(ddim::DenoisingDiffusionImplicitModel{T}, rng::AbstractRNG,
                  image_shape::Tuple{Int, Int, Int, Int}, diffusion_steps::Int, ps,
                  st::NamedTuple; save_each_step=false) where {T}
    initial_noise = randn(rng, T, image_shape...) |> gpu
    generated_images, images_each_step = reverse_diffusion(ddim, initial_noise,
                                                           diffusion_steps, ps, st;
                                                           save_each_step=save_each_step)
    generated_images = denormalize(ddim, generated_images, st.batchnorm)
    clamp!(generated_images, 0.0f0, 1.0f0)

    if !isnothing(images_each_step)
        for (i, images) in enumerate(images_each_step)
            images_each_step[i] = denormalize(ddim, images, st.batchnorm)
            clamp!(images_each_step[i], 0.0f0, 1.0f0)
        end
    end
    return generated_images, images_each_step
end
