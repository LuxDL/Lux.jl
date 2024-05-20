# # Denoising Diffusion Implicit Model (DDIM)

# [Lux.jl](https://github.com/LuxDL/Lux.jl) implementation of Denoising Diffusion Implicit
# Models ([arXiv:2010.02502](https://arxiv.org/abs/2010.02502)).
# The model generates images from Gaussian noises by <em>denoising</em> iteratively.

# ## Package Imports

using ArgCheck, ChainRulesCore, ConcreteStructs, DataAugmentation, DataDeps, Images, Lux,
      LuxCUDA, Random, Setfield
const CRC = ChainRulesCore

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
    freqs = exp.(lower:d:upper) |> get_device(x)

    @argcheck length(freqs) == n && size(freqs) == (n,)

    angular_speeds = reshape(T(2π) .* freqs, (1, 1, n, 1))
    embeddings = cat(sin.(angular_speeds .* x), cos.(angular_speeds .* x); dims=Val(3))
    return embeddings
end

function residual_block(in_channels::Int, out_channels::Int)
    first_layer = in_channels == out_channels ? NoOpLayer() :
                  Conv((3, 3), in_channels => out_channels; pad=SamePad())

    return Chain(first_layer,
        SkipConnection(
            Chain(BatchNorm(out_channels; affine=false, momentum=0.99f0),
                Conv((3, 3), out_channels => out_channels, swish; stride=1, pad=(1, 1)),
                Conv((3, 3), out_channels => out_channels; stride=1, pad=(1, 1))),
            +);
        name="ResidualBlock(in_chs=$in_channels, out_chs=$out_channels)")
end

function downsample_block(in_channels::Int, out_channels::Int, block_depth::Int)
    return @compact(;
        name="DownsampleBlock(in_chs=$in_channels, out_chs=$out_channels, block_depth=$block_depth)",
        residual_blocks=Tuple(residual_block(
                                  ifelse(i == 1, in_channels, out_channels), out_channels)
        for i in 1:block_depth),
        maxpool=MaxPool((2, 2); pad=0), block_depth) do x
        skips = (x,)
        for i in 1:block_depth
            skips = (skips..., residual_blocks[i](last(skips)))
        end
        y = maxpool(last(skips))
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
    bn = BatchNorm(3; affine=false, momentum=0.99f0, track_stats=true)

    return @compact(; unet, bn, rng, min_signal_rate,
        max_signal_rate, dispatch=:DDIM) do x::AbstractArray{<:Real, 4}
        images = bn(x)
        rng = Lux.replicate(rng)
        T = eltype(x)

        noises = CRC.@ignore_derivatives randn!(rng, similar(images, T, size(images)...))
        diffusion_times = CRC.@ignore_derivatives rand!(
            rng, similar(images, T, 1, 1, 1, size(images, 4)))

        noise_rates, signal_rates = __diffusion_schedules(
            diffusion_times, min_signal_rate, max_signal_rate)

        noisy_images = @. signal_rates * images + noise_rates * noises

        pred_noises, pred_images = __denoise(unet, noisy_images, noise_rates, signal_rates)

        @return noises, images, pred_noises, pred_images
    end
end

@inline function __diffusion_schedules(
        diffusion_times::AbstractArray{T, 4}, min_signal_rate::T,
        max_signal_rate::T) where {T <: Real}
    start_angle = acos(max_signal_rate)
    end_angle = acos(min_signal_rate)

    diffusion_angles = @. start_angle + (end_angle - start_angle) * diffusion_times

    signal_rates = @. cos(diffusion_angles)
    noise_rates = @. sin(diffusion_angles)

    return noise_rates, signal_rates
end

@inline function __denoise(
        unet, noisy_images::AbstractArray{T, 4}, noise_rates::AbstractArray{T, 4},
        signal_rates::AbstractArray{T, 4}) where {T <: Real}
    pred_noises = unet((noisy_images, noise_rates .^ 2))
    pred_images = @. (noisy_images - pred_noises * noise_rates) / signal_rates
    return pred_noises, pred_images
end

# ## Dataset

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
    img = Images.load(ds.image_files[i])
    img = ds.preprocess(img)
    img = permutedims(channelview(img), (2, 3, 1))
    ds.use_cache && (ds.cache[i] = img)
    return convert(AbstractArray{Float32}, img)
end

# function preprocess_image(image::Matrix{RGB{T}}, image_size::Int) where {T <: Real}
#     sigma = min(size(image)...) / image_size
#     k = round(Int, 2 * sigma) * 2 + 1 # kernel size of two sigma in each direction
#     pl = CropRatio(1.0) |> GaussianBlur(k, sigma) |> Resize(image_size, image_size)
#     return augment(image, pl)
# end
