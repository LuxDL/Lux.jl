---
url: /dev/tutorials/intermediate/5_ConvolutionalVAE.md
---

# Convolutional VAE for MNIST {#Convolutional-VAE-Tutorial}

Convolutional variational autoencoder (CVAE) implementation in MLX using MNIST. This is based on the [CVAE implementation in MLX](https://github.com/ml-explore/mlx-examples/blob/main/cvae/).

```julia
using Lux,
    Reactant,
    MLDatasets,
    Random,
    Statistics,
    Enzyme,
    MLUtils,
    DataAugmentation,
    ConcreteStructs,
    OneHotArrays,
    ImageShow,
    Images,
    Printf,
    Optimisers

const xdev = reactant_device(; force=true)
const cdev = cpu_device()

const IN_VSCODE = isdefined(Main, :VSCodeServer)
```

```
false
```

## Model Definition {#Model-Definition}

First we will define the encoder.It maps the input to a normal distribution in latent space and sample a latent vector from that distribution.

```julia
function cvae_encoder(
    rng=Random.default_rng();
    num_latent_dims::Int,
    image_shape::Dims{3},
    max_num_filters::Int,
)
    flattened_dim = prod(image_shape[1:2] .÷ 8) * max_num_filters
    return @compact(;
        embed=Chain(
            Chain(
                Conv((3, 3), image_shape[3] => max_num_filters ÷ 4; stride=2, pad=1),
                BatchNorm(max_num_filters ÷ 4, leakyrelu),
            ),
            Chain(
                Conv((3, 3), max_num_filters ÷ 4 => max_num_filters ÷ 2; stride=2, pad=1),
                BatchNorm(max_num_filters ÷ 2, leakyrelu),
            ),
            Chain(
                Conv((3, 3), max_num_filters ÷ 2 => max_num_filters; stride=2, pad=1),
                BatchNorm(max_num_filters, leakyrelu),
            ),
            FlattenLayer(),
        ),
        proj_mu=Dense(flattened_dim, num_latent_dims; init_bias=zeros32),
        proj_log_var=Dense(flattened_dim, num_latent_dims; init_bias=zeros32),
        rng
    ) do x
        y = embed(x)

        μ = proj_mu(y)
        logσ² = proj_log_var(y)

        T = eltype(logσ²)
        logσ² = clamp.(logσ², -T(20.0f0), T(10.0f0))
        σ = exp.(logσ² .* T(0.5))

        # Generate a tensor of random values from a normal distribution
        ϵ = randn_like(Lux.replicate(rng), σ)

        # Reparameterization trick to backpropagate through sampling
        z = ϵ .* σ .+ μ

        @return z, μ, logσ²
    end
end
```

Similarly we define the decoder.

```julia
function cvae_decoder(; num_latent_dims::Int, image_shape::Dims{3}, max_num_filters::Int)
    flattened_dim = prod(image_shape[1:2] .÷ 8) * max_num_filters
    return @compact(;
        linear=Dense(num_latent_dims, flattened_dim),
        upchain=Chain(
            Chain(
                Upsample(2),
                Conv((3, 3), max_num_filters => max_num_filters ÷ 2; stride=1, pad=1),
                BatchNorm(max_num_filters ÷ 2, leakyrelu),
            ),
            Chain(
                Upsample(2),
                Conv((3, 3), max_num_filters ÷ 2 => max_num_filters ÷ 4; stride=1, pad=1),
                BatchNorm(max_num_filters ÷ 4, leakyrelu),
            ),
            Chain(
                Upsample(2),
                Conv(
                    (3, 3), max_num_filters ÷ 4 => image_shape[3], sigmoid; stride=1, pad=1
                ),
            ),
        ),
        max_num_filters
    ) do x
        y = linear(x)
        img = reshape(y, image_shape[1] ÷ 8, image_shape[2] ÷ 8, max_num_filters, :)
        @return upchain(img)
    end
end

@concrete struct CVAE <: AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder <: AbstractLuxLayer
    decoder <: AbstractLuxLayer
end

function CVAE(
    rng=Random.default_rng();
    num_latent_dims::Int,
    image_shape::Dims{3},
    max_num_filters::Int,
)
    decoder = cvae_decoder(; num_latent_dims, image_shape, max_num_filters)
    encoder = cvae_encoder(rng; num_latent_dims, image_shape, max_num_filters)
    return CVAE(encoder, decoder)
end

function (cvae::CVAE)(x, ps, st)
    (z, μ, logσ²), st_enc = cvae.encoder(x, ps.encoder, st.encoder)
    x_rec, st_dec = cvae.decoder(z, ps.decoder, st.decoder)
    return (x_rec, μ, logσ²), (; encoder=st_enc, decoder=st_dec)
end

function encode(cvae::CVAE, x, ps, st)
    (z, _, _), st_enc = cvae.encoder(x, ps.encoder, st.encoder)
    return z, (; encoder=st_enc, st.decoder)
end

function decode(cvae::CVAE, z, ps, st)
    x_rec, st_dec = cvae.decoder(z, ps.decoder, st.decoder)
    return x_rec, (; decoder=st_dec, st.encoder)
end
```

## Loading MNIST {#Loading-MNIST}

```julia
@concrete struct TensorDataset
    dataset
    transform
    total_samples::Int
end

Base.length(ds::TensorDataset) = ds.total_samples

function Base.getindex(ds::TensorDataset, idxs::Union{Vector{<:Integer},AbstractRange})
    img = Image.(eachslice(convert2image(ds.dataset, idxs); dims=3))
    return stack(parent ∘ itemdata ∘ Base.Fix1(apply, ds.transform), img)
end

function loadmnist(batchsize, image_size::Dims{2})
    # Load MNIST: Only 1500 for demonstration purposes on CI
    train_dataset = MNIST(; split=:train)
    N = parse(Bool, get(ENV, "CI", "false")) ? 5000 : length(train_dataset)

    train_transform = ScaleKeepAspect(image_size) |> ImageToTensor()
    trainset = TensorDataset(train_dataset, train_transform, N)
    trainloader = DataLoader(trainset; batchsize, shuffle=true, partial=false)

    return trainloader
end
```

## Helper Functions {#Helper-Functions}

Generate an Image Grid from a list of images

```julia
function create_image_grid(imgs::AbstractArray, grid_rows::Int, grid_cols::Int)
    total_images = grid_rows * grid_cols
    imgs = map(eachslice(imgs[:, :, :, 1:total_images]; dims=4)) do img
        cimg = if size(img, 3) == 1
            colorview(Gray, view(img, :, :, 1))
        else
            colorview(RGB, permutedims(img, (3, 1, 2)))
        end
        return cimg'
    end
    return create_image_grid(imgs, grid_rows, grid_cols)
end

function create_image_grid(images::Vector, grid_rows::Int, grid_cols::Int)
    # Check if the number of images matches the grid
    total_images = grid_rows * grid_cols
    @assert length(images) == total_images

    # Get the size of a single image (assuming all images are the same size)
    img_height, img_width = size(images[1])

    # Create a blank grid canvas
    grid_height = img_height * grid_rows
    grid_width = img_width * grid_cols
    grid_canvas = similar(images[1], grid_height, grid_width)

    # Place each image in the correct position on the canvas
    for idx in 1:total_images
        row = div(idx - 1, grid_cols) + 1
        col = mod(idx - 1, grid_cols) + 1

        start_row = (row - 1) * img_height + 1
        start_col = (col - 1) * img_width + 1

        grid_canvas[start_row:(start_row + img_height - 1), start_col:(start_col + img_width - 1)] .= images[idx]
    end

    return grid_canvas
end

function loss_function(model, ps, st, X)
    (y, μ, logσ²), st = model(X, ps, st)
    reconstruction_loss = MSELoss(; agg=sum)(y, X)
    kldiv_loss = -sum(1 .+ logσ² .- μ .^ 2 .- exp.(logσ²)) / 2
    loss = reconstruction_loss + kldiv_loss
    return loss, st, (; y, μ, logσ², reconstruction_loss, kldiv_loss)
end

function generate_images(
    model, ps, st; num_samples::Int=128, num_latent_dims::Int, decode_compiled=nothing
)
    z = get_device((ps, st))(randn(Float32, num_latent_dims, num_samples))
    if decode_compiled === nothing
        images, _ = decode(model, z, ps, Lux.testmode(st))
    else
        images, _ = decode_compiled(model, z, ps, Lux.testmode(st))
        images = cpu_device()(images)
    end
    return create_image_grid(images, 8, num_samples ÷ 8)
end

function reconstruct_images(model, ps, st, X)
    (recon, _, _), _ = model(X, ps, Lux.testmode(st))
    recon = cpu_device()(recon)
    return create_image_grid(recon, 8, size(X, ndims(X)) ÷ 8)
end
```

```
reconstruct_images (generic function with 1 method)
```

## Training the Model {#Training-the-Model}

```julia
function main(;
    batchsize=128,
    image_size=(64, 64),
    num_latent_dims=8,
    max_num_filters=64,
    seed=0,
    epochs=50,
    weight_decay=1.0e-5,
    learning_rate=1.0e-3,
    num_samples=batchsize,
)
    rng = Xoshiro()
    Random.seed!(rng, seed)

    cvae = CVAE(rng; num_latent_dims, image_shape=(image_size..., 1), max_num_filters)
    ps, st = Lux.setup(rng, cvae) |> xdev

    z = xdev(randn(Float32, num_latent_dims, num_samples))
    decode_compiled = @compile decode(cvae, z, ps, Lux.testmode(st))
    x = randn(Float32, image_size..., 1, batchsize) |> xdev
    cvae_compiled = @compile cvae(x, ps, Lux.testmode(st))

    train_dataloader = loadmnist(batchsize, image_size) |> xdev

    opt = AdamW(; eta=learning_rate, lambda=weight_decay)

    train_state = Training.TrainState(cvae, ps, st, opt)

    @printf "Total Trainable Parameters: %0.4f M\n" (Lux.parameterlength(ps) / 1.0e6)

    empty_row, model_img_full = nothing, nothing

    for epoch in 1:epochs
        loss_total = 0.0f0
        total_samples = 0

        start_time = time()
        for (i, X) in enumerate(train_dataloader)
            (_, loss, _, train_state) = Training.single_train_step!(
                AutoEnzyme(), loss_function, X, train_state; return_gradients=Val(false)
            )

            loss_total += loss
            total_samples += size(X, ndims(X))

            if i % 250 == 0 || i == length(train_dataloader)
                throughput = total_samples / (time() - start_time)
                @printf "Epoch %d, Iter %d, Loss: %.7f, Throughput: %.6f im/s\n" epoch i loss throughput
            end
        end
        total_time = time() - start_time

        train_loss = loss_total / length(train_dataloader)
        throughput = total_samples / total_time
        @printf "Epoch %d, Train Loss: %.7f, Time: %.4fs, Throughput: %.6f im/s\n" epoch train_loss total_time throughput

        if IN_VSCODE || epoch == epochs
            recon_images = reconstruct_images(
                cvae_compiled,
                train_state.parameters,
                train_state.states,
                first(train_dataloader),
            )
            gen_images = generate_images(
                cvae,
                train_state.parameters,
                train_state.states;
                num_samples,
                num_latent_dims,
                decode_compiled,
            )
            if empty_row === nothing
                empty_row = similar(gen_images, image_size[1], size(gen_images, 2))
                fill!(empty_row, 0)
            end
            model_img_full = vcat(recon_images, empty_row, gen_images)
            IN_VSCODE && display(model_img_full)
        end
    end

    return model_img_full
end

img = main()
```

```
Total Trainable Parameters: 0.1493 M
Epoch 1, Iter 39, Loss: 24372.9042969, Throughput: 11.562854 im/s
Epoch 1, Train Loss: 39714.4960938, Time: 432.0219s, Throughput: 11.554971 im/s
Epoch 2, Iter 39, Loss: 17505.5859375, Throughput: 98.262393 im/s
Epoch 2, Train Loss: 20095.0214844, Time: 50.8030s, Throughput: 98.261993 im/s
Epoch 3, Iter 39, Loss: 16085.7988281, Throughput: 98.417893 im/s
Epoch 3, Train Loss: 16551.1601562, Time: 50.7227s, Throughput: 98.417513 im/s
Epoch 4, Iter 39, Loss: 15223.7001953, Throughput: 97.710641 im/s
Epoch 4, Train Loss: 14971.8896484, Time: 51.0906s, Throughput: 97.708752 im/s
Epoch 5, Iter 39, Loss: 13532.3330078, Throughput: 99.125505 im/s
Epoch 5, Train Loss: 14034.1074219, Time: 50.3606s, Throughput: 99.125116 im/s
Epoch 6, Iter 39, Loss: 13642.3681641, Throughput: 99.182665 im/s
Epoch 6, Train Loss: 13311.8203125, Time: 50.3316s, Throughput: 99.182277 im/s
Epoch 7, Iter 39, Loss: 12322.7324219, Throughput: 99.347516 im/s
Epoch 7, Train Loss: 12947.8750000, Time: 50.2481s, Throughput: 99.347066 im/s
Epoch 8, Iter 39, Loss: 12115.5947266, Throughput: 99.140967 im/s
Epoch 8, Train Loss: 12421.3574219, Time: 50.3528s, Throughput: 99.140513 im/s
Epoch 9, Iter 39, Loss: 12022.9667969, Throughput: 98.573957 im/s
Epoch 9, Train Loss: 12132.7138672, Time: 50.6424s, Throughput: 98.573557 im/s
Epoch 10, Iter 39, Loss: 11882.0517578, Throughput: 98.384739 im/s
Epoch 10, Train Loss: 11943.0673828, Time: 50.7398s, Throughput: 98.384353 im/s
Epoch 11, Iter 39, Loss: 11284.8320312, Throughput: 98.698319 im/s
Epoch 11, Train Loss: 11776.0068359, Time: 50.5786s, Throughput: 98.697890 im/s
Epoch 12, Iter 39, Loss: 11614.9482422, Throughput: 98.238999 im/s
Epoch 12, Train Loss: 11542.2988281, Time: 50.8150s, Throughput: 98.238614 im/s
Epoch 13, Iter 39, Loss: 11185.0458984, Throughput: 97.627580 im/s
Epoch 13, Train Loss: 11396.4052734, Time: 51.1333s, Throughput: 97.627188 im/s
Epoch 14, Iter 39, Loss: 10660.5800781, Throughput: 99.434746 im/s
Epoch 14, Train Loss: 11229.9355469, Time: 50.2040s, Throughput: 99.434291 im/s
Epoch 15, Iter 39, Loss: 10423.9121094, Throughput: 100.276783 im/s
Epoch 15, Train Loss: 11161.9462891, Time: 49.7824s, Throughput: 100.276322 im/s
Epoch 16, Iter 39, Loss: 11370.2812500, Throughput: 100.539319 im/s
Epoch 16, Train Loss: 11015.1582031, Time: 49.6524s, Throughput: 100.538937 im/s
Epoch 17, Iter 39, Loss: 10732.5517578, Throughput: 100.202845 im/s
Epoch 17, Train Loss: 10767.1601562, Time: 49.8192s, Throughput: 100.202412 im/s
Epoch 18, Iter 39, Loss: 10289.4277344, Throughput: 98.215311 im/s
Epoch 18, Train Loss: 10800.5498047, Time: 50.8273s, Throughput: 98.214868 im/s
Epoch 19, Iter 39, Loss: 10597.3017578, Throughput: 98.037911 im/s
Epoch 19, Train Loss: 10661.7753906, Time: 50.9193s, Throughput: 98.037574 im/s
Epoch 20, Iter 39, Loss: 10807.9208984, Throughput: 98.417107 im/s
Epoch 20, Train Loss: 10572.5332031, Time: 50.7231s, Throughput: 98.416665 im/s
Epoch 21, Iter 39, Loss: 10950.4746094, Throughput: 98.288137 im/s
Epoch 21, Train Loss: 10534.0966797, Time: 50.7897s, Throughput: 98.287705 im/s
Epoch 22, Iter 39, Loss: 10375.5830078, Throughput: 98.098435 im/s
Epoch 22, Train Loss: 10363.3974609, Time: 50.8879s, Throughput: 98.098024 im/s
Epoch 23, Iter 39, Loss: 10328.7148438, Throughput: 99.131332 im/s
Epoch 23, Train Loss: 10327.9667969, Time: 50.3576s, Throughput: 99.130928 im/s
Epoch 24, Iter 39, Loss: 10260.6005859, Throughput: 98.374399 im/s
Epoch 24, Train Loss: 10415.9882812, Time: 50.7451s, Throughput: 98.373993 im/s
Epoch 25, Iter 39, Loss: 10527.6386719, Throughput: 98.118756 im/s
Epoch 25, Train Loss: 10280.7255859, Time: 50.8773s, Throughput: 98.118368 im/s
Epoch 26, Iter 39, Loss: 9967.4863281, Throughput: 99.122297 im/s
Epoch 26, Train Loss: 10149.4531250, Time: 50.3622s, Throughput: 99.121937 im/s
Epoch 27, Iter 39, Loss: 10079.5107422, Throughput: 97.536744 im/s
Epoch 27, Train Loss: 10057.7617188, Time: 51.1809s, Throughput: 97.536359 im/s
Epoch 28, Iter 39, Loss: 9764.3027344, Throughput: 98.548598 im/s
Epoch 28, Train Loss: 10055.7343750, Time: 50.6554s, Throughput: 98.548216 im/s
Epoch 29, Iter 39, Loss: 10065.3310547, Throughput: 98.227969 im/s
Epoch 29, Train Loss: 10008.4960938, Time: 50.8208s, Throughput: 98.227544 im/s
Epoch 30, Iter 39, Loss: 9825.8710938, Throughput: 98.482365 im/s
Epoch 30, Train Loss: 9964.5244141, Time: 50.6895s, Throughput: 98.481998 im/s
Epoch 31, Iter 39, Loss: 10178.8222656, Throughput: 99.492219 im/s
Epoch 31, Train Loss: 9878.9287109, Time: 50.1750s, Throughput: 99.491809 im/s
Epoch 32, Iter 39, Loss: 10290.6835938, Throughput: 99.399931 im/s
Epoch 32, Train Loss: 9882.0498047, Time: 50.2215s, Throughput: 99.399562 im/s
Epoch 33, Iter 39, Loss: 9788.9082031, Throughput: 98.464112 im/s
Epoch 33, Train Loss: 9833.0107422, Time: 50.6989s, Throughput: 98.463735 im/s
Epoch 34, Iter 39, Loss: 10246.5039062, Throughput: 97.908051 im/s
Epoch 34, Train Loss: 9862.3105469, Time: 50.9868s, Throughput: 97.907609 im/s
Epoch 35, Iter 39, Loss: 10017.0146484, Throughput: 98.695135 im/s
Epoch 35, Train Loss: 9789.9326172, Time: 50.5802s, Throughput: 98.694743 im/s
Epoch 36, Iter 39, Loss: 9162.5458984, Throughput: 98.335093 im/s
Epoch 36, Train Loss: 9748.3056641, Time: 50.7654s, Throughput: 98.334673 im/s
Epoch 37, Iter 39, Loss: 9377.5224609, Throughput: 99.667468 im/s
Epoch 37, Train Loss: 9675.8486328, Time: 50.0867s, Throughput: 99.667116 im/s
Epoch 38, Iter 39, Loss: 10318.0703125, Throughput: 99.682275 im/s
Epoch 38, Train Loss: 9655.0292969, Time: 50.0793s, Throughput: 99.681893 im/s
Epoch 39, Iter 39, Loss: 10008.7773438, Throughput: 98.342005 im/s
Epoch 39, Train Loss: 9644.2529297, Time: 50.7618s, Throughput: 98.341597 im/s
Epoch 40, Iter 39, Loss: 9559.2480469, Throughput: 98.671594 im/s
Epoch 40, Train Loss: 9644.7216797, Time: 50.5922s, Throughput: 98.671274 im/s
Epoch 41, Iter 39, Loss: 9984.8320312, Throughput: 96.520251 im/s
Epoch 41, Train Loss: 9588.2626953, Time: 51.7199s, Throughput: 96.519859 im/s
Epoch 42, Iter 39, Loss: 10215.9462891, Throughput: 98.068762 im/s
Epoch 42, Train Loss: 9599.9345703, Time: 50.9033s, Throughput: 98.068381 im/s
Epoch 43, Iter 39, Loss: 9374.7685547, Throughput: 97.367939 im/s
Epoch 43, Train Loss: 9488.4794922, Time: 51.2696s, Throughput: 97.367576 im/s
Epoch 44, Iter 39, Loss: 9309.9628906, Throughput: 96.941848 im/s
Epoch 44, Train Loss: 9413.8720703, Time: 51.4950s, Throughput: 96.941524 im/s
Epoch 45, Iter 39, Loss: 9505.3046875, Throughput: 97.262315 im/s
Epoch 45, Train Loss: 9411.3535156, Time: 51.3253s, Throughput: 97.261904 im/s
Epoch 46, Iter 39, Loss: 9341.1953125, Throughput: 97.857660 im/s
Epoch 46, Train Loss: 9332.6464844, Time: 51.0131s, Throughput: 97.857282 im/s
Epoch 47, Iter 39, Loss: 9133.4531250, Throughput: 97.807777 im/s
Epoch 47, Train Loss: 9388.8056641, Time: 51.0391s, Throughput: 97.807423 im/s
Epoch 48, Iter 39, Loss: 9633.8554688, Throughput: 96.775783 im/s
Epoch 48, Train Loss: 9333.2119141, Time: 51.5834s, Throughput: 96.775398 im/s
Epoch 49, Iter 39, Loss: 9336.5371094, Throughput: 98.601933 im/s
Epoch 49, Train Loss: 9314.8300781, Time: 50.6280s, Throughput: 98.601492 im/s
Epoch 50, Iter 39, Loss: 9300.1279297, Throughput: 97.219529 im/s
Epoch 50, Train Loss: 9267.1162109, Time: 51.3479s, Throughput: 97.219120 im/s

```

***

## Appendix {#Appendix}

```julia
using InteractiveUtils
InteractiveUtils.versioninfo()

if @isdefined(MLDataDevices)
    if @isdefined(CUDA) && MLDataDevices.functional(CUDADevice)
        println()
        CUDA.versioninfo()
    end

    if @isdefined(AMDGPU) && MLDataDevices.functional(AMDGPUDevice)
        println()
        AMDGPU.versioninfo()
    end
end

```

```
Julia Version 1.12.7
Commit 6d172b025e4 (2026-08-15 08:05 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × AMD EPYC 9V74 80-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver4)
  GC: Built with stock GC
Threads: 4 default, 1 interactive, 4 GC (on 4 virtual cores)
Environment:
  JULIA_DEBUG = Literate
  LD_LIBRARY_PATH = 
  JULIA_NUM_THREADS = 4
  JULIA_CPU_HARD_MEMORY_LIMIT = 100%
  JULIA_PKG_PRECOMPILE_AUTO = 0

```

***

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
