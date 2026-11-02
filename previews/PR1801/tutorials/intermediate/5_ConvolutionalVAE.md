---
url: /previews/PR1801/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24193.1835938, Throughput: 18.357468 im/s
Epoch 1, Train Loss: 39575.5703125, Time: 272.1342s, Throughput: 18.343889 im/s
Epoch 2, Iter 39, Loss: 18972.8671875, Throughput: 206.556693 im/s
Epoch 2, Train Loss: 20161.4511719, Time: 24.1679s, Throughput: 206.555275 im/s
Epoch 3, Iter 39, Loss: 16236.0683594, Throughput: 206.603404 im/s
Epoch 3, Train Loss: 16629.9140625, Time: 24.1624s, Throughput: 206.602028 im/s
Epoch 4, Iter 39, Loss: 14125.9150391, Throughput: 206.107071 im/s
Epoch 4, Train Loss: 15048.8769531, Time: 24.2206s, Throughput: 206.105769 im/s
Epoch 5, Iter 39, Loss: 14042.0976562, Throughput: 206.839219 im/s
Epoch 5, Train Loss: 14081.9472656, Time: 24.1348s, Throughput: 206.837864 im/s
Epoch 6, Iter 39, Loss: 13273.0117188, Throughput: 206.304613 im/s
Epoch 6, Train Loss: 13448.0996094, Time: 24.1974s, Throughput: 206.303070 im/s
Epoch 7, Iter 39, Loss: 13064.4140625, Throughput: 206.958369 im/s
Epoch 7, Train Loss: 12861.9941406, Time: 24.1210s, Throughput: 206.956902 im/s
Epoch 8, Iter 39, Loss: 12430.6269531, Throughput: 206.713716 im/s
Epoch 8, Train Loss: 12509.9687500, Time: 24.1495s, Throughput: 206.712441 im/s
Epoch 9, Iter 39, Loss: 12222.4042969, Throughput: 206.840615 im/s
Epoch 9, Train Loss: 12200.2812500, Time: 24.1347s, Throughput: 206.839321 im/s
Epoch 10, Iter 39, Loss: 12700.4326172, Throughput: 206.215071 im/s
Epoch 10, Train Loss: 11958.2910156, Time: 24.2079s, Throughput: 206.213751 im/s
Epoch 11, Iter 39, Loss: 11462.6904297, Throughput: 206.593425 im/s
Epoch 11, Train Loss: 11670.5849609, Time: 24.1636s, Throughput: 206.592117 im/s
Epoch 12, Iter 39, Loss: 11159.3535156, Throughput: 206.408779 im/s
Epoch 12, Train Loss: 11538.9121094, Time: 24.1852s, Throughput: 206.406986 im/s
Epoch 13, Iter 39, Loss: 10822.5039062, Throughput: 205.961911 im/s
Epoch 13, Train Loss: 11394.2041016, Time: 24.2376s, Throughput: 205.960629 im/s
Epoch 14, Iter 39, Loss: 11073.8945312, Throughput: 205.920334 im/s
Epoch 14, Train Loss: 11284.3056641, Time: 24.2425s, Throughput: 205.919026 im/s
Epoch 15, Iter 39, Loss: 11417.3466797, Throughput: 205.460458 im/s
Epoch 15, Train Loss: 11104.7236328, Time: 24.2968s, Throughput: 205.459140 im/s
Epoch 16, Iter 39, Loss: 11058.0527344, Throughput: 205.473034 im/s
Epoch 16, Train Loss: 10917.0332031, Time: 24.2953s, Throughput: 205.471816 im/s
Epoch 17, Iter 39, Loss: 11867.3496094, Throughput: 205.273641 im/s
Epoch 17, Train Loss: 10834.6582031, Time: 24.3189s, Throughput: 205.272375 im/s
Epoch 18, Iter 39, Loss: 11536.4482422, Throughput: 203.253829 im/s
Epoch 18, Train Loss: 10767.2236328, Time: 24.5606s, Throughput: 203.252464 im/s
Epoch 19, Iter 39, Loss: 10579.3535156, Throughput: 205.683905 im/s
Epoch 19, Train Loss: 10676.6240234, Time: 24.2704s, Throughput: 205.682583 im/s
Epoch 20, Iter 39, Loss: 10255.0263672, Throughput: 206.342655 im/s
Epoch 20, Train Loss: 10583.4355469, Time: 24.1929s, Throughput: 206.341392 im/s
Epoch 21, Iter 39, Loss: 10335.6914062, Throughput: 205.806887 im/s
Epoch 21, Train Loss: 10480.7578125, Time: 24.2559s, Throughput: 205.805487 im/s
Epoch 22, Iter 39, Loss: 10396.2207031, Throughput: 206.077178 im/s
Epoch 22, Train Loss: 10440.7792969, Time: 24.2241s, Throughput: 206.075793 im/s
Epoch 23, Iter 39, Loss: 10204.3632812, Throughput: 205.017820 im/s
Epoch 23, Train Loss: 10350.0478516, Time: 24.3493s, Throughput: 205.016304 im/s
Epoch 24, Iter 39, Loss: 9930.4687500, Throughput: 205.426444 im/s
Epoch 24, Train Loss: 10283.0019531, Time: 24.3008s, Throughput: 205.424998 im/s
Epoch 25, Iter 39, Loss: 10237.4335938, Throughput: 203.450217 im/s
Epoch 25, Train Loss: 10289.6396484, Time: 24.5369s, Throughput: 203.448908 im/s
Epoch 26, Iter 39, Loss: 10266.7929688, Throughput: 204.994373 im/s
Epoch 26, Train Loss: 10309.5957031, Time: 24.3520s, Throughput: 204.993042 im/s
Epoch 27, Iter 39, Loss: 10303.9404297, Throughput: 204.245992 im/s
Epoch 27, Train Loss: 10138.5097656, Time: 24.4413s, Throughput: 204.244587 im/s
Epoch 28, Iter 39, Loss: 10134.8105469, Throughput: 205.421455 im/s
Epoch 28, Train Loss: 10075.9541016, Time: 24.3014s, Throughput: 205.420256 im/s
Epoch 29, Iter 39, Loss: 10585.6083984, Throughput: 205.175740 im/s
Epoch 29, Train Loss: 9993.0244141, Time: 24.3305s, Throughput: 205.174558 im/s
Epoch 30, Iter 39, Loss: 9908.5996094, Throughput: 204.881980 im/s
Epoch 30, Train Loss: 10004.7646484, Time: 24.3654s, Throughput: 204.880643 im/s
Epoch 31, Iter 39, Loss: 10238.2148438, Throughput: 205.297194 im/s
Epoch 31, Train Loss: 9914.8847656, Time: 24.3161s, Throughput: 205.295962 im/s
Epoch 32, Iter 39, Loss: 9775.7939453, Throughput: 205.161615 im/s
Epoch 32, Train Loss: 9893.4794922, Time: 24.3322s, Throughput: 205.160383 im/s
Epoch 33, Iter 39, Loss: 9348.5175781, Throughput: 205.099566 im/s
Epoch 33, Train Loss: 9946.5917969, Time: 24.3396s, Throughput: 205.098210 im/s
Epoch 34, Iter 39, Loss: 9671.0615234, Throughput: 205.543380 im/s
Epoch 34, Train Loss: 9879.2851562, Time: 24.2870s, Throughput: 205.542110 im/s
Epoch 35, Iter 39, Loss: 9333.1162109, Throughput: 205.940679 im/s
Epoch 35, Train Loss: 9830.4619141, Time: 24.2402s, Throughput: 205.939302 im/s
Epoch 36, Iter 39, Loss: 9655.6142578, Throughput: 202.494258 im/s
Epoch 36, Train Loss: 9714.6015625, Time: 24.6527s, Throughput: 202.492836 im/s
Epoch 37, Iter 39, Loss: 9397.6103516, Throughput: 205.412919 im/s
Epoch 37, Train Loss: 9648.0185547, Time: 24.3024s, Throughput: 205.411564 im/s
Epoch 38, Iter 39, Loss: 9939.3642578, Throughput: 204.106348 im/s
Epoch 38, Train Loss: 9685.2412109, Time: 24.4581s, Throughput: 204.104545 im/s
Epoch 39, Iter 39, Loss: 9652.0712891, Throughput: 205.308441 im/s
Epoch 39, Train Loss: 9618.7832031, Time: 24.3148s, Throughput: 205.307156 im/s
Epoch 40, Iter 39, Loss: 10177.9394531, Throughput: 206.734165 im/s
Epoch 40, Train Loss: 9537.7285156, Time: 24.1471s, Throughput: 206.732592 im/s
Epoch 41, Iter 39, Loss: 9456.7871094, Throughput: 205.341071 im/s
Epoch 41, Train Loss: 9535.8349609, Time: 24.3110s, Throughput: 205.339466 im/s
Epoch 42, Iter 39, Loss: 9441.3535156, Throughput: 205.992610 im/s
Epoch 42, Train Loss: 9521.9169922, Time: 24.2340s, Throughput: 205.991191 im/s
Epoch 43, Iter 39, Loss: 9158.2617188, Throughput: 205.337601 im/s
Epoch 43, Train Loss: 9469.0615234, Time: 24.3113s, Throughput: 205.336268 im/s
Epoch 44, Iter 39, Loss: 9385.1171875, Throughput: 206.456923 im/s
Epoch 44, Train Loss: 9417.3222656, Time: 24.1796s, Throughput: 206.455395 im/s
Epoch 45, Iter 39, Loss: 9246.3193359, Throughput: 205.193668 im/s
Epoch 45, Train Loss: 9395.6396484, Time: 24.3284s, Throughput: 205.192303 im/s
Epoch 46, Iter 39, Loss: 8994.5039062, Throughput: 205.315078 im/s
Epoch 46, Train Loss: 9390.2060547, Time: 24.3140s, Throughput: 205.313796 im/s
Epoch 47, Iter 39, Loss: 10079.5390625, Throughput: 205.316850 im/s
Epoch 47, Train Loss: 9342.9345703, Time: 24.3138s, Throughput: 205.315525 im/s
Epoch 48, Iter 39, Loss: 9750.4892578, Throughput: 205.057139 im/s
Epoch 48, Train Loss: 9355.1220703, Time: 24.3447s, Throughput: 205.055211 im/s
Epoch 49, Iter 39, Loss: 9719.5615234, Throughput: 205.233242 im/s
Epoch 49, Train Loss: 9432.9345703, Time: 24.3237s, Throughput: 205.232027 im/s
Epoch 50, Iter 39, Loss: 9250.0859375, Throughput: 205.370328 im/s
Epoch 50, Train Loss: 9340.9033203, Time: 24.3075s, Throughput: 205.368992 im/s

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
