---
url: /previews/PR1804/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 22873.5058594, Throughput: 13.100498 im/s
Epoch 1, Train Loss: 39363.6406250, Time: 381.3263s, Throughput: 13.091152 im/s
Epoch 2, Iter 39, Loss: 17439.3906250, Throughput: 96.758066 im/s
Epoch 2, Train Loss: 19992.7031250, Time: 51.5928s, Throughput: 96.757730 im/s
Epoch 3, Iter 39, Loss: 15843.6484375, Throughput: 97.156275 im/s
Epoch 3, Train Loss: 16468.2753906, Time: 51.3813s, Throughput: 97.155931 im/s
Epoch 4, Iter 39, Loss: 14945.1064453, Throughput: 96.435488 im/s
Epoch 4, Train Loss: 14935.0292969, Time: 51.7653s, Throughput: 96.435190 im/s
Epoch 5, Iter 39, Loss: 14013.6181641, Throughput: 96.701360 im/s
Epoch 5, Train Loss: 13937.3554688, Time: 51.6230s, Throughput: 96.701012 im/s
Epoch 6, Iter 39, Loss: 12833.0791016, Throughput: 96.742508 im/s
Epoch 6, Train Loss: 13425.4609375, Time: 51.6011s, Throughput: 96.742176 im/s
Epoch 7, Iter 39, Loss: 12813.6171875, Throughput: 97.031784 im/s
Epoch 7, Train Loss: 12884.5166016, Time: 51.4472s, Throughput: 97.031450 im/s
Epoch 8, Iter 39, Loss: 12139.5605469, Throughput: 96.750695 im/s
Epoch 8, Train Loss: 12503.2880859, Time: 51.5967s, Throughput: 96.750441 im/s
Epoch 9, Iter 39, Loss: 12201.2109375, Throughput: 97.005403 im/s
Epoch 9, Train Loss: 12182.6005859, Time: 51.4612s, Throughput: 97.005083 im/s
Epoch 10, Iter 39, Loss: 12596.9882812, Throughput: 96.863083 im/s
Epoch 10, Train Loss: 11966.9521484, Time: 51.5368s, Throughput: 96.862745 im/s
Epoch 11, Iter 39, Loss: 11679.9550781, Throughput: 96.964035 im/s
Epoch 11, Train Loss: 11696.6298828, Time: 51.4832s, Throughput: 96.963719 im/s
Epoch 12, Iter 39, Loss: 11772.5927734, Throughput: 97.183730 im/s
Epoch 12, Train Loss: 11576.7167969, Time: 51.3668s, Throughput: 97.183414 im/s
Epoch 13, Iter 39, Loss: 10993.3457031, Throughput: 96.862358 im/s
Epoch 13, Train Loss: 11471.2382812, Time: 51.5372s, Throughput: 96.862080 im/s
Epoch 14, Iter 39, Loss: 11553.6201172, Throughput: 97.479274 im/s
Epoch 14, Train Loss: 11262.7490234, Time: 51.2111s, Throughput: 97.478956 im/s
Epoch 15, Iter 39, Loss: 11249.5058594, Throughput: 97.490008 im/s
Epoch 15, Train Loss: 11139.3203125, Time: 51.2054s, Throughput: 97.489659 im/s
Epoch 16, Iter 39, Loss: 11492.2861328, Throughput: 97.852554 im/s
Epoch 16, Train Loss: 11015.9980469, Time: 51.0157s, Throughput: 97.852193 im/s
Epoch 17, Iter 39, Loss: 10950.7363281, Throughput: 97.417553 im/s
Epoch 17, Train Loss: 10978.9521484, Time: 51.2436s, Throughput: 97.417129 im/s
Epoch 18, Iter 39, Loss: 10076.7734375, Throughput: 97.440556 im/s
Epoch 18, Train Loss: 10815.6474609, Time: 51.2314s, Throughput: 97.440200 im/s
Epoch 19, Iter 39, Loss: 10552.8164062, Throughput: 97.493901 im/s
Epoch 19, Train Loss: 10718.1318359, Time: 51.2034s, Throughput: 97.493505 im/s
Epoch 20, Iter 39, Loss: 10238.5185547, Throughput: 97.343057 im/s
Epoch 20, Train Loss: 10735.5029297, Time: 51.2827s, Throughput: 97.342717 im/s
Epoch 21, Iter 39, Loss: 10884.6474609, Throughput: 97.272896 im/s
Epoch 21, Train Loss: 10521.4482422, Time: 51.3197s, Throughput: 97.272584 im/s
Epoch 22, Iter 39, Loss: 10599.5068359, Throughput: 97.769862 im/s
Epoch 22, Train Loss: 10518.0458984, Time: 51.0588s, Throughput: 97.769537 im/s
Epoch 23, Iter 39, Loss: 10656.9990234, Throughput: 97.347275 im/s
Epoch 23, Train Loss: 10406.9267578, Time: 51.2805s, Throughput: 97.346931 im/s
Epoch 24, Iter 39, Loss: 10131.4580078, Throughput: 97.579340 im/s
Epoch 24, Train Loss: 10367.1093750, Time: 51.1586s, Throughput: 97.578835 im/s
Epoch 25, Iter 39, Loss: 10050.0585938, Throughput: 98.520255 im/s
Epoch 25, Train Loss: 10350.4794922, Time: 50.6700s, Throughput: 98.519887 im/s
Epoch 26, Iter 39, Loss: 10428.4794922, Throughput: 97.017234 im/s
Epoch 26, Train Loss: 10251.8242188, Time: 51.4549s, Throughput: 97.016932 im/s
Epoch 27, Iter 39, Loss: 10445.2167969, Throughput: 97.510743 im/s
Epoch 27, Train Loss: 10188.1582031, Time: 51.1945s, Throughput: 97.510408 im/s
Epoch 28, Iter 39, Loss: 10112.3828125, Throughput: 97.242577 im/s
Epoch 28, Train Loss: 10107.5488281, Time: 51.3357s, Throughput: 97.242247 im/s
Epoch 29, Iter 39, Loss: 10515.3750000, Throughput: 97.139554 im/s
Epoch 29, Train Loss: 10010.3896484, Time: 51.3902s, Throughput: 97.139213 im/s
Epoch 30, Iter 39, Loss: 9661.4150391, Throughput: 97.441786 im/s
Epoch 30, Train Loss: 10074.1914062, Time: 51.2308s, Throughput: 97.441478 im/s
Epoch 31, Iter 39, Loss: 10864.3613281, Throughput: 97.047882 im/s
Epoch 31, Train Loss: 9943.2177734, Time: 51.4387s, Throughput: 97.047595 im/s
Epoch 32, Iter 39, Loss: 9613.5302734, Throughput: 97.806731 im/s
Epoch 32, Train Loss: 9947.3525391, Time: 51.0396s, Throughput: 97.806439 im/s
Epoch 33, Iter 39, Loss: 9873.9746094, Throughput: 97.822683 im/s
Epoch 33, Train Loss: 9918.2666016, Time: 51.0313s, Throughput: 97.822327 im/s
Epoch 34, Iter 39, Loss: 9605.9628906, Throughput: 97.875861 im/s
Epoch 34, Train Loss: 9828.7832031, Time: 51.0035s, Throughput: 97.875555 im/s
Epoch 35, Iter 39, Loss: 9699.5048828, Throughput: 97.769962 im/s
Epoch 35, Train Loss: 9780.4794922, Time: 51.0588s, Throughput: 97.769609 im/s
Epoch 36, Iter 39, Loss: 9623.7333984, Throughput: 98.409244 im/s
Epoch 36, Train Loss: 9763.0644531, Time: 50.7271s, Throughput: 98.408930 im/s
Epoch 37, Iter 39, Loss: 9843.2685547, Throughput: 97.912097 im/s
Epoch 37, Train Loss: 9711.3017578, Time: 50.9847s, Throughput: 97.911792 im/s
Epoch 38, Iter 39, Loss: 9818.5234375, Throughput: 97.785946 im/s
Epoch 38, Train Loss: 9763.1308594, Time: 51.0504s, Throughput: 97.785632 im/s
Epoch 39, Iter 39, Loss: 9785.1210938, Throughput: 97.746721 im/s
Epoch 39, Train Loss: 9693.8583984, Time: 51.0709s, Throughput: 97.746413 im/s
Epoch 40, Iter 39, Loss: 10051.3203125, Throughput: 97.899992 im/s
Epoch 40, Train Loss: 9664.6230469, Time: 50.9910s, Throughput: 97.899658 im/s
Epoch 41, Iter 39, Loss: 9529.7324219, Throughput: 97.859523 im/s
Epoch 41, Train Loss: 9620.3427734, Time: 51.0121s, Throughput: 97.859203 im/s
Epoch 42, Iter 39, Loss: 9556.8203125, Throughput: 97.596542 im/s
Epoch 42, Train Loss: 9576.9111328, Time: 51.1495s, Throughput: 97.596239 im/s
Epoch 43, Iter 39, Loss: 9262.0781250, Throughput: 98.047927 im/s
Epoch 43, Train Loss: 9563.6523438, Time: 50.9140s, Throughput: 98.047612 im/s
Epoch 44, Iter 39, Loss: 9231.0488281, Throughput: 97.665387 im/s
Epoch 44, Train Loss: 9486.5332031, Time: 51.1135s, Throughput: 97.665064 im/s
Epoch 45, Iter 39, Loss: 9457.3574219, Throughput: 98.043350 im/s
Epoch 45, Train Loss: 9456.7080078, Time: 50.9164s, Throughput: 98.043061 im/s
Epoch 46, Iter 39, Loss: 9730.5449219, Throughput: 98.054718 im/s
Epoch 46, Train Loss: 9435.7587891, Time: 50.9105s, Throughput: 98.054417 im/s
Epoch 47, Iter 39, Loss: 9720.1552734, Throughput: 97.582486 im/s
Epoch 47, Train Loss: 9468.2148438, Time: 51.1569s, Throughput: 97.582121 im/s
Epoch 48, Iter 39, Loss: 9262.5234375, Throughput: 96.228669 im/s
Epoch 48, Train Loss: 9385.5683594, Time: 51.8766s, Throughput: 96.228346 im/s
Epoch 49, Iter 39, Loss: 9804.2021484, Throughput: 97.771289 im/s
Epoch 49, Train Loss: 9353.4667969, Time: 51.0581s, Throughput: 97.770940 im/s
Epoch 50, Iter 39, Loss: 9062.9101562, Throughput: 97.853845 im/s
Epoch 50, Train Loss: 9331.3662109, Time: 51.0150s, Throughput: 97.853486 im/s

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
  CPU: 4 × Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, icelake-server)
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
