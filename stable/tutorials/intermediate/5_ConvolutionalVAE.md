---
url: /stable/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 25126.8515625, Throughput: 24.563215 im/s
Epoch 1, Train Loss: 40408.1171875, Time: 203.4503s, Throughput: 24.536704 im/s
Epoch 2, Iter 39, Loss: 18410.1425781, Throughput: 98.644792 im/s
Epoch 2, Train Loss: 20985.2246094, Time: 50.6060s, Throughput: 98.644424 im/s
Epoch 3, Iter 39, Loss: 16996.2988281, Throughput: 99.573044 im/s
Epoch 3, Train Loss: 17376.1289062, Time: 50.1342s, Throughput: 99.572667 im/s
Epoch 4, Iter 39, Loss: 15543.1582031, Throughput: 100.689856 im/s
Epoch 4, Train Loss: 15686.8417969, Time: 49.5781s, Throughput: 100.689548 im/s
Epoch 5, Iter 39, Loss: 14895.4990234, Throughput: 100.904871 im/s
Epoch 5, Train Loss: 14689.5546875, Time: 49.4725s, Throughput: 100.904576 im/s
Epoch 6, Iter 39, Loss: 13964.3681641, Throughput: 101.146122 im/s
Epoch 6, Train Loss: 13969.4042969, Time: 49.3555s, Throughput: 101.143759 im/s
Epoch 7, Iter 39, Loss: 13879.1591797, Throughput: 100.245545 im/s
Epoch 7, Train Loss: 13436.8349609, Time: 49.7979s, Throughput: 100.245179 im/s
Epoch 8, Iter 39, Loss: 12240.1181641, Throughput: 100.378254 im/s
Epoch 8, Train Loss: 12946.2060547, Time: 49.7321s, Throughput: 100.377895 im/s
Epoch 9, Iter 39, Loss: 12493.3798828, Throughput: 101.370549 im/s
Epoch 9, Train Loss: 12581.1298828, Time: 49.2453s, Throughput: 101.370084 im/s
Epoch 10, Iter 39, Loss: 12524.0625000, Throughput: 101.538384 im/s
Epoch 10, Train Loss: 12424.6621094, Time: 49.1639s, Throughput: 101.537998 im/s
Epoch 11, Iter 39, Loss: 12182.1015625, Throughput: 101.771285 im/s
Epoch 11, Train Loss: 12342.9082031, Time: 49.0518s, Throughput: 101.769978 im/s
Epoch 12, Iter 39, Loss: 11349.1044922, Throughput: 99.291707 im/s
Epoch 12, Train Loss: 12027.0927734, Time: 50.2767s, Throughput: 99.290443 im/s
Epoch 13, Iter 39, Loss: 11940.9970703, Throughput: 101.555394 im/s
Epoch 13, Train Loss: 11813.3837891, Time: 49.1556s, Throughput: 101.555031 im/s
Epoch 14, Iter 39, Loss: 11658.9609375, Throughput: 101.944383 im/s
Epoch 14, Train Loss: 11610.3193359, Time: 48.9686s, Throughput: 101.942909 im/s
Epoch 15, Iter 39, Loss: 11885.9306641, Throughput: 101.919107 im/s
Epoch 15, Train Loss: 11501.0839844, Time: 48.9802s, Throughput: 101.918710 im/s
Epoch 16, Iter 39, Loss: 11185.7402344, Throughput: 101.775574 im/s
Epoch 16, Train Loss: 11367.8457031, Time: 49.0493s, Throughput: 101.775240 im/s
Epoch 17, Iter 39, Loss: 11425.4062500, Throughput: 102.401517 im/s
Epoch 17, Train Loss: 11392.5351562, Time: 48.7494s, Throughput: 102.401155 im/s
Epoch 18, Iter 39, Loss: 11323.3369141, Throughput: 101.699449 im/s
Epoch 18, Train Loss: 11179.8789062, Time: 49.0860s, Throughput: 101.699082 im/s
Epoch 19, Iter 39, Loss: 11186.3066406, Throughput: 102.060685 im/s
Epoch 19, Train Loss: 11070.7529297, Time: 48.9124s, Throughput: 102.060080 im/s
Epoch 20, Iter 39, Loss: 10320.9052734, Throughput: 101.760806 im/s
Epoch 20, Train Loss: 11002.4892578, Time: 49.0564s, Throughput: 101.760499 im/s
Epoch 21, Iter 39, Loss: 10880.0585938, Throughput: 99.930423 im/s
Epoch 21, Train Loss: 10944.6044922, Time: 49.9549s, Throughput: 99.930049 im/s
Epoch 22, Iter 39, Loss: 11079.4531250, Throughput: 100.550733 im/s
Epoch 22, Train Loss: 10855.0751953, Time: 49.6468s, Throughput: 100.550364 im/s
Epoch 23, Iter 39, Loss: 11046.0166016, Throughput: 101.675772 im/s
Epoch 23, Train Loss: 10737.0927734, Time: 49.0974s, Throughput: 101.675416 im/s
Epoch 24, Iter 39, Loss: 10182.8554688, Throughput: 101.472677 im/s
Epoch 24, Train Loss: 10618.1835938, Time: 49.1957s, Throughput: 101.472327 im/s
Epoch 25, Iter 39, Loss: 11206.7832031, Throughput: 101.928782 im/s
Epoch 25, Train Loss: 10615.1845703, Time: 48.9755s, Throughput: 101.928467 im/s
Epoch 26, Iter 39, Loss: 10029.5185547, Throughput: 99.242956 im/s
Epoch 26, Train Loss: 10601.6816406, Time: 50.3012s, Throughput: 99.242256 im/s
Epoch 27, Iter 39, Loss: 11215.1367188, Throughput: 100.962481 im/s
Epoch 27, Train Loss: 10521.5292969, Time: 49.4443s, Throughput: 100.962072 im/s
Epoch 28, Iter 39, Loss: 10720.3339844, Throughput: 101.613300 im/s
Epoch 28, Train Loss: 10500.5830078, Time: 49.1276s, Throughput: 101.612961 im/s
Epoch 29, Iter 39, Loss: 10805.6074219, Throughput: 101.518652 im/s
Epoch 29, Train Loss: 10448.6240234, Time: 49.1734s, Throughput: 101.518257 im/s
Epoch 30, Iter 39, Loss: 11137.7050781, Throughput: 101.958742 im/s
Epoch 30, Train Loss: 10350.8281250, Time: 48.9611s, Throughput: 101.958410 im/s
Epoch 31, Iter 39, Loss: 9888.8154297, Throughput: 102.032636 im/s
Epoch 31, Train Loss: 10282.3515625, Time: 48.9257s, Throughput: 102.032261 im/s
Epoch 32, Iter 39, Loss: 10299.5429688, Throughput: 97.198340 im/s
Epoch 32, Train Loss: 10191.9257812, Time: 51.3595s, Throughput: 97.197149 im/s
Epoch 33, Iter 39, Loss: 10530.5693359, Throughput: 97.264686 im/s
Epoch 33, Train Loss: 10206.8632812, Time: 51.3241s, Throughput: 97.264311 im/s
Epoch 34, Iter 39, Loss: 9428.6962891, Throughput: 92.617571 im/s
Epoch 34, Train Loss: 10064.5146484, Time: 53.8993s, Throughput: 92.617224 im/s
Epoch 35, Iter 39, Loss: 9990.2548828, Throughput: 92.037789 im/s
Epoch 35, Train Loss: 10068.4208984, Time: 54.2389s, Throughput: 92.037339 im/s
Epoch 36, Iter 39, Loss: 9801.5703125, Throughput: 95.389068 im/s
Epoch 36, Train Loss: 10053.6689453, Time: 52.3333s, Throughput: 95.388679 im/s
Epoch 37, Iter 39, Loss: 10549.6640625, Throughput: 98.475885 im/s
Epoch 37, Train Loss: 10010.5048828, Time: 50.6928s, Throughput: 98.475568 im/s
Epoch 38, Iter 39, Loss: 9820.2578125, Throughput: 93.848080 im/s
Epoch 38, Train Loss: 9954.5781250, Time: 53.1925s, Throughput: 93.847765 im/s
Epoch 39, Iter 39, Loss: 9874.8828125, Throughput: 92.402138 im/s
Epoch 39, Train Loss: 9933.5927734, Time: 54.0250s, Throughput: 92.401614 im/s
Epoch 40, Iter 39, Loss: 9477.9462891, Throughput: 91.360976 im/s
Epoch 40, Train Loss: 9944.6103516, Time: 54.6406s, Throughput: 91.360665 im/s
Epoch 41, Iter 39, Loss: 10117.4980469, Throughput: 91.578855 im/s
Epoch 41, Train Loss: 9862.0058594, Time: 54.5106s, Throughput: 91.578565 im/s
Epoch 42, Iter 39, Loss: 9888.1914062, Throughput: 90.522025 im/s
Epoch 42, Train Loss: 9853.9941406, Time: 55.1470s, Throughput: 90.521732 im/s
Epoch 43, Iter 39, Loss: 9802.2001953, Throughput: 90.903991 im/s
Epoch 43, Train Loss: 9797.3857422, Time: 54.9153s, Throughput: 90.903615 im/s
Epoch 44, Iter 39, Loss: 10311.9765625, Throughput: 97.444466 im/s
Epoch 44, Train Loss: 9774.1943359, Time: 51.2324s, Throughput: 97.438395 im/s
Epoch 45, Iter 39, Loss: 9800.0546875, Throughput: 100.688368 im/s
Epoch 45, Train Loss: 9795.1699219, Time: 49.5807s, Throughput: 100.684284 im/s
Epoch 46, Iter 39, Loss: 9513.8798828, Throughput: 101.291403 im/s
Epoch 46, Train Loss: 9843.2880859, Time: 49.2837s, Throughput: 101.291049 im/s
Epoch 47, Iter 39, Loss: 10010.8671875, Throughput: 101.558965 im/s
Epoch 47, Train Loss: 9695.9130859, Time: 49.1540s, Throughput: 101.558382 im/s
Epoch 48, Iter 39, Loss: 10098.6640625, Throughput: 99.873294 im/s
Epoch 48, Train Loss: 9763.0957031, Time: 49.9835s, Throughput: 99.872908 im/s
Epoch 49, Iter 39, Loss: 8907.5429688, Throughput: 100.049548 im/s
Epoch 49, Train Loss: 9663.6953125, Time: 49.8956s, Throughput: 100.048830 im/s
Epoch 50, Iter 39, Loss: 9177.3466797, Throughput: 100.521603 im/s
Epoch 50, Train Loss: 9611.9658203, Time: 49.6612s, Throughput: 100.521137 im/s

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
Julia Version 1.13.4
Commit c1ac32ab733 (2026-11-13 17:30 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × AMD EPYC 9V45 96-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-20.1.8 (ORCJIT, znver5)
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
