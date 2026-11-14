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
Epoch 1, Iter 39, Loss: 24592.1992188, Throughput: 14.414585 im/s
Epoch 1, Train Loss: 39817.2656250, Time: 346.5804s, Throughput: 14.403584 im/s
Epoch 2, Iter 39, Loss: 18541.5683594, Throughput: 170.738762 im/s
Epoch 2, Train Loss: 20314.7324219, Time: 29.2378s, Throughput: 170.737594 im/s
Epoch 3, Iter 39, Loss: 15428.5781250, Throughput: 171.104712 im/s
Epoch 3, Train Loss: 16747.1992188, Time: 29.1753s, Throughput: 171.103568 im/s
Epoch 4, Iter 39, Loss: 14573.0507812, Throughput: 171.029794 im/s
Epoch 4, Train Loss: 15257.4423828, Time: 29.1881s, Throughput: 171.028856 im/s
Epoch 5, Iter 39, Loss: 14297.5527344, Throughput: 170.965590 im/s
Epoch 5, Train Loss: 14313.2080078, Time: 29.1990s, Throughput: 170.964472 im/s
Epoch 6, Iter 39, Loss: 13550.8671875, Throughput: 171.083579 im/s
Epoch 6, Train Loss: 13636.4619141, Time: 29.1789s, Throughput: 171.082406 im/s
Epoch 7, Iter 39, Loss: 13493.5312500, Throughput: 171.291556 im/s
Epoch 7, Train Loss: 13173.4111328, Time: 29.1434s, Throughput: 171.290668 im/s
Epoch 8, Iter 39, Loss: 12561.4687500, Throughput: 170.773450 im/s
Epoch 8, Train Loss: 12782.4785156, Time: 29.2319s, Throughput: 170.772428 im/s
Epoch 9, Iter 39, Loss: 13101.1347656, Throughput: 170.951006 im/s
Epoch 9, Train Loss: 12529.7343750, Time: 29.2015s, Throughput: 170.949936 im/s
Epoch 10, Iter 39, Loss: 12194.2558594, Throughput: 171.359686 im/s
Epoch 10, Train Loss: 12152.7001953, Time: 29.1319s, Throughput: 171.358622 im/s
Epoch 11, Iter 39, Loss: 11512.8945312, Throughput: 171.436696 im/s
Epoch 11, Train Loss: 11874.0468750, Time: 29.1188s, Throughput: 171.435454 im/s
Epoch 12, Iter 39, Loss: 11125.9648438, Throughput: 170.869208 im/s
Epoch 12, Train Loss: 11700.6357422, Time: 29.2155s, Throughput: 170.868091 im/s
Epoch 13, Iter 39, Loss: 11416.1816406, Throughput: 171.352170 im/s
Epoch 13, Train Loss: 11560.3105469, Time: 29.1332s, Throughput: 171.351116 im/s
Epoch 14, Iter 39, Loss: 11215.4003906, Throughput: 170.921237 im/s
Epoch 14, Train Loss: 11303.2978516, Time: 29.2066s, Throughput: 170.920195 im/s
Epoch 15, Iter 39, Loss: 11507.1923828, Throughput: 171.052320 im/s
Epoch 15, Train Loss: 11274.3818359, Time: 29.1842s, Throughput: 171.051470 im/s
Epoch 16, Iter 39, Loss: 11714.2167969, Throughput: 171.021110 im/s
Epoch 16, Train Loss: 11113.1787109, Time: 29.1896s, Throughput: 171.020003 im/s
Epoch 17, Iter 39, Loss: 10118.6494141, Throughput: 171.352299 im/s
Epoch 17, Train Loss: 11025.6757812, Time: 29.1332s, Throughput: 171.351187 im/s
Epoch 18, Iter 39, Loss: 10850.8417969, Throughput: 169.598273 im/s
Epoch 18, Train Loss: 10896.6669922, Time: 29.4345s, Throughput: 169.596833 im/s
Epoch 19, Iter 39, Loss: 11067.6982422, Throughput: 170.723837 im/s
Epoch 19, Train Loss: 10754.6552734, Time: 29.2404s, Throughput: 170.722938 im/s
Epoch 20, Iter 39, Loss: 9977.6503906, Throughput: 171.323048 im/s
Epoch 20, Train Loss: 10675.5351562, Time: 29.1381s, Throughput: 171.322136 im/s
Epoch 21, Iter 39, Loss: 11493.0175781, Throughput: 171.164396 im/s
Epoch 21, Train Loss: 10708.8544922, Time: 29.1651s, Throughput: 171.163315 im/s
Epoch 22, Iter 39, Loss: 10873.3134766, Throughput: 171.452164 im/s
Epoch 22, Train Loss: 10531.5908203, Time: 29.1162s, Throughput: 171.451109 im/s
Epoch 23, Iter 39, Loss: 10459.4238281, Throughput: 171.044889 im/s
Epoch 23, Train Loss: 10403.5517578, Time: 29.1855s, Throughput: 171.043934 im/s
Epoch 24, Iter 39, Loss: 10380.3222656, Throughput: 171.176403 im/s
Epoch 24, Train Loss: 10334.9492188, Time: 29.1631s, Throughput: 171.175242 im/s
Epoch 25, Iter 39, Loss: 10176.0175781, Throughput: 171.095311 im/s
Epoch 25, Train Loss: 10267.0908203, Time: 29.1769s, Throughput: 171.094354 im/s
Epoch 26, Iter 39, Loss: 10792.8613281, Throughput: 170.912629 im/s
Epoch 26, Train Loss: 10205.4824219, Time: 29.2081s, Throughput: 170.911763 im/s
Epoch 27, Iter 39, Loss: 9396.5625000, Throughput: 171.554339 im/s
Epoch 27, Train Loss: 10162.9130859, Time: 29.0989s, Throughput: 171.553142 im/s
Epoch 28, Iter 39, Loss: 10027.0781250, Throughput: 171.239649 im/s
Epoch 28, Train Loss: 10148.5253906, Time: 29.1523s, Throughput: 171.238380 im/s
Epoch 29, Iter 39, Loss: 9881.5136719, Throughput: 171.330115 im/s
Epoch 29, Train Loss: 10134.3564453, Time: 29.1369s, Throughput: 171.329027 im/s
Epoch 30, Iter 39, Loss: 10794.2070312, Throughput: 171.341677 im/s
Epoch 30, Train Loss: 9958.0039062, Time: 29.1350s, Throughput: 171.340570 im/s
Epoch 31, Iter 39, Loss: 10236.7832031, Throughput: 171.569946 im/s
Epoch 31, Train Loss: 9935.5019531, Time: 29.0962s, Throughput: 171.568944 im/s
Epoch 32, Iter 39, Loss: 10255.5039062, Throughput: 171.041913 im/s
Epoch 32, Train Loss: 9985.5302734, Time: 29.1860s, Throughput: 171.040928 im/s
Epoch 33, Iter 39, Loss: 9802.0244141, Throughput: 170.621185 im/s
Epoch 33, Train Loss: 9886.3613281, Time: 29.2580s, Throughput: 170.620194 im/s
Epoch 34, Iter 39, Loss: 9948.9023438, Throughput: 171.395422 im/s
Epoch 34, Train Loss: 9821.0556641, Time: 29.1258s, Throughput: 171.394346 im/s
Epoch 35, Iter 39, Loss: 9965.7392578, Throughput: 169.156817 im/s
Epoch 35, Train Loss: 9813.6044922, Time: 29.5113s, Throughput: 169.155694 im/s
Epoch 36, Iter 39, Loss: 10117.7578125, Throughput: 171.152658 im/s
Epoch 36, Train Loss: 9700.7392578, Time: 29.1671s, Throughput: 171.151732 im/s
Epoch 37, Iter 39, Loss: 9787.2744141, Throughput: 171.478749 im/s
Epoch 37, Train Loss: 9644.2343750, Time: 29.1117s, Throughput: 171.477683 im/s
Epoch 38, Iter 39, Loss: 9508.6035156, Throughput: 171.613617 im/s
Epoch 38, Train Loss: 9604.0068359, Time: 29.0888s, Throughput: 171.612614 im/s
Epoch 39, Iter 39, Loss: 9424.9013672, Throughput: 171.250898 im/s
Epoch 39, Train Loss: 9579.0556641, Time: 29.1504s, Throughput: 171.249817 im/s
Epoch 40, Iter 39, Loss: 8905.2128906, Throughput: 171.309902 im/s
Epoch 40, Train Loss: 9526.0937500, Time: 29.1404s, Throughput: 171.308849 im/s
Epoch 41, Iter 39, Loss: 9288.0937500, Throughput: 170.841288 im/s
Epoch 41, Train Loss: 9465.0996094, Time: 29.2203s, Throughput: 170.840353 im/s
Epoch 42, Iter 39, Loss: 9740.9179688, Throughput: 171.301230 im/s
Epoch 42, Train Loss: 9478.6630859, Time: 29.1418s, Throughput: 171.300290 im/s
Epoch 43, Iter 39, Loss: 9685.1582031, Throughput: 171.473706 im/s
Epoch 43, Train Loss: 9456.9218750, Time: 29.1125s, Throughput: 171.472535 im/s
Epoch 44, Iter 39, Loss: 9307.7792969, Throughput: 171.651659 im/s
Epoch 44, Train Loss: 9416.4267578, Time: 29.0823s, Throughput: 171.650627 im/s
Epoch 45, Iter 39, Loss: 9073.8134766, Throughput: 171.430909 im/s
Epoch 45, Train Loss: 9393.8632812, Time: 29.1198s, Throughput: 171.429890 im/s
Epoch 46, Iter 39, Loss: 9179.7900391, Throughput: 171.441706 im/s
Epoch 46, Train Loss: 9441.5097656, Time: 29.1180s, Throughput: 171.440582 im/s
Epoch 47, Iter 39, Loss: 9648.1718750, Throughput: 170.958383 im/s
Epoch 47, Train Loss: 9354.5000000, Time: 29.2003s, Throughput: 170.957340 im/s
Epoch 48, Iter 39, Loss: 9383.0439453, Throughput: 170.896129 im/s
Epoch 48, Train Loss: 9318.0869141, Time: 29.2109s, Throughput: 170.894853 im/s
Epoch 49, Iter 39, Loss: 9017.7548828, Throughput: 170.511323 im/s
Epoch 49, Train Loss: 9314.5136719, Time: 29.2768s, Throughput: 170.510246 im/s
Epoch 50, Iter 39, Loss: 10277.6113281, Throughput: 171.469277 im/s
Epoch 50, Train Loss: 9276.8964844, Time: 29.1133s, Throughput: 171.468229 im/s

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
  CPU: 4 × AMD EPYC 7763 64-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver3)
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
