---
url: /previews/PR1824/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23995.0429688, Throughput: 23.557456 im/s
Epoch 1, Train Loss: 39658.5039062, Time: 212.0980s, Throughput: 23.536290 im/s
Epoch 2, Iter 39, Loss: 18019.8828125, Throughput: 125.523719 im/s
Epoch 2, Train Loss: 20169.2812500, Time: 39.7696s, Throughput: 125.523104 im/s
Epoch 3, Iter 39, Loss: 15834.6367188, Throughput: 124.807529 im/s
Epoch 3, Train Loss: 16557.9980469, Time: 39.9977s, Throughput: 124.807070 im/s
Epoch 4, Iter 39, Loss: 14473.9501953, Throughput: 124.805672 im/s
Epoch 4, Train Loss: 15083.1455078, Time: 39.9984s, Throughput: 124.805108 im/s
Epoch 5, Iter 39, Loss: 13113.3769531, Throughput: 124.428370 im/s
Epoch 5, Train Loss: 14117.4267578, Time: 40.1196s, Throughput: 124.427849 im/s
Epoch 6, Iter 39, Loss: 12825.8095703, Throughput: 123.400851 im/s
Epoch 6, Train Loss: 13413.4687500, Time: 40.4537s, Throughput: 123.400393 im/s
Epoch 7, Iter 39, Loss: 13730.2480469, Throughput: 123.877192 im/s
Epoch 7, Train Loss: 13014.5332031, Time: 40.2981s, Throughput: 123.876695 im/s
Epoch 8, Iter 39, Loss: 12691.7978516, Throughput: 122.037650 im/s
Epoch 8, Train Loss: 12639.2685547, Time: 40.9056s, Throughput: 122.037110 im/s
Epoch 9, Iter 39, Loss: 13321.4550781, Throughput: 121.788172 im/s
Epoch 9, Train Loss: 12273.6494141, Time: 40.9894s, Throughput: 121.787664 im/s
Epoch 10, Iter 39, Loss: 11420.4833984, Throughput: 123.353544 im/s
Epoch 10, Train Loss: 11984.7099609, Time: 40.4692s, Throughput: 123.353032 im/s
Epoch 11, Iter 39, Loss: 11494.8730469, Throughput: 122.311140 im/s
Epoch 11, Train Loss: 11817.3183594, Time: 40.8141s, Throughput: 122.310681 im/s
Epoch 12, Iter 39, Loss: 11397.9023438, Throughput: 124.299588 im/s
Epoch 12, Train Loss: 11507.8242188, Time: 40.1612s, Throughput: 124.299106 im/s
Epoch 13, Iter 39, Loss: 11780.2734375, Throughput: 124.998530 im/s
Epoch 13, Train Loss: 11359.8134766, Time: 39.9366s, Throughput: 124.998044 im/s
Epoch 14, Iter 39, Loss: 10986.4794922, Throughput: 126.263242 im/s
Epoch 14, Train Loss: 11323.6591797, Time: 39.5366s, Throughput: 126.262684 im/s
Epoch 15, Iter 39, Loss: 11081.6210938, Throughput: 124.253994 im/s
Epoch 15, Train Loss: 11207.1894531, Time: 40.1759s, Throughput: 124.253481 im/s
Epoch 16, Iter 39, Loss: 11509.4023438, Throughput: 127.402847 im/s
Epoch 16, Train Loss: 11094.4306641, Time: 39.1830s, Throughput: 127.402339 im/s
Epoch 17, Iter 39, Loss: 11178.9863281, Throughput: 126.417003 im/s
Epoch 17, Train Loss: 10981.8828125, Time: 39.4885s, Throughput: 126.416500 im/s
Epoch 18, Iter 39, Loss: 10974.3515625, Throughput: 127.566170 im/s
Epoch 18, Train Loss: 10902.0185547, Time: 39.1328s, Throughput: 127.565613 im/s
Epoch 19, Iter 39, Loss: 10429.1093750, Throughput: 124.421439 im/s
Epoch 19, Train Loss: 10759.5517578, Time: 40.1219s, Throughput: 124.420902 im/s
Epoch 20, Iter 39, Loss: 10865.2587891, Throughput: 123.978014 im/s
Epoch 20, Train Loss: 10620.3183594, Time: 40.2654s, Throughput: 123.977515 im/s
Epoch 21, Iter 39, Loss: 10795.3710938, Throughput: 125.944139 im/s
Epoch 21, Train Loss: 10542.7392578, Time: 39.6368s, Throughput: 125.943653 im/s
Epoch 22, Iter 39, Loss: 10191.2099609, Throughput: 127.018363 im/s
Epoch 22, Train Loss: 10528.1367188, Time: 39.3016s, Throughput: 127.017801 im/s
Epoch 23, Iter 39, Loss: 10048.9482422, Throughput: 129.363752 im/s
Epoch 23, Train Loss: 10455.9082031, Time: 38.5890s, Throughput: 129.363266 im/s
Epoch 24, Iter 39, Loss: 10212.3300781, Throughput: 128.518380 im/s
Epoch 24, Train Loss: 10450.4677734, Time: 38.8429s, Throughput: 128.517778 im/s
Epoch 25, Iter 39, Loss: 10261.8115234, Throughput: 129.739888 im/s
Epoch 25, Train Loss: 10323.9003906, Time: 38.4772s, Throughput: 129.739305 im/s
Epoch 26, Iter 39, Loss: 10030.1894531, Throughput: 129.890088 im/s
Epoch 26, Train Loss: 10277.2080078, Time: 38.4327s, Throughput: 129.889550 im/s
Epoch 27, Iter 39, Loss: 9669.1162109, Throughput: 125.233414 im/s
Epoch 27, Train Loss: 10155.6308594, Time: 39.8617s, Throughput: 125.232899 im/s
Epoch 28, Iter 39, Loss: 10344.8496094, Throughput: 128.551353 im/s
Epoch 28, Train Loss: 10121.6689453, Time: 38.8329s, Throughput: 128.550834 im/s
Epoch 29, Iter 39, Loss: 9855.9433594, Throughput: 128.888864 im/s
Epoch 29, Train Loss: 10135.5830078, Time: 38.7312s, Throughput: 128.888361 im/s
Epoch 30, Iter 39, Loss: 10518.2138672, Throughput: 127.479462 im/s
Epoch 30, Train Loss: 9968.6787109, Time: 39.1594s, Throughput: 127.478889 im/s
Epoch 31, Iter 39, Loss: 10141.6494141, Throughput: 126.690163 im/s
Epoch 31, Train Loss: 9954.0888672, Time: 39.4034s, Throughput: 126.689655 im/s
Epoch 32, Iter 39, Loss: 9972.2011719, Throughput: 123.224905 im/s
Epoch 32, Train Loss: 9976.6318359, Time: 40.5115s, Throughput: 123.224370 im/s
Epoch 33, Iter 39, Loss: 10091.0839844, Throughput: 123.107600 im/s
Epoch 33, Train Loss: 9909.0898438, Time: 40.5501s, Throughput: 123.107099 im/s
Epoch 34, Iter 39, Loss: 9795.4941406, Throughput: 123.347975 im/s
Epoch 34, Train Loss: 9863.2832031, Time: 40.4710s, Throughput: 123.347478 im/s
Epoch 35, Iter 39, Loss: 9750.3144531, Throughput: 123.805010 im/s
Epoch 35, Train Loss: 9935.7421875, Time: 40.3216s, Throughput: 123.804470 im/s
Epoch 36, Iter 39, Loss: 9163.1289062, Throughput: 123.320651 im/s
Epoch 36, Train Loss: 9859.4794922, Time: 40.4800s, Throughput: 123.320179 im/s
Epoch 37, Iter 39, Loss: 9797.6250000, Throughput: 125.413867 im/s
Epoch 37, Train Loss: 9727.5107422, Time: 39.8044s, Throughput: 125.413390 im/s
Epoch 38, Iter 39, Loss: 9823.0429688, Throughput: 123.674509 im/s
Epoch 38, Train Loss: 9722.0605469, Time: 40.3642s, Throughput: 123.674022 im/s
Epoch 39, Iter 39, Loss: 9808.4560547, Throughput: 124.118160 im/s
Epoch 39, Train Loss: 9684.7900391, Time: 40.2199s, Throughput: 124.117685 im/s
Epoch 40, Iter 39, Loss: 10270.2275391, Throughput: 123.316902 im/s
Epoch 40, Train Loss: 9648.7353516, Time: 40.4812s, Throughput: 123.316390 im/s
Epoch 41, Iter 39, Loss: 9470.2265625, Throughput: 127.305305 im/s
Epoch 41, Train Loss: 9621.2500000, Time: 39.2130s, Throughput: 127.304779 im/s
Epoch 42, Iter 39, Loss: 9726.6250000, Throughput: 126.085840 im/s
Epoch 42, Train Loss: 9646.7138672, Time: 39.5922s, Throughput: 126.085362 im/s
Epoch 43, Iter 39, Loss: 9292.9257812, Throughput: 126.465295 im/s
Epoch 43, Train Loss: 9534.2695312, Time: 39.4734s, Throughput: 126.464827 im/s
Epoch 44, Iter 39, Loss: 9417.7656250, Throughput: 126.739170 im/s
Epoch 44, Train Loss: 9479.4746094, Time: 39.3881s, Throughput: 126.738677 im/s
Epoch 45, Iter 39, Loss: 9641.8925781, Throughput: 125.647714 im/s
Epoch 45, Train Loss: 9493.1621094, Time: 39.7303s, Throughput: 125.647243 im/s
Epoch 46, Iter 39, Loss: 9206.0742188, Throughput: 128.129967 im/s
Epoch 46, Train Loss: 9518.7216797, Time: 38.9606s, Throughput: 128.129438 im/s
Epoch 47, Iter 39, Loss: 9855.1757812, Throughput: 126.749972 im/s
Epoch 47, Train Loss: 9440.5468750, Time: 39.3848s, Throughput: 126.749500 im/s
Epoch 48, Iter 39, Loss: 9942.7734375, Throughput: 125.683056 im/s
Epoch 48, Train Loss: 9446.0205078, Time: 39.7192s, Throughput: 125.682385 im/s
Epoch 49, Iter 39, Loss: 9423.3818359, Throughput: 125.298161 im/s
Epoch 49, Train Loss: 9421.3935547, Time: 39.8411s, Throughput: 125.297701 im/s
Epoch 50, Iter 39, Loss: 8910.9072266, Throughput: 126.879127 im/s
Epoch 50, Train Loss: 9352.9628906, Time: 39.3447s, Throughput: 126.878634 im/s

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
  CPU: 4 × AMD EPYC 9V45 96-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver5)
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
