---
url: /previews/PR1829/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23455.9765625, Throughput: 21.525024 im/s
Epoch 1, Train Loss: 39788.3789062, Time: 232.1182s, Throughput: 21.506283 im/s
Epoch 2, Iter 39, Loss: 17263.2363281, Throughput: 205.820616 im/s
Epoch 2, Train Loss: 20247.5292969, Time: 24.2543s, Throughput: 205.819274 im/s
Epoch 3, Iter 39, Loss: 15962.8886719, Throughput: 206.249159 im/s
Epoch 3, Train Loss: 16558.0312500, Time: 24.2039s, Throughput: 206.247736 im/s
Epoch 4, Iter 39, Loss: 14212.5000000, Throughput: 205.752913 im/s
Epoch 4, Train Loss: 15106.2929688, Time: 24.2623s, Throughput: 205.751684 im/s
Epoch 5, Iter 39, Loss: 13787.4550781, Throughput: 206.151669 im/s
Epoch 5, Train Loss: 14275.9746094, Time: 24.2153s, Throughput: 206.150376 im/s
Epoch 6, Iter 39, Loss: 13496.8505859, Throughput: 205.216832 im/s
Epoch 6, Train Loss: 13455.8300781, Time: 24.3256s, Throughput: 205.215609 im/s
Epoch 7, Iter 39, Loss: 12412.1943359, Throughput: 206.075846 im/s
Epoch 7, Train Loss: 12994.4589844, Time: 24.2243s, Throughput: 206.074314 im/s
Epoch 8, Iter 39, Loss: 13085.2187500, Throughput: 205.891218 im/s
Epoch 8, Train Loss: 12620.1083984, Time: 24.2460s, Throughput: 205.890046 im/s
Epoch 9, Iter 39, Loss: 12101.9873047, Throughput: 205.961480 im/s
Epoch 9, Train Loss: 12336.7070312, Time: 24.2377s, Throughput: 205.960094 im/s
Epoch 10, Iter 39, Loss: 11266.4277344, Throughput: 206.100023 im/s
Epoch 10, Train Loss: 12055.5810547, Time: 24.2214s, Throughput: 206.098739 im/s
Epoch 11, Iter 39, Loss: 11057.7373047, Throughput: 206.263954 im/s
Epoch 11, Train Loss: 11831.1669922, Time: 24.2022s, Throughput: 206.262471 im/s
Epoch 12, Iter 39, Loss: 12498.8291016, Throughput: 205.935565 im/s
Epoch 12, Train Loss: 11646.4091797, Time: 24.2407s, Throughput: 205.934376 im/s
Epoch 13, Iter 39, Loss: 11787.9150391, Throughput: 205.850459 im/s
Epoch 13, Train Loss: 11524.2919922, Time: 24.2508s, Throughput: 205.849074 im/s
Epoch 14, Iter 39, Loss: 12176.1875000, Throughput: 205.594137 im/s
Epoch 14, Train Loss: 11392.5869141, Time: 24.2810s, Throughput: 205.592884 im/s
Epoch 15, Iter 39, Loss: 11162.6113281, Throughput: 206.434871 im/s
Epoch 15, Train Loss: 11328.4902344, Time: 24.1821s, Throughput: 206.433454 im/s
Epoch 16, Iter 39, Loss: 11025.4775391, Throughput: 205.267429 im/s
Epoch 16, Train Loss: 11134.3144531, Time: 24.3196s, Throughput: 205.266163 im/s
Epoch 17, Iter 39, Loss: 10624.2294922, Throughput: 206.428921 im/s
Epoch 17, Train Loss: 10997.7255859, Time: 24.1828s, Throughput: 206.427436 im/s
Epoch 18, Iter 39, Loss: 11594.1748047, Throughput: 206.192150 im/s
Epoch 18, Train Loss: 10860.8818359, Time: 24.2106s, Throughput: 206.190881 im/s
Epoch 19, Iter 39, Loss: 10790.4687500, Throughput: 205.341724 im/s
Epoch 19, Train Loss: 10832.6474609, Time: 24.3108s, Throughput: 205.340439 im/s
Epoch 20, Iter 39, Loss: 11516.3906250, Throughput: 206.155834 im/s
Epoch 20, Train Loss: 10756.9179688, Time: 24.2148s, Throughput: 206.154651 im/s
Epoch 21, Iter 39, Loss: 11108.5400391, Throughput: 206.462219 im/s
Epoch 21, Train Loss: 10648.1210938, Time: 24.1789s, Throughput: 206.460877 im/s
Epoch 22, Iter 39, Loss: 10418.7392578, Throughput: 203.553152 im/s
Epoch 22, Train Loss: 10656.8642578, Time: 24.5245s, Throughput: 203.551832 im/s
Epoch 23, Iter 39, Loss: 10652.8515625, Throughput: 206.144129 im/s
Epoch 23, Train Loss: 10483.6123047, Time: 24.2162s, Throughput: 206.142885 im/s
Epoch 24, Iter 39, Loss: 10570.5390625, Throughput: 206.294009 im/s
Epoch 24, Train Loss: 10417.2519531, Time: 24.1987s, Throughput: 206.292499 im/s
Epoch 25, Iter 39, Loss: 10562.4550781, Throughput: 206.297231 im/s
Epoch 25, Train Loss: 10375.7822266, Time: 24.1982s, Throughput: 206.295917 im/s
Epoch 26, Iter 39, Loss: 10109.0830078, Throughput: 205.334814 im/s
Epoch 26, Train Loss: 10322.6181641, Time: 24.3117s, Throughput: 205.333522 im/s
Epoch 27, Iter 39, Loss: 9949.6894531, Throughput: 205.918803 im/s
Epoch 27, Train Loss: 10287.9814453, Time: 24.2427s, Throughput: 205.917394 im/s
Epoch 28, Iter 39, Loss: 9983.3417969, Throughput: 205.873846 im/s
Epoch 28, Train Loss: 10169.8828125, Time: 24.2480s, Throughput: 205.872547 im/s
Epoch 29, Iter 39, Loss: 10591.7773438, Throughput: 206.154795 im/s
Epoch 29, Train Loss: 10113.6884766, Time: 24.2150s, Throughput: 206.153510 im/s
Epoch 30, Iter 39, Loss: 10380.9570312, Throughput: 206.320021 im/s
Epoch 30, Train Loss: 10050.9804688, Time: 24.1956s, Throughput: 206.318683 im/s
Epoch 31, Iter 39, Loss: 10861.0712891, Throughput: 206.069276 im/s
Epoch 31, Train Loss: 10055.1074219, Time: 24.2250s, Throughput: 206.068043 im/s
Epoch 32, Iter 39, Loss: 9927.2060547, Throughput: 206.662037 im/s
Epoch 32, Train Loss: 9983.9912109, Time: 24.1555s, Throughput: 206.660805 im/s
Epoch 33, Iter 39, Loss: 10369.6796875, Throughput: 205.955394 im/s
Epoch 33, Train Loss: 9931.8437500, Time: 24.2384s, Throughput: 205.954264 im/s
Epoch 34, Iter 39, Loss: 10329.8007812, Throughput: 206.408283 im/s
Epoch 34, Train Loss: 9887.7011719, Time: 24.1852s, Throughput: 206.406925 im/s
Epoch 35, Iter 39, Loss: 9951.3525391, Throughput: 205.919119 im/s
Epoch 35, Train Loss: 9830.4531250, Time: 24.2427s, Throughput: 205.917837 im/s
Epoch 36, Iter 39, Loss: 10407.7939453, Throughput: 206.249900 im/s
Epoch 36, Train Loss: 9824.3359375, Time: 24.2038s, Throughput: 206.248553 im/s
Epoch 37, Iter 39, Loss: 9814.4902344, Throughput: 205.869423 im/s
Epoch 37, Train Loss: 9844.2509766, Time: 24.2486s, Throughput: 205.867946 im/s
Epoch 38, Iter 39, Loss: 9910.0263672, Throughput: 205.026938 im/s
Epoch 38, Train Loss: 9812.1318359, Time: 24.3482s, Throughput: 205.025675 im/s
Epoch 39, Iter 39, Loss: 9882.9101562, Throughput: 205.119540 im/s
Epoch 39, Train Loss: 9836.7265625, Time: 24.3372s, Throughput: 205.118015 im/s
Epoch 40, Iter 39, Loss: 9236.8457031, Throughput: 206.475976 im/s
Epoch 40, Train Loss: 9666.7392578, Time: 24.1774s, Throughput: 206.473848 im/s
Epoch 41, Iter 39, Loss: 9499.5615234, Throughput: 206.187953 im/s
Epoch 41, Train Loss: 9603.3906250, Time: 24.2111s, Throughput: 206.186751 im/s
Epoch 42, Iter 39, Loss: 10086.7929688, Throughput: 205.562897 im/s
Epoch 42, Train Loss: 9557.2451172, Time: 24.2847s, Throughput: 205.561660 im/s
Epoch 43, Iter 39, Loss: 9824.2753906, Throughput: 203.265134 im/s
Epoch 43, Train Loss: 9555.3037109, Time: 24.5592s, Throughput: 203.263890 im/s
Epoch 44, Iter 39, Loss: 9405.8378906, Throughput: 205.183244 im/s
Epoch 44, Train Loss: 9556.1396484, Time: 24.3296s, Throughput: 205.181887 im/s
Epoch 45, Iter 39, Loss: 10266.5000000, Throughput: 205.673643 im/s
Epoch 45, Train Loss: 9534.2451172, Time: 24.2716s, Throughput: 205.672303 im/s
Epoch 46, Iter 39, Loss: 9606.7695312, Throughput: 205.049972 im/s
Epoch 46, Train Loss: 9485.1142578, Time: 24.3454s, Throughput: 205.048649 im/s
Epoch 47, Iter 39, Loss: 9110.7871094, Throughput: 205.964114 im/s
Epoch 47, Train Loss: 9510.9863281, Time: 24.2374s, Throughput: 205.962932 im/s
Epoch 48, Iter 39, Loss: 10219.0595703, Throughput: 205.017693 im/s
Epoch 48, Train Loss: 9469.3017578, Time: 24.3493s, Throughput: 205.015925 im/s
Epoch 49, Iter 39, Loss: 9548.3593750, Throughput: 205.348412 im/s
Epoch 49, Train Loss: 9432.9921875, Time: 24.3100s, Throughput: 205.347256 im/s
Epoch 50, Iter 39, Loss: 9900.3320312, Throughput: 206.562729 im/s
Epoch 50, Train Loss: 9384.2275391, Time: 24.1671s, Throughput: 206.561437 im/s

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
