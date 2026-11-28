---
url: /previews/PR1822/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24570.8203125, Throughput: 16.360413 im/s
Epoch 1, Train Loss: 39958.3203125, Time: 305.3801s, Throughput: 16.346844 im/s
Epoch 2, Iter 39, Loss: 17801.8691406, Throughput: 158.677944 im/s
Epoch 2, Train Loss: 20465.3515625, Time: 31.4601s, Throughput: 158.677087 im/s
Epoch 3, Iter 39, Loss: 15597.1943359, Throughput: 158.895999 im/s
Epoch 3, Train Loss: 17037.1601562, Time: 31.4169s, Throughput: 158.895225 im/s
Epoch 4, Iter 39, Loss: 16024.0087891, Throughput: 158.961250 im/s
Epoch 4, Train Loss: 15520.6953125, Time: 31.4040s, Throughput: 158.960415 im/s
Epoch 5, Iter 39, Loss: 14219.3115234, Throughput: 158.804149 im/s
Epoch 5, Train Loss: 14625.5205078, Time: 31.4351s, Throughput: 158.803245 im/s
Epoch 6, Iter 39, Loss: 13706.4277344, Throughput: 159.016053 im/s
Epoch 6, Train Loss: 13908.2851562, Time: 31.3932s, Throughput: 159.015308 im/s
Epoch 7, Iter 39, Loss: 13831.2890625, Throughput: 158.562367 im/s
Epoch 7, Train Loss: 13366.0429688, Time: 31.4831s, Throughput: 158.561476 im/s
Epoch 8, Iter 39, Loss: 13138.0771484, Throughput: 159.130383 im/s
Epoch 8, Train Loss: 12973.8232422, Time: 31.3707s, Throughput: 159.129510 im/s
Epoch 9, Iter 39, Loss: 12394.3300781, Throughput: 156.957909 im/s
Epoch 9, Train Loss: 12710.4541016, Time: 31.8049s, Throughput: 156.957031 im/s
Epoch 10, Iter 39, Loss: 11725.0673828, Throughput: 159.084687 im/s
Epoch 10, Train Loss: 12399.3193359, Time: 31.3798s, Throughput: 159.083480 im/s
Epoch 11, Iter 39, Loss: 12138.7988281, Throughput: 159.104182 im/s
Epoch 11, Train Loss: 12329.9687500, Time: 31.3758s, Throughput: 159.103269 im/s
Epoch 12, Iter 39, Loss: 12375.1562500, Throughput: 159.190704 im/s
Epoch 12, Train Loss: 12149.5722656, Time: 31.3588s, Throughput: 159.189832 im/s
Epoch 13, Iter 39, Loss: 11629.2138672, Throughput: 159.011680 im/s
Epoch 13, Train Loss: 11861.6484375, Time: 31.3941s, Throughput: 159.010749 im/s
Epoch 14, Iter 39, Loss: 11808.6591797, Throughput: 158.909731 im/s
Epoch 14, Train Loss: 11661.1669922, Time: 31.4142s, Throughput: 158.908821 im/s
Epoch 15, Iter 39, Loss: 11777.5644531, Throughput: 159.063403 im/s
Epoch 15, Train Loss: 11571.8701172, Time: 31.3839s, Throughput: 159.062637 im/s
Epoch 16, Iter 39, Loss: 11452.7753906, Throughput: 158.961082 im/s
Epoch 16, Train Loss: 11436.9716797, Time: 31.4041s, Throughput: 158.960266 im/s
Epoch 17, Iter 39, Loss: 11545.0078125, Throughput: 158.939932 im/s
Epoch 17, Train Loss: 11330.8916016, Time: 31.4082s, Throughput: 158.939152 im/s
Epoch 18, Iter 39, Loss: 11938.9492188, Throughput: 158.895942 im/s
Epoch 18, Train Loss: 11242.6708984, Time: 31.4170s, Throughput: 158.894976 im/s
Epoch 19, Iter 39, Loss: 12079.3076172, Throughput: 159.086771 im/s
Epoch 19, Train Loss: 11262.9345703, Time: 31.3793s, Throughput: 159.085975 im/s
Epoch 20, Iter 39, Loss: 11229.5000000, Throughput: 158.681756 im/s
Epoch 20, Train Loss: 11097.5400391, Time: 31.4594s, Throughput: 158.680913 im/s
Epoch 21, Iter 39, Loss: 10628.1074219, Throughput: 159.120487 im/s
Epoch 21, Train Loss: 11007.4843750, Time: 31.3726s, Throughput: 159.119690 im/s
Epoch 22, Iter 39, Loss: 10433.2548828, Throughput: 158.718596 im/s
Epoch 22, Train Loss: 10898.9580078, Time: 31.4521s, Throughput: 158.717708 im/s
Epoch 23, Iter 39, Loss: 10950.3955078, Throughput: 158.632260 im/s
Epoch 23, Train Loss: 10804.1875000, Time: 31.4692s, Throughput: 158.631277 im/s
Epoch 24, Iter 39, Loss: 10780.7255859, Throughput: 158.218785 im/s
Epoch 24, Train Loss: 10886.0751953, Time: 31.5514s, Throughput: 158.218008 im/s
Epoch 25, Iter 39, Loss: 10953.5107422, Throughput: 159.228381 im/s
Epoch 25, Train Loss: 10722.5205078, Time: 31.3514s, Throughput: 159.227309 im/s
Epoch 26, Iter 39, Loss: 10448.1337891, Throughput: 159.093235 im/s
Epoch 26, Train Loss: 10604.7148438, Time: 31.3780s, Throughput: 159.092242 im/s
Epoch 27, Iter 39, Loss: 10775.5595703, Throughput: 159.089255 im/s
Epoch 27, Train Loss: 10541.9960938, Time: 31.3788s, Throughput: 159.088514 im/s
Epoch 28, Iter 39, Loss: 10911.1777344, Throughput: 158.911674 im/s
Epoch 28, Train Loss: 10504.9423828, Time: 31.4138s, Throughput: 158.910829 im/s
Epoch 29, Iter 39, Loss: 11037.6738281, Throughput: 159.293660 im/s
Epoch 29, Train Loss: 10492.0332031, Time: 31.3385s, Throughput: 159.292836 im/s
Epoch 30, Iter 39, Loss: 10441.4599609, Throughput: 157.106240 im/s
Epoch 30, Train Loss: 10445.1357422, Time: 31.7749s, Throughput: 157.105359 im/s
Epoch 31, Iter 39, Loss: 10551.0712891, Throughput: 159.077301 im/s
Epoch 31, Train Loss: 10343.8769531, Time: 31.3811s, Throughput: 159.076434 im/s
Epoch 32, Iter 39, Loss: 10239.1650391, Throughput: 159.129273 im/s
Epoch 32, Train Loss: 10286.1357422, Time: 31.3709s, Throughput: 159.128496 im/s
Epoch 33, Iter 39, Loss: 10564.9531250, Throughput: 158.850220 im/s
Epoch 33, Train Loss: 10284.4248047, Time: 31.4260s, Throughput: 158.849391 im/s
Epoch 34, Iter 39, Loss: 9647.6777344, Throughput: 158.855365 im/s
Epoch 34, Train Loss: 10198.6630859, Time: 31.4250s, Throughput: 158.854536 im/s
Epoch 35, Iter 39, Loss: 10282.0703125, Throughput: 159.286128 im/s
Epoch 35, Train Loss: 10182.5468750, Time: 31.3400s, Throughput: 159.285248 im/s
Epoch 36, Iter 39, Loss: 10399.4052734, Throughput: 158.928223 im/s
Epoch 36, Train Loss: 10111.0361328, Time: 31.4106s, Throughput: 158.927490 im/s
Epoch 37, Iter 39, Loss: 10550.9218750, Throughput: 158.955459 im/s
Epoch 37, Train Loss: 10220.4726562, Time: 31.4052s, Throughput: 158.954684 im/s
Epoch 38, Iter 39, Loss: 10416.5937500, Throughput: 158.835531 im/s
Epoch 38, Train Loss: 10076.9687500, Time: 31.4289s, Throughput: 158.834577 im/s
Epoch 39, Iter 39, Loss: 10204.3691406, Throughput: 158.787333 im/s
Epoch 39, Train Loss: 10083.4375000, Time: 31.4384s, Throughput: 158.786469 im/s
Epoch 40, Iter 39, Loss: 10275.6279297, Throughput: 159.312378 im/s
Epoch 40, Train Loss: 9964.9218750, Time: 31.3348s, Throughput: 159.311539 im/s
Epoch 41, Iter 39, Loss: 9615.4199219, Throughput: 158.879740 im/s
Epoch 41, Train Loss: 9945.6914062, Time: 31.4202s, Throughput: 158.878890 im/s
Epoch 42, Iter 39, Loss: 9926.5771484, Throughput: 159.150488 im/s
Epoch 42, Train Loss: 9941.5664062, Time: 31.3667s, Throughput: 159.149554 im/s
Epoch 43, Iter 39, Loss: 9197.0742188, Throughput: 159.038064 im/s
Epoch 43, Train Loss: 9901.2011719, Time: 31.3889s, Throughput: 159.037228 im/s
Epoch 44, Iter 39, Loss: 9364.2480469, Throughput: 159.246398 im/s
Epoch 44, Train Loss: 9849.8974609, Time: 31.3478s, Throughput: 159.245544 im/s
Epoch 45, Iter 39, Loss: 10793.9960938, Throughput: 158.985713 im/s
Epoch 45, Train Loss: 9814.9433594, Time: 31.3992s, Throughput: 158.984862 im/s
Epoch 46, Iter 39, Loss: 10191.4521484, Throughput: 159.021169 im/s
Epoch 46, Train Loss: 9875.7451172, Time: 31.3922s, Throughput: 159.020272 im/s
Epoch 47, Iter 39, Loss: 9405.5000000, Throughput: 159.032988 im/s
Epoch 47, Train Loss: 9905.4433594, Time: 31.3899s, Throughput: 159.032207 im/s
Epoch 48, Iter 39, Loss: 9586.2001953, Throughput: 158.912594 im/s
Epoch 48, Train Loss: 9747.1035156, Time: 31.4137s, Throughput: 158.911497 im/s
Epoch 49, Iter 39, Loss: 10009.8730469, Throughput: 158.660363 im/s
Epoch 49, Train Loss: 9726.9335938, Time: 31.4636s, Throughput: 158.659510 im/s
Epoch 50, Iter 39, Loss: 9957.6904297, Throughput: 158.839818 im/s
Epoch 50, Train Loss: 9639.6757812, Time: 31.4280s, Throughput: 158.839025 im/s

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
