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
Epoch 1, Iter 39, Loss: 24765.9375000, Throughput: 5.460590 im/s
Epoch 1, Train Loss: 39879.9101562, Time: 914.5046s, Throughput: 5.458693 im/s
Epoch 2, Iter 39, Loss: 17594.8496094, Throughput: 86.491929 im/s
Epoch 2, Train Loss: 20042.3925781, Time: 57.7166s, Throughput: 86.491622 im/s
Epoch 3, Iter 39, Loss: 15535.0996094, Throughput: 87.099749 im/s
Epoch 3, Train Loss: 16497.7460938, Time: 57.3138s, Throughput: 87.099453 im/s
Epoch 4, Iter 39, Loss: 14573.3632812, Throughput: 86.624056 im/s
Epoch 4, Train Loss: 15037.9755859, Time: 57.6285s, Throughput: 86.623759 im/s
Epoch 5, Iter 39, Loss: 14076.3964844, Throughput: 86.855704 im/s
Epoch 5, Train Loss: 14107.4599609, Time: 57.4748s, Throughput: 86.855414 im/s
Epoch 6, Iter 39, Loss: 12715.8066406, Throughput: 88.302768 im/s
Epoch 6, Train Loss: 13469.7656250, Time: 56.5330s, Throughput: 88.302470 im/s
Epoch 7, Iter 39, Loss: 12504.2792969, Throughput: 88.185573 im/s
Epoch 7, Train Loss: 12975.6513672, Time: 56.6081s, Throughput: 88.185300 im/s
Epoch 8, Iter 39, Loss: 11972.4570312, Throughput: 88.170858 im/s
Epoch 8, Train Loss: 12599.6269531, Time: 56.6175s, Throughput: 88.170542 im/s
Epoch 9, Iter 39, Loss: 12250.9062500, Throughput: 88.122257 im/s
Epoch 9, Train Loss: 12293.0732422, Time: 56.6488s, Throughput: 88.121946 im/s
Epoch 10, Iter 39, Loss: 12147.8222656, Throughput: 89.306181 im/s
Epoch 10, Train Loss: 11963.8798828, Time: 55.8978s, Throughput: 89.305835 im/s
Epoch 11, Iter 39, Loss: 11834.6054688, Throughput: 88.851529 im/s
Epoch 11, Train Loss: 11789.0292969, Time: 56.1838s, Throughput: 88.851203 im/s
Epoch 12, Iter 39, Loss: 11436.6640625, Throughput: 88.041957 im/s
Epoch 12, Train Loss: 11618.1582031, Time: 56.7004s, Throughput: 88.041651 im/s
Epoch 13, Iter 39, Loss: 11218.4707031, Throughput: 88.463301 im/s
Epoch 13, Train Loss: 11611.7656250, Time: 56.4304s, Throughput: 88.462988 im/s
Epoch 14, Iter 39, Loss: 11028.3349609, Throughput: 88.297920 im/s
Epoch 14, Train Loss: 11309.4326172, Time: 56.5361s, Throughput: 88.297612 im/s
Epoch 15, Iter 39, Loss: 11021.3681641, Throughput: 89.113541 im/s
Epoch 15, Train Loss: 11170.9482422, Time: 56.0186s, Throughput: 89.113241 im/s
Epoch 16, Iter 39, Loss: 11405.4287109, Throughput: 89.089082 im/s
Epoch 16, Train Loss: 11055.0019531, Time: 56.0340s, Throughput: 89.088782 im/s
Epoch 17, Iter 39, Loss: 11200.2226562, Throughput: 89.153592 im/s
Epoch 17, Train Loss: 10917.3691406, Time: 55.9935s, Throughput: 89.153212 im/s
Epoch 18, Iter 39, Loss: 10905.9150391, Throughput: 88.376103 im/s
Epoch 18, Train Loss: 10906.7919922, Time: 56.4860s, Throughput: 88.375806 im/s
Epoch 19, Iter 39, Loss: 11240.4746094, Throughput: 89.079463 im/s
Epoch 19, Train Loss: 10837.8681641, Time: 56.0401s, Throughput: 89.079147 im/s
Epoch 20, Iter 39, Loss: 10538.3427734, Throughput: 88.644065 im/s
Epoch 20, Train Loss: 10713.4023438, Time: 56.3153s, Throughput: 88.643750 im/s
Epoch 21, Iter 39, Loss: 10732.3935547, Throughput: 88.565937 im/s
Epoch 21, Train Loss: 10668.2031250, Time: 56.3650s, Throughput: 88.565658 im/s
Epoch 22, Iter 39, Loss: 10938.8007812, Throughput: 87.687548 im/s
Epoch 22, Train Loss: 10632.6337891, Time: 56.9296s, Throughput: 87.687246 im/s
Epoch 23, Iter 39, Loss: 10274.5927734, Throughput: 87.276748 im/s
Epoch 23, Train Loss: 10547.1972656, Time: 57.1977s, Throughput: 87.276203 im/s
Epoch 24, Iter 39, Loss: 9552.5507812, Throughput: 88.093575 im/s
Epoch 24, Train Loss: 10384.9746094, Time: 56.6672s, Throughput: 88.093303 im/s
Epoch 25, Iter 39, Loss: 10640.5566406, Throughput: 88.952236 im/s
Epoch 25, Train Loss: 10434.6484375, Time: 56.1202s, Throughput: 88.951927 im/s
Epoch 26, Iter 39, Loss: 9955.5908203, Throughput: 88.326316 im/s
Epoch 26, Train Loss: 10294.6679688, Time: 56.5179s, Throughput: 88.326004 im/s
Epoch 27, Iter 39, Loss: 11146.9238281, Throughput: 87.849626 im/s
Epoch 27, Train Loss: 10239.6953125, Time: 56.8246s, Throughput: 87.849298 im/s
Epoch 28, Iter 39, Loss: 9901.1875000, Throughput: 88.325602 im/s
Epoch 28, Train Loss: 10265.0244141, Time: 56.5184s, Throughput: 88.325290 im/s
Epoch 29, Iter 39, Loss: 10374.0205078, Throughput: 87.376383 im/s
Epoch 29, Train Loss: 10174.7929688, Time: 57.1323s, Throughput: 87.376088 im/s
Epoch 30, Iter 39, Loss: 9898.6855469, Throughput: 88.115778 im/s
Epoch 30, Train Loss: 10109.9931641, Time: 56.6529s, Throughput: 88.115472 im/s
Epoch 31, Iter 39, Loss: 10419.9316406, Throughput: 87.952286 im/s
Epoch 31, Train Loss: 10095.0810547, Time: 56.7583s, Throughput: 87.951953 im/s
Epoch 32, Iter 39, Loss: 9900.8789062, Throughput: 88.157872 im/s
Epoch 32, Train Loss: 10028.5449219, Time: 56.6259s, Throughput: 88.157573 im/s
Epoch 33, Iter 39, Loss: 10002.7685547, Throughput: 88.667552 im/s
Epoch 33, Train Loss: 10008.4121094, Time: 56.3004s, Throughput: 88.667252 im/s
Epoch 34, Iter 39, Loss: 10076.6357422, Throughput: 88.851717 im/s
Epoch 34, Train Loss: 9916.7832031, Time: 56.1837s, Throughput: 88.851429 im/s
Epoch 35, Iter 39, Loss: 10163.4218750, Throughput: 89.436283 im/s
Epoch 35, Train Loss: 9864.8300781, Time: 55.8165s, Throughput: 89.435974 im/s
Epoch 36, Iter 39, Loss: 9841.3867188, Throughput: 87.553644 im/s
Epoch 36, Train Loss: 9949.9218750, Time: 57.0167s, Throughput: 87.553340 im/s
Epoch 37, Iter 39, Loss: 9207.7851562, Throughput: 87.385676 im/s
Epoch 37, Train Loss: 9754.8818359, Time: 57.1263s, Throughput: 87.385355 im/s
Epoch 38, Iter 39, Loss: 10417.5527344, Throughput: 87.549565 im/s
Epoch 38, Train Loss: 9784.5732422, Time: 57.0193s, Throughput: 87.549250 im/s
Epoch 39, Iter 39, Loss: 10575.4570312, Throughput: 88.273897 im/s
Epoch 39, Train Loss: 9806.4316406, Time: 56.5515s, Throughput: 88.273588 im/s
Epoch 40, Iter 39, Loss: 9903.7919922, Throughput: 88.741169 im/s
Epoch 40, Train Loss: 9773.1748047, Time: 56.2537s, Throughput: 88.740831 im/s
Epoch 41, Iter 39, Loss: 9508.0712891, Throughput: 88.289423 im/s
Epoch 41, Train Loss: 9746.3623047, Time: 56.5415s, Throughput: 88.289112 im/s
Epoch 42, Iter 39, Loss: 9622.5205078, Throughput: 88.975881 im/s
Epoch 42, Train Loss: 9672.9589844, Time: 56.1054s, Throughput: 88.975456 im/s
Epoch 43, Iter 39, Loss: 9777.4453125, Throughput: 88.459131 im/s
Epoch 43, Train Loss: 9619.6220703, Time: 56.4331s, Throughput: 88.458799 im/s
Epoch 44, Iter 39, Loss: 9881.0292969, Throughput: 88.872800 im/s
Epoch 44, Train Loss: 9563.3964844, Time: 56.1704s, Throughput: 88.872491 im/s
Epoch 45, Iter 39, Loss: 9554.9970703, Throughput: 87.972970 im/s
Epoch 45, Train Loss: 9485.3896484, Time: 56.7449s, Throughput: 87.972612 im/s
Epoch 46, Iter 39, Loss: 9446.1591797, Throughput: 88.771831 im/s
Epoch 46, Train Loss: 9542.0605469, Time: 56.2343s, Throughput: 88.771496 im/s
Epoch 47, Iter 39, Loss: 9005.0263672, Throughput: 88.235382 im/s
Epoch 47, Train Loss: 9499.3544922, Time: 56.5761s, Throughput: 88.235073 im/s
Epoch 48, Iter 39, Loss: 9206.9462891, Throughput: 88.027574 im/s
Epoch 48, Train Loss: 9540.9580078, Time: 56.7097s, Throughput: 88.027227 im/s
Epoch 49, Iter 39, Loss: 9919.7802734, Throughput: 89.391846 im/s
Epoch 49, Train Loss: 9563.1757812, Time: 55.8443s, Throughput: 89.391445 im/s
Epoch 50, Iter 39, Loss: 9114.7812500, Throughput: 90.295210 im/s
Epoch 50, Train Loss: 9519.9033203, Time: 55.2856s, Throughput: 90.294832 im/s

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
Julia Version 1.12.5
Commit 5fe89b8ddc1 (2026-02-09 16:05 UTC)
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
