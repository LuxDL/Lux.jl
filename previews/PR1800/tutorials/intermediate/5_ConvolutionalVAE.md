---
url: /previews/PR1800/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 22928.5664062, Throughput: 15.495785 im/s
Epoch 1, Train Loss: 39643.3632812, Time: 322.4107s, Throughput: 15.483356 im/s
Epoch 2, Iter 39, Loss: 17304.5332031, Throughput: 97.744422 im/s
Epoch 2, Train Loss: 20206.0566406, Time: 51.0722s, Throughput: 97.744061 im/s
Epoch 3, Iter 39, Loss: 16819.1308594, Throughput: 97.704737 im/s
Epoch 3, Train Loss: 16612.0585938, Time: 51.0929s, Throughput: 97.704426 im/s
Epoch 4, Iter 39, Loss: 14763.3125000, Throughput: 97.954989 im/s
Epoch 4, Train Loss: 15133.0927734, Time: 50.9623s, Throughput: 97.954713 im/s
Epoch 5, Iter 39, Loss: 13575.7216797, Throughput: 97.587571 im/s
Epoch 5, Train Loss: 14167.5703125, Time: 51.1542s, Throughput: 97.587211 im/s
Epoch 6, Iter 39, Loss: 13165.9667969, Throughput: 97.603357 im/s
Epoch 6, Train Loss: 13497.9101562, Time: 51.1460s, Throughput: 97.603023 im/s
Epoch 7, Iter 39, Loss: 12660.0292969, Throughput: 97.965220 im/s
Epoch 7, Train Loss: 13047.7802734, Time: 50.9570s, Throughput: 97.964870 im/s
Epoch 8, Iter 39, Loss: 12582.3867188, Throughput: 98.058882 im/s
Epoch 8, Train Loss: 12616.5810547, Time: 50.9083s, Throughput: 98.058604 im/s
Epoch 9, Iter 39, Loss: 12309.8925781, Throughput: 97.891109 im/s
Epoch 9, Train Loss: 12349.6445312, Time: 50.9956s, Throughput: 97.890785 im/s
Epoch 10, Iter 39, Loss: 12125.1992188, Throughput: 97.842146 im/s
Epoch 10, Train Loss: 12003.7587891, Time: 51.0211s, Throughput: 97.841804 im/s
Epoch 11, Iter 39, Loss: 12640.1982422, Throughput: 97.840174 im/s
Epoch 11, Train Loss: 11771.9394531, Time: 51.0222s, Throughput: 97.839845 im/s
Epoch 12, Iter 39, Loss: 11421.0341797, Throughput: 97.812166 im/s
Epoch 12, Train Loss: 11689.0830078, Time: 51.0368s, Throughput: 97.811800 im/s
Epoch 13, Iter 39, Loss: 12160.2187500, Throughput: 97.724895 im/s
Epoch 13, Train Loss: 11517.4462891, Time: 51.0823s, Throughput: 97.724577 im/s
Epoch 14, Iter 39, Loss: 11236.2060547, Throughput: 97.705212 im/s
Epoch 14, Train Loss: 11364.2617188, Time: 51.0926s, Throughput: 97.704909 im/s
Epoch 15, Iter 39, Loss: 12076.4433594, Throughput: 97.853450 im/s
Epoch 15, Train Loss: 11226.4306641, Time: 51.0152s, Throughput: 97.853112 im/s
Epoch 16, Iter 39, Loss: 11201.9589844, Throughput: 97.453848 im/s
Epoch 16, Train Loss: 11262.7939453, Time: 51.2244s, Throughput: 97.453480 im/s
Epoch 17, Iter 39, Loss: 11081.7041016, Throughput: 97.541327 im/s
Epoch 17, Train Loss: 11056.5917969, Time: 51.1785s, Throughput: 97.541002 im/s
Epoch 18, Iter 39, Loss: 10382.2050781, Throughput: 97.847891 im/s
Epoch 18, Train Loss: 10897.8896484, Time: 51.0181s, Throughput: 97.847559 im/s
Epoch 19, Iter 39, Loss: 10890.0781250, Throughput: 96.912814 im/s
Epoch 19, Train Loss: 10857.6621094, Time: 51.5104s, Throughput: 96.912490 im/s
Epoch 20, Iter 39, Loss: 11457.5937500, Throughput: 98.214793 im/s
Epoch 20, Train Loss: 10747.0615234, Time: 50.8275s, Throughput: 98.214453 im/s
Epoch 21, Iter 39, Loss: 11148.6572266, Throughput: 97.942755 im/s
Epoch 21, Train Loss: 10670.7441406, Time: 50.9687s, Throughput: 97.942403 im/s
Epoch 22, Iter 39, Loss: 11601.3183594, Throughput: 98.195380 im/s
Epoch 22, Train Loss: 10673.2050781, Time: 50.8376s, Throughput: 98.194992 im/s
Epoch 23, Iter 39, Loss: 11213.8886719, Throughput: 98.106206 im/s
Epoch 23, Train Loss: 10515.8232422, Time: 50.8838s, Throughput: 98.105827 im/s
Epoch 24, Iter 39, Loss: 10408.5761719, Throughput: 98.199976 im/s
Epoch 24, Train Loss: 10419.0429688, Time: 50.8352s, Throughput: 98.199653 im/s
Epoch 25, Iter 39, Loss: 10735.0117188, Throughput: 98.108306 im/s
Epoch 25, Train Loss: 10374.5390625, Time: 50.8827s, Throughput: 98.107926 im/s
Epoch 26, Iter 39, Loss: 10863.7939453, Throughput: 97.884869 im/s
Epoch 26, Train Loss: 10320.4394531, Time: 50.9989s, Throughput: 97.884521 im/s
Epoch 27, Iter 39, Loss: 10336.4208984, Throughput: 97.989903 im/s
Epoch 27, Train Loss: 10322.7207031, Time: 50.9442s, Throughput: 97.989613 im/s
Epoch 28, Iter 39, Loss: 9793.2216797, Throughput: 97.732200 im/s
Epoch 28, Train Loss: 10253.2246094, Time: 51.0785s, Throughput: 97.731922 im/s
Epoch 29, Iter 39, Loss: 11076.3193359, Throughput: 98.346675 im/s
Epoch 29, Train Loss: 10167.2871094, Time: 50.7594s, Throughput: 98.346318 im/s
Epoch 30, Iter 39, Loss: 9636.0097656, Throughput: 98.203797 im/s
Epoch 30, Train Loss: 10106.5830078, Time: 50.8332s, Throughput: 98.203459 im/s
Epoch 31, Iter 39, Loss: 10465.9511719, Throughput: 98.390530 im/s
Epoch 31, Train Loss: 10152.7119141, Time: 50.7370s, Throughput: 98.389739 im/s
Epoch 32, Iter 39, Loss: 9471.9912109, Throughput: 97.926207 im/s
Epoch 32, Train Loss: 10032.7255859, Time: 50.9773s, Throughput: 97.925905 im/s
Epoch 33, Iter 39, Loss: 9371.3710938, Throughput: 97.802410 im/s
Epoch 33, Train Loss: 10045.0205078, Time: 51.0419s, Throughput: 97.802098 im/s
Epoch 34, Iter 39, Loss: 9552.7041016, Throughput: 98.200372 im/s
Epoch 34, Train Loss: 9969.1689453, Time: 50.8350s, Throughput: 98.200082 im/s
Epoch 35, Iter 39, Loss: 9136.0322266, Throughput: 98.196062 im/s
Epoch 35, Train Loss: 9890.2080078, Time: 50.8372s, Throughput: 98.195745 im/s
Epoch 36, Iter 39, Loss: 9622.3212891, Throughput: 98.136515 im/s
Epoch 36, Train Loss: 9843.3896484, Time: 50.8681s, Throughput: 98.136195 im/s
Epoch 37, Iter 39, Loss: 9640.5937500, Throughput: 98.220099 im/s
Epoch 37, Train Loss: 9819.3623047, Time: 50.8248s, Throughput: 98.219790 im/s
Epoch 38, Iter 39, Loss: 9934.2724609, Throughput: 97.627478 im/s
Epoch 38, Train Loss: 9793.5712891, Time: 51.1333s, Throughput: 97.627152 im/s
Epoch 39, Iter 39, Loss: 9784.1152344, Throughput: 98.232759 im/s
Epoch 39, Train Loss: 9772.3310547, Time: 50.8183s, Throughput: 98.232407 im/s
Epoch 40, Iter 39, Loss: 10787.0859375, Throughput: 98.197177 im/s
Epoch 40, Train Loss: 9731.3115234, Time: 50.8367s, Throughput: 98.196845 im/s
Epoch 41, Iter 39, Loss: 9619.7460938, Throughput: 98.208025 im/s
Epoch 41, Train Loss: 9742.9375000, Time: 50.8310s, Throughput: 98.207719 im/s
Epoch 42, Iter 39, Loss: 8972.7314453, Throughput: 98.151477 im/s
Epoch 42, Train Loss: 9690.5332031, Time: 50.8603s, Throughput: 98.151148 im/s
Epoch 43, Iter 39, Loss: 9424.4453125, Throughput: 97.937311 im/s
Epoch 43, Train Loss: 9746.1933594, Time: 50.9715s, Throughput: 97.936988 im/s
Epoch 44, Iter 39, Loss: 9588.2822266, Throughput: 98.237760 im/s
Epoch 44, Train Loss: 9613.3730469, Time: 50.8157s, Throughput: 98.237428 im/s
Epoch 45, Iter 39, Loss: 9470.1396484, Throughput: 97.742397 im/s
Epoch 45, Train Loss: 9570.0810547, Time: 51.0732s, Throughput: 97.742063 im/s
Epoch 46, Iter 39, Loss: 9999.3593750, Throughput: 98.132714 im/s
Epoch 46, Train Loss: 9585.3720703, Time: 50.8700s, Throughput: 98.132429 im/s
Epoch 47, Iter 39, Loss: 9360.5859375, Throughput: 98.154725 im/s
Epoch 47, Train Loss: 9432.5068359, Time: 50.8586s, Throughput: 98.154421 im/s
Epoch 48, Iter 39, Loss: 9976.5507812, Throughput: 97.877311 im/s
Epoch 48, Train Loss: 9526.1611328, Time: 51.0028s, Throughput: 97.876985 im/s
Epoch 49, Iter 39, Loss: 9105.9970703, Throughput: 98.155462 im/s
Epoch 49, Train Loss: 9481.1445312, Time: 50.8583s, Throughput: 98.155122 im/s
Epoch 50, Iter 39, Loss: 9682.2226562, Throughput: 98.129165 im/s
Epoch 50, Train Loss: 9473.0791016, Time: 50.8719s, Throughput: 98.128845 im/s

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
