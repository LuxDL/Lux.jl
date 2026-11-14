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
Epoch 1, Iter 39, Loss: 23920.7148438, Throughput: 16.275544 im/s
Epoch 1, Train Loss: 39480.1406250, Time: 306.9905s, Throughput: 16.261092 im/s
Epoch 2, Iter 39, Loss: 17446.4277344, Throughput: 173.804453 im/s
Epoch 2, Train Loss: 19967.7500000, Time: 28.7221s, Throughput: 173.803388 im/s
Epoch 3, Iter 39, Loss: 16156.3818359, Throughput: 172.873628 im/s
Epoch 3, Train Loss: 16348.8720703, Time: 28.8768s, Throughput: 172.872634 im/s
Epoch 4, Iter 39, Loss: 14667.4570312, Throughput: 174.072368 im/s
Epoch 4, Train Loss: 14845.0224609, Time: 28.6779s, Throughput: 174.071408 im/s
Epoch 5, Iter 39, Loss: 13041.0087891, Throughput: 173.901186 im/s
Epoch 5, Train Loss: 13884.0273438, Time: 28.7061s, Throughput: 173.900241 im/s
Epoch 6, Iter 39, Loss: 13272.8886719, Throughput: 172.953119 im/s
Epoch 6, Train Loss: 13257.2832031, Time: 28.8635s, Throughput: 172.952100 im/s
Epoch 7, Iter 39, Loss: 13671.8886719, Throughput: 173.734752 im/s
Epoch 7, Train Loss: 12931.2509766, Time: 28.7336s, Throughput: 173.733802 im/s
Epoch 8, Iter 39, Loss: 12759.9013672, Throughput: 172.642905 im/s
Epoch 8, Train Loss: 12474.2949219, Time: 28.9153s, Throughput: 172.642009 im/s
Epoch 9, Iter 39, Loss: 11426.2734375, Throughput: 173.484198 im/s
Epoch 9, Train Loss: 12145.1406250, Time: 28.7751s, Throughput: 173.483211 im/s
Epoch 10, Iter 39, Loss: 11287.2617188, Throughput: 173.069337 im/s
Epoch 10, Train Loss: 11940.1171875, Time: 28.8441s, Throughput: 173.068401 im/s
Epoch 11, Iter 39, Loss: 11721.7421875, Throughput: 173.181402 im/s
Epoch 11, Train Loss: 11701.3369141, Time: 28.8254s, Throughput: 173.180424 im/s
Epoch 12, Iter 39, Loss: 11718.6152344, Throughput: 173.353564 im/s
Epoch 12, Train Loss: 11508.9394531, Time: 28.7968s, Throughput: 173.352577 im/s
Epoch 13, Iter 39, Loss: 11713.3125000, Throughput: 173.418314 im/s
Epoch 13, Train Loss: 11328.9433594, Time: 28.7861s, Throughput: 173.417332 im/s
Epoch 14, Iter 39, Loss: 10748.4511719, Throughput: 170.826713 im/s
Epoch 14, Train Loss: 11120.6787109, Time: 29.2228s, Throughput: 170.825672 im/s
Epoch 15, Iter 39, Loss: 10500.9521484, Throughput: 173.761758 im/s
Epoch 15, Train Loss: 11029.4755859, Time: 28.7292s, Throughput: 173.760761 im/s
Epoch 16, Iter 39, Loss: 10895.0351562, Throughput: 172.797774 im/s
Epoch 16, Train Loss: 11005.5156250, Time: 28.8894s, Throughput: 172.796907 im/s
Epoch 17, Iter 39, Loss: 10444.2812500, Throughput: 174.138124 im/s
Epoch 17, Train Loss: 10898.0732422, Time: 28.6671s, Throughput: 174.137072 im/s
Epoch 18, Iter 39, Loss: 10666.2900391, Throughput: 173.242334 im/s
Epoch 18, Train Loss: 10776.2568359, Time: 28.8153s, Throughput: 173.241348 im/s
Epoch 19, Iter 39, Loss: 10134.0371094, Throughput: 173.198606 im/s
Epoch 19, Train Loss: 10622.8740234, Time: 28.8226s, Throughput: 173.197457 im/s
Epoch 20, Iter 39, Loss: 10677.2128906, Throughput: 172.809486 im/s
Epoch 20, Train Loss: 10528.4755859, Time: 28.8875s, Throughput: 172.808577 im/s
Epoch 21, Iter 39, Loss: 10372.8945312, Throughput: 173.588129 im/s
Epoch 21, Train Loss: 10470.7880859, Time: 28.7579s, Throughput: 173.587144 im/s
Epoch 22, Iter 39, Loss: 10230.7460938, Throughput: 173.076633 im/s
Epoch 22, Train Loss: 10464.2158203, Time: 28.8429s, Throughput: 173.075686 im/s
Epoch 23, Iter 39, Loss: 10458.8261719, Throughput: 173.755420 im/s
Epoch 23, Train Loss: 10390.3730469, Time: 28.7302s, Throughput: 173.754410 im/s
Epoch 24, Iter 39, Loss: 10467.3076172, Throughput: 173.199200 im/s
Epoch 24, Train Loss: 10339.9306641, Time: 28.8225s, Throughput: 173.198233 im/s
Epoch 25, Iter 39, Loss: 11030.8613281, Throughput: 173.826596 im/s
Epoch 25, Train Loss: 10204.0429688, Time: 28.7184s, Throughput: 173.825574 im/s
Epoch 26, Iter 39, Loss: 10673.6318359, Throughput: 172.720038 im/s
Epoch 26, Train Loss: 10117.6904297, Time: 28.9024s, Throughput: 172.718951 im/s
Epoch 27, Iter 39, Loss: 10339.2822266, Throughput: 173.170872 im/s
Epoch 27, Train Loss: 10095.3281250, Time: 28.8272s, Throughput: 173.169886 im/s
Epoch 28, Iter 39, Loss: 9806.0703125, Throughput: 173.278715 im/s
Epoch 28, Train Loss: 10127.3535156, Time: 28.8093s, Throughput: 173.277415 im/s
Epoch 29, Iter 39, Loss: 10753.8378906, Throughput: 173.561404 im/s
Epoch 29, Train Loss: 10032.6894531, Time: 28.7623s, Throughput: 173.560415 im/s
Epoch 30, Iter 39, Loss: 10509.1308594, Throughput: 173.173538 im/s
Epoch 30, Train Loss: 9950.7773438, Time: 28.8267s, Throughput: 173.172607 im/s
Epoch 31, Iter 39, Loss: 9511.3046875, Throughput: 173.060895 im/s
Epoch 31, Train Loss: 9910.6328125, Time: 28.8455s, Throughput: 173.059928 im/s
Epoch 32, Iter 39, Loss: 9657.8486328, Throughput: 172.700262 im/s
Epoch 32, Train Loss: 9787.0712891, Time: 28.9057s, Throughput: 172.699366 im/s
Epoch 33, Iter 39, Loss: 11175.7832031, Throughput: 171.099586 im/s
Epoch 33, Train Loss: 9799.6298828, Time: 29.1762s, Throughput: 171.098637 im/s
Epoch 34, Iter 39, Loss: 9360.1035156, Throughput: 173.137646 im/s
Epoch 34, Train Loss: 9855.4707031, Time: 28.8327s, Throughput: 173.136649 im/s
Epoch 35, Iter 39, Loss: 9834.1542969, Throughput: 172.466951 im/s
Epoch 35, Train Loss: 9806.5429688, Time: 28.9449s, Throughput: 172.465843 im/s
Epoch 36, Iter 39, Loss: 9328.7216797, Throughput: 173.518414 im/s
Epoch 36, Train Loss: 9665.7011719, Time: 28.7694s, Throughput: 173.517498 im/s
Epoch 37, Iter 39, Loss: 9679.1679688, Throughput: 173.498385 im/s
Epoch 37, Train Loss: 9753.4716797, Time: 28.7728s, Throughput: 173.497409 im/s
Epoch 38, Iter 39, Loss: 9790.8730469, Throughput: 173.875621 im/s
Epoch 38, Train Loss: 9746.0546875, Time: 28.7103s, Throughput: 173.874627 im/s
Epoch 39, Iter 39, Loss: 9792.4433594, Throughput: 173.094966 im/s
Epoch 39, Train Loss: 9630.0292969, Time: 28.8398s, Throughput: 173.093995 im/s
Epoch 40, Iter 39, Loss: 9921.5898438, Throughput: 173.755492 im/s
Epoch 40, Train Loss: 9569.2578125, Time: 28.7302s, Throughput: 173.754567 im/s
Epoch 41, Iter 39, Loss: 9347.1425781, Throughput: 172.863882 im/s
Epoch 41, Train Loss: 9580.1289062, Time: 28.8784s, Throughput: 172.862859 im/s
Epoch 42, Iter 39, Loss: 9874.7148438, Throughput: 173.734364 im/s
Epoch 42, Train Loss: 9450.4843750, Time: 28.7337s, Throughput: 173.733378 im/s
Epoch 43, Iter 39, Loss: 10051.5000000, Throughput: 173.292218 im/s
Epoch 43, Train Loss: 9483.2880859, Time: 28.8070s, Throughput: 173.291273 im/s
Epoch 44, Iter 39, Loss: 10400.2871094, Throughput: 173.880241 im/s
Epoch 44, Train Loss: 9545.8613281, Time: 28.7096s, Throughput: 173.879327 im/s
Epoch 45, Iter 39, Loss: 9879.9296875, Throughput: 173.352835 im/s
Epoch 45, Train Loss: 9460.6806641, Time: 28.7969s, Throughput: 173.351908 im/s
Epoch 46, Iter 39, Loss: 9961.0820312, Throughput: 173.493562 im/s
Epoch 46, Train Loss: 9381.0019531, Time: 28.7736s, Throughput: 173.492537 im/s
Epoch 47, Iter 39, Loss: 9297.5781250, Throughput: 172.581197 im/s
Epoch 47, Train Loss: 9353.8574219, Time: 28.9257s, Throughput: 172.580159 im/s
Epoch 48, Iter 39, Loss: 9478.7929688, Throughput: 174.134528 im/s
Epoch 48, Train Loss: 9325.7060547, Time: 28.6677s, Throughput: 174.133429 im/s
Epoch 49, Iter 39, Loss: 9441.6533203, Throughput: 173.257574 im/s
Epoch 49, Train Loss: 9322.8945312, Time: 28.8128s, Throughput: 173.256631 im/s
Epoch 50, Iter 39, Loss: 9134.7773438, Throughput: 173.267582 im/s
Epoch 50, Train Loss: 9334.7841797, Time: 28.8111s, Throughput: 173.266674 im/s

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
