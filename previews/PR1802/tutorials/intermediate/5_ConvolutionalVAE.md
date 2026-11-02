---
url: /previews/PR1802/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24751.0800781, Throughput: 14.155525 im/s
Epoch 1, Train Loss: 39714.5976562, Time: 352.9177s, Throughput: 14.144942 im/s
Epoch 2, Iter 39, Loss: 17699.5722656, Throughput: 168.794838 im/s
Epoch 2, Train Loss: 20285.7558594, Time: 29.5745s, Throughput: 168.793999 im/s
Epoch 3, Iter 39, Loss: 15144.0087891, Throughput: 169.224579 im/s
Epoch 3, Train Loss: 16609.1464844, Time: 29.4994s, Throughput: 169.223677 im/s
Epoch 4, Iter 39, Loss: 14980.0429688, Throughput: 169.233127 im/s
Epoch 4, Train Loss: 14977.9726562, Time: 29.4980s, Throughput: 169.232082 im/s
Epoch 5, Iter 39, Loss: 13528.3691406, Throughput: 169.280483 im/s
Epoch 5, Train Loss: 14110.6044922, Time: 29.4897s, Throughput: 169.279639 im/s
Epoch 6, Iter 39, Loss: 12593.3906250, Throughput: 169.155864 im/s
Epoch 6, Train Loss: 13348.3398438, Time: 29.5114s, Throughput: 169.154947 im/s
Epoch 7, Iter 39, Loss: 12991.6689453, Throughput: 169.631018 im/s
Epoch 7, Train Loss: 12845.3710938, Time: 29.4287s, Throughput: 169.630062 im/s
Epoch 8, Iter 39, Loss: 12449.0332031, Throughput: 169.058593 im/s
Epoch 8, Train Loss: 12474.9443359, Time: 29.5284s, Throughput: 169.057476 im/s
Epoch 9, Iter 39, Loss: 12536.4589844, Throughput: 168.928112 im/s
Epoch 9, Train Loss: 12154.1181641, Time: 29.5512s, Throughput: 168.927237 im/s
Epoch 10, Iter 39, Loss: 10842.5576172, Throughput: 169.305853 im/s
Epoch 10, Train Loss: 11935.5419922, Time: 29.4853s, Throughput: 169.304923 im/s
Epoch 11, Iter 39, Loss: 11744.5263672, Throughput: 169.198549 im/s
Epoch 11, Train Loss: 11825.5185547, Time: 29.5039s, Throughput: 169.197712 im/s
Epoch 12, Iter 39, Loss: 12288.6269531, Throughput: 169.174318 im/s
Epoch 12, Train Loss: 11590.1357422, Time: 29.5082s, Throughput: 169.173401 im/s
Epoch 13, Iter 39, Loss: 11688.4902344, Throughput: 169.615952 im/s
Epoch 13, Train Loss: 11455.1533203, Time: 29.4314s, Throughput: 169.614943 im/s
Epoch 14, Iter 39, Loss: 10970.3964844, Throughput: 169.069168 im/s
Epoch 14, Train Loss: 11233.0107422, Time: 29.5265s, Throughput: 169.068321 im/s
Epoch 15, Iter 39, Loss: 10661.5078125, Throughput: 167.694405 im/s
Epoch 15, Train Loss: 11130.4580078, Time: 29.7686s, Throughput: 167.693504 im/s
Epoch 16, Iter 39, Loss: 10954.9423828, Throughput: 169.048637 im/s
Epoch 16, Train Loss: 10994.9560547, Time: 29.5301s, Throughput: 169.047669 im/s
Epoch 17, Iter 39, Loss: 10861.3388672, Throughput: 169.888068 im/s
Epoch 17, Train Loss: 10836.0791016, Time: 29.3842s, Throughput: 169.887156 im/s
Epoch 18, Iter 39, Loss: 10942.2968750, Throughput: 169.970044 im/s
Epoch 18, Train Loss: 10811.4648438, Time: 29.3700s, Throughput: 169.969165 im/s
Epoch 19, Iter 39, Loss: 10915.5507812, Throughput: 170.221002 im/s
Epoch 19, Train Loss: 10756.8398438, Time: 29.3268s, Throughput: 170.219986 im/s
Epoch 20, Iter 39, Loss: 10929.1074219, Throughput: 169.399333 im/s
Epoch 20, Train Loss: 10649.5947266, Time: 29.4690s, Throughput: 169.398178 im/s
Epoch 21, Iter 39, Loss: 10508.7597656, Throughput: 169.831415 im/s
Epoch 21, Train Loss: 10613.8369141, Time: 29.3940s, Throughput: 169.830444 im/s
Epoch 22, Iter 39, Loss: 10101.1152344, Throughput: 169.373045 im/s
Epoch 22, Train Loss: 10485.1650391, Time: 29.4736s, Throughput: 169.372149 im/s
Epoch 23, Iter 39, Loss: 11390.9667969, Throughput: 169.718403 im/s
Epoch 23, Train Loss: 10358.2597656, Time: 29.4136s, Throughput: 169.717197 im/s
Epoch 24, Iter 39, Loss: 10701.5419922, Throughput: 169.209176 im/s
Epoch 24, Train Loss: 10350.2929688, Time: 29.5021s, Throughput: 169.208167 im/s
Epoch 25, Iter 39, Loss: 9367.5029297, Throughput: 168.809628 im/s
Epoch 25, Train Loss: 10397.7792969, Time: 29.5719s, Throughput: 168.808686 im/s
Epoch 26, Iter 39, Loss: 10527.3945312, Throughput: 169.007944 im/s
Epoch 26, Train Loss: 10160.7519531, Time: 29.5372s, Throughput: 169.006984 im/s
Epoch 27, Iter 39, Loss: 10320.3095703, Throughput: 169.631623 im/s
Epoch 27, Train Loss: 10198.2744141, Time: 29.4287s, Throughput: 169.630493 im/s
Epoch 28, Iter 39, Loss: 9763.6552734, Throughput: 168.810335 im/s
Epoch 28, Train Loss: 10028.5644531, Time: 29.5718s, Throughput: 168.809411 im/s
Epoch 29, Iter 39, Loss: 10412.4833984, Throughput: 169.932951 im/s
Epoch 29, Train Loss: 10002.3515625, Time: 29.3765s, Throughput: 169.932037 im/s
Epoch 30, Iter 39, Loss: 9800.6044922, Throughput: 169.351837 im/s
Epoch 30, Train Loss: 10005.0869141, Time: 29.4772s, Throughput: 169.351033 im/s
Epoch 31, Iter 39, Loss: 10354.2011719, Throughput: 168.942580 im/s
Epoch 31, Train Loss: 9930.9218750, Time: 29.5487s, Throughput: 168.941608 im/s
Epoch 32, Iter 39, Loss: 10174.6308594, Throughput: 167.270173 im/s
Epoch 32, Train Loss: 9908.0683594, Time: 29.8441s, Throughput: 167.269202 im/s
Epoch 33, Iter 39, Loss: 9826.3164062, Throughput: 169.564487 im/s
Epoch 33, Train Loss: 9841.8105469, Time: 29.4403s, Throughput: 169.563628 im/s
Epoch 34, Iter 39, Loss: 9553.8583984, Throughput: 169.190349 im/s
Epoch 34, Train Loss: 9822.8789062, Time: 29.5054s, Throughput: 169.189323 im/s
Epoch 35, Iter 39, Loss: 9634.6972656, Throughput: 169.614252 im/s
Epoch 35, Train Loss: 9814.3945312, Time: 29.4317s, Throughput: 169.613215 im/s
Epoch 36, Iter 39, Loss: 10097.4277344, Throughput: 169.145571 im/s
Epoch 36, Train Loss: 9754.3486328, Time: 29.5132s, Throughput: 169.144723 im/s
Epoch 37, Iter 39, Loss: 9855.6357422, Throughput: 169.706296 im/s
Epoch 37, Train Loss: 9706.0205078, Time: 29.4157s, Throughput: 169.705345 im/s
Epoch 38, Iter 39, Loss: 9519.5585938, Throughput: 169.035151 im/s
Epoch 38, Train Loss: 9630.2138672, Time: 29.5324s, Throughput: 169.034430 im/s
Epoch 39, Iter 39, Loss: 9630.6796875, Throughput: 169.853119 im/s
Epoch 39, Train Loss: 9688.4472656, Time: 29.3902s, Throughput: 169.852253 im/s
Epoch 40, Iter 39, Loss: 9216.2207031, Throughput: 169.287418 im/s
Epoch 40, Train Loss: 9592.3623047, Time: 29.4885s, Throughput: 169.286441 im/s
Epoch 41, Iter 39, Loss: 9603.5478516, Throughput: 168.851967 im/s
Epoch 41, Train Loss: 9637.7705078, Time: 29.5645s, Throughput: 168.851110 im/s
Epoch 42, Iter 39, Loss: 9145.6855469, Throughput: 168.769735 im/s
Epoch 42, Train Loss: 9563.0039062, Time: 29.5789s, Throughput: 168.768914 im/s
Epoch 43, Iter 39, Loss: 9820.1328125, Throughput: 169.291114 im/s
Epoch 43, Train Loss: 9479.0781250, Time: 29.4878s, Throughput: 169.290156 im/s
Epoch 44, Iter 39, Loss: 9315.2500000, Throughput: 169.790968 im/s
Epoch 44, Train Loss: 9497.7998047, Time: 29.4010s, Throughput: 169.790022 im/s
Epoch 45, Iter 39, Loss: 9861.1406250, Throughput: 170.079939 im/s
Epoch 45, Train Loss: 9569.1474609, Time: 29.3511s, Throughput: 170.079082 im/s
Epoch 46, Iter 39, Loss: 9611.9541016, Throughput: 168.964081 im/s
Epoch 46, Train Loss: 9358.2773438, Time: 29.5449s, Throughput: 168.963120 im/s
Epoch 47, Iter 39, Loss: 9624.3496094, Throughput: 168.017476 im/s
Epoch 47, Train Loss: 9364.2851562, Time: 29.7114s, Throughput: 168.016352 im/s
Epoch 48, Iter 39, Loss: 9451.9785156, Throughput: 168.679502 im/s
Epoch 48, Train Loss: 9416.3027344, Time: 29.5948s, Throughput: 168.678413 im/s
Epoch 49, Iter 39, Loss: 9360.6240234, Throughput: 166.593349 im/s
Epoch 49, Train Loss: 9386.9394531, Time: 29.9653s, Throughput: 166.592470 im/s
Epoch 50, Iter 39, Loss: 9535.1914062, Throughput: 168.971961 im/s
Epoch 50, Train Loss: 9257.0498047, Time: 29.5435s, Throughput: 168.971075 im/s

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
