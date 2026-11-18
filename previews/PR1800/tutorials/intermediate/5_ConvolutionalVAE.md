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
Epoch 1, Iter 39, Loss: 23695.3320312, Throughput: 15.728362 im/s
Epoch 1, Train Loss: 39331.6992188, Time: 317.6534s, Throughput: 15.715240 im/s
Epoch 2, Iter 39, Loss: 16940.0195312, Throughput: 157.210545 im/s
Epoch 2, Train Loss: 20084.5898438, Time: 31.7538s, Throughput: 157.209584 im/s
Epoch 3, Iter 39, Loss: 15562.9003906, Throughput: 157.010920 im/s
Epoch 3, Train Loss: 16566.3925781, Time: 31.7941s, Throughput: 157.010017 im/s
Epoch 4, Iter 39, Loss: 14330.7177734, Throughput: 156.420399 im/s
Epoch 4, Train Loss: 14954.9980469, Time: 31.9142s, Throughput: 156.419546 im/s
Epoch 5, Iter 39, Loss: 13268.0195312, Throughput: 156.713296 im/s
Epoch 5, Train Loss: 14118.4443359, Time: 31.8545s, Throughput: 156.712459 im/s
Epoch 6, Iter 39, Loss: 12909.6054688, Throughput: 156.346792 im/s
Epoch 6, Train Loss: 13396.6982422, Time: 31.9292s, Throughput: 156.345755 im/s
Epoch 7, Iter 39, Loss: 13102.1699219, Throughput: 156.641997 im/s
Epoch 7, Train Loss: 12930.4052734, Time: 31.8690s, Throughput: 156.641161 im/s
Epoch 8, Iter 39, Loss: 12695.7851562, Throughput: 157.752900 im/s
Epoch 8, Train Loss: 12602.7441406, Time: 31.6446s, Throughput: 157.751994 im/s
Epoch 9, Iter 39, Loss: 12191.8544922, Throughput: 156.413651 im/s
Epoch 9, Train Loss: 12285.5488281, Time: 31.9156s, Throughput: 156.412705 im/s
Epoch 10, Iter 39, Loss: 11482.2333984, Throughput: 154.744941 im/s
Epoch 10, Train Loss: 11925.1083984, Time: 32.2597s, Throughput: 154.744083 im/s
Epoch 11, Iter 39, Loss: 11481.3300781, Throughput: 157.649910 im/s
Epoch 11, Train Loss: 11753.5898438, Time: 31.6653s, Throughput: 157.648969 im/s
Epoch 12, Iter 39, Loss: 11611.3339844, Throughput: 156.937896 im/s
Epoch 12, Train Loss: 11639.4755859, Time: 31.8090s, Throughput: 156.936810 im/s
Epoch 13, Iter 39, Loss: 11405.1171875, Throughput: 157.559982 im/s
Epoch 13, Train Loss: 11393.5205078, Time: 31.6834s, Throughput: 157.558942 im/s
Epoch 14, Iter 39, Loss: 11054.0332031, Throughput: 157.211452 im/s
Epoch 14, Train Loss: 11243.3769531, Time: 31.7536s, Throughput: 157.210536 im/s
Epoch 15, Iter 39, Loss: 10712.4941406, Throughput: 156.993410 im/s
Epoch 15, Train Loss: 11196.9843750, Time: 31.7977s, Throughput: 156.992428 im/s
Epoch 16, Iter 39, Loss: 10949.0878906, Throughput: 156.870514 im/s
Epoch 16, Train Loss: 10987.6738281, Time: 31.8226s, Throughput: 156.869528 im/s
Epoch 17, Iter 39, Loss: 10440.1953125, Throughput: 157.018570 im/s
Epoch 17, Train Loss: 10954.6582031, Time: 31.7926s, Throughput: 157.017715 im/s
Epoch 18, Iter 39, Loss: 10528.6826172, Throughput: 157.182741 im/s
Epoch 18, Train Loss: 10862.2880859, Time: 31.7594s, Throughput: 157.181821 im/s
Epoch 19, Iter 39, Loss: 10519.9863281, Throughput: 155.543262 im/s
Epoch 19, Train Loss: 10709.2333984, Time: 32.0942s, Throughput: 155.542350 im/s
Epoch 20, Iter 39, Loss: 10835.5888672, Throughput: 156.646573 im/s
Epoch 20, Train Loss: 10768.0410156, Time: 31.8681s, Throughput: 156.645664 im/s
Epoch 21, Iter 39, Loss: 11334.7089844, Throughput: 156.806948 im/s
Epoch 21, Train Loss: 10623.7568359, Time: 31.8355s, Throughput: 156.805869 im/s
Epoch 22, Iter 39, Loss: 10369.4511719, Throughput: 157.129604 im/s
Epoch 22, Train Loss: 10492.1796875, Time: 31.7701s, Throughput: 157.128784 im/s
Epoch 23, Iter 39, Loss: 11089.7382812, Throughput: 156.567611 im/s
Epoch 23, Train Loss: 10393.0566406, Time: 31.8842s, Throughput: 156.566756 im/s
Epoch 24, Iter 39, Loss: 10381.6337891, Throughput: 156.936415 im/s
Epoch 24, Train Loss: 10366.9833984, Time: 31.8092s, Throughput: 156.935527 im/s
Epoch 25, Iter 39, Loss: 10449.8886719, Throughput: 156.723420 im/s
Epoch 25, Train Loss: 10248.6708984, Time: 31.8525s, Throughput: 156.722505 im/s
Epoch 26, Iter 39, Loss: 10526.2792969, Throughput: 156.596589 im/s
Epoch 26, Train Loss: 10265.4951172, Time: 31.8783s, Throughput: 156.595691 im/s
Epoch 27, Iter 39, Loss: 10297.7968750, Throughput: 156.715740 im/s
Epoch 27, Train Loss: 10215.9707031, Time: 31.8540s, Throughput: 156.714899 im/s
Epoch 28, Iter 39, Loss: 9995.4179688, Throughput: 156.723637 im/s
Epoch 28, Train Loss: 10240.4199219, Time: 31.8524s, Throughput: 156.722762 im/s
Epoch 29, Iter 39, Loss: 9513.2695312, Throughput: 157.083607 im/s
Epoch 29, Train Loss: 10168.2548828, Time: 31.7794s, Throughput: 157.082727 im/s
Epoch 30, Iter 39, Loss: 10907.0312500, Throughput: 155.395822 im/s
Epoch 30, Train Loss: 9999.4531250, Time: 32.1246s, Throughput: 155.394719 im/s
Epoch 31, Iter 39, Loss: 10052.8457031, Throughput: 156.932484 im/s
Epoch 31, Train Loss: 10032.7617188, Time: 31.8101s, Throughput: 156.931482 im/s
Epoch 32, Iter 39, Loss: 10311.3261719, Throughput: 156.717265 im/s
Epoch 32, Train Loss: 10013.6542969, Time: 31.8537s, Throughput: 156.716281 im/s
Epoch 33, Iter 39, Loss: 9835.6630859, Throughput: 156.969903 im/s
Epoch 33, Train Loss: 9937.3085938, Time: 31.8025s, Throughput: 156.969009 im/s
Epoch 34, Iter 39, Loss: 9186.5214844, Throughput: 157.201164 im/s
Epoch 34, Train Loss: 9866.1289062, Time: 31.7557s, Throughput: 157.200333 im/s
Epoch 35, Iter 39, Loss: 10065.2167969, Throughput: 156.674291 im/s
Epoch 35, Train Loss: 9803.9404297, Time: 31.8625s, Throughput: 156.673401 im/s
Epoch 36, Iter 39, Loss: 9895.4355469, Throughput: 156.763763 im/s
Epoch 36, Train Loss: 9829.0117188, Time: 31.8444s, Throughput: 156.762497 im/s
Epoch 37, Iter 39, Loss: 10069.1484375, Throughput: 156.971516 im/s
Epoch 37, Train Loss: 9801.6845703, Time: 31.8021s, Throughput: 156.970686 im/s
Epoch 38, Iter 39, Loss: 9253.6972656, Throughput: 156.533589 im/s
Epoch 38, Train Loss: 9709.2890625, Time: 31.8911s, Throughput: 156.532808 im/s
Epoch 39, Iter 39, Loss: 10155.3007812, Throughput: 156.298599 im/s
Epoch 39, Train Loss: 9660.4062500, Time: 31.9390s, Throughput: 156.297718 im/s
Epoch 40, Iter 39, Loss: 9575.4130859, Throughput: 155.987673 im/s
Epoch 40, Train Loss: 9637.0126953, Time: 32.0027s, Throughput: 155.986746 im/s
Epoch 41, Iter 39, Loss: 10359.1933594, Throughput: 156.338673 im/s
Epoch 41, Train Loss: 9626.5800781, Time: 31.9309s, Throughput: 156.337743 im/s
Epoch 42, Iter 39, Loss: 10219.2880859, Throughput: 156.594501 im/s
Epoch 42, Train Loss: 9615.6298828, Time: 31.8787s, Throughput: 156.593563 im/s
Epoch 43, Iter 39, Loss: 9928.3417969, Throughput: 156.515258 im/s
Epoch 43, Train Loss: 9626.7031250, Time: 31.8949s, Throughput: 156.514173 im/s
Epoch 44, Iter 39, Loss: 10265.2978516, Throughput: 157.038071 im/s
Epoch 44, Train Loss: 9505.4521484, Time: 31.7887s, Throughput: 157.037152 im/s
Epoch 45, Iter 39, Loss: 9914.9169922, Throughput: 156.594943 im/s
Epoch 45, Train Loss: 9461.1494141, Time: 31.8786s, Throughput: 156.594044 im/s
Epoch 46, Iter 39, Loss: 9484.2666016, Throughput: 157.137915 im/s
Epoch 46, Train Loss: 9473.0615234, Time: 31.7685s, Throughput: 157.136876 im/s
Epoch 47, Iter 39, Loss: 10018.3691406, Throughput: 156.452966 im/s
Epoch 47, Train Loss: 9470.3876953, Time: 31.9076s, Throughput: 156.451999 im/s
Epoch 48, Iter 39, Loss: 8590.9531250, Throughput: 156.215006 im/s
Epoch 48, Train Loss: 9452.4472656, Time: 31.9562s, Throughput: 156.213877 im/s
Epoch 49, Iter 39, Loss: 9476.7753906, Throughput: 156.703595 im/s
Epoch 49, Train Loss: 9351.3115234, Time: 31.8565s, Throughput: 156.702562 im/s
Epoch 50, Iter 39, Loss: 9305.5859375, Throughput: 154.618116 im/s
Epoch 50, Train Loss: 9359.5634766, Time: 32.2862s, Throughput: 154.617229 im/s

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
