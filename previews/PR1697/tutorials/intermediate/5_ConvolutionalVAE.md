---
url: /previews/PR1697/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24254.2910156, Throughput: 4.157531 im/s
Epoch 1, Train Loss: 39745.3750000, Time: 1201.0012s, Throughput: 4.156532 im/s
Epoch 2, Iter 39, Loss: 17976.5351562, Throughput: 85.220204 im/s
Epoch 2, Train Loss: 20002.7363281, Time: 58.5779s, Throughput: 85.219914 im/s
Epoch 3, Iter 39, Loss: 16033.9453125, Throughput: 84.692947 im/s
Epoch 3, Train Loss: 16470.5625000, Time: 58.9425s, Throughput: 84.692657 im/s
Epoch 4, Iter 39, Loss: 13994.4707031, Throughput: 85.324352 im/s
Epoch 4, Train Loss: 14844.6699219, Time: 58.5063s, Throughput: 85.324085 im/s
Epoch 5, Iter 39, Loss: 14004.2568359, Throughput: 85.712920 im/s
Epoch 5, Train Loss: 13952.8798828, Time: 58.2411s, Throughput: 85.712602 im/s
Epoch 6, Iter 39, Loss: 12668.4746094, Throughput: 86.236115 im/s
Epoch 6, Train Loss: 13287.2109375, Time: 57.8878s, Throughput: 86.235829 im/s
Epoch 7, Iter 39, Loss: 12561.1230469, Throughput: 86.403436 im/s
Epoch 7, Train Loss: 12859.7646484, Time: 57.7757s, Throughput: 86.403140 im/s
Epoch 8, Iter 39, Loss: 11967.8769531, Throughput: 86.347233 im/s
Epoch 8, Train Loss: 12442.7968750, Time: 57.8133s, Throughput: 86.346943 im/s
Epoch 9, Iter 39, Loss: 11486.6210938, Throughput: 86.455485 im/s
Epoch 9, Train Loss: 12249.1650391, Time: 57.7409s, Throughput: 86.455222 im/s
Epoch 10, Iter 39, Loss: 11445.7812500, Throughput: 87.123993 im/s
Epoch 10, Train Loss: 11937.4697266, Time: 57.2978s, Throughput: 87.123691 im/s
Epoch 11, Iter 39, Loss: 11203.1875000, Throughput: 86.479488 im/s
Epoch 11, Train Loss: 11673.1679688, Time: 57.7249s, Throughput: 86.479194 im/s
Epoch 12, Iter 39, Loss: 12595.1113281, Throughput: 86.834681 im/s
Epoch 12, Train Loss: 11543.2636719, Time: 57.4888s, Throughput: 86.834378 im/s
Epoch 13, Iter 39, Loss: 12885.2636719, Throughput: 86.779126 im/s
Epoch 13, Train Loss: 11302.6318359, Time: 57.5255s, Throughput: 86.778836 im/s
Epoch 14, Iter 39, Loss: 11224.7861328, Throughput: 86.137388 im/s
Epoch 14, Train Loss: 11188.7431641, Time: 57.9541s, Throughput: 86.137103 im/s
Epoch 15, Iter 39, Loss: 10891.8027344, Throughput: 86.662537 im/s
Epoch 15, Train Loss: 11002.7480469, Time: 57.6030s, Throughput: 86.662213 im/s
Epoch 16, Iter 39, Loss: 11478.1425781, Throughput: 87.130989 im/s
Epoch 16, Train Loss: 10993.4482422, Time: 57.2932s, Throughput: 87.130689 im/s
Epoch 17, Iter 39, Loss: 11214.8457031, Throughput: 87.149041 im/s
Epoch 17, Train Loss: 10767.2666016, Time: 57.2814s, Throughput: 87.148734 im/s
Epoch 18, Iter 39, Loss: 10869.9804688, Throughput: 87.249876 im/s
Epoch 18, Train Loss: 10785.8144531, Time: 57.2152s, Throughput: 87.249560 im/s
Epoch 19, Iter 39, Loss: 11058.7968750, Throughput: 86.634039 im/s
Epoch 19, Train Loss: 10759.6523438, Time: 57.6219s, Throughput: 86.633753 im/s
Epoch 20, Iter 39, Loss: 10823.8740234, Throughput: 86.959585 im/s
Epoch 20, Train Loss: 10633.8593750, Time: 57.4062s, Throughput: 86.959264 im/s
Epoch 21, Iter 39, Loss: 11170.0429688, Throughput: 87.032144 im/s
Epoch 21, Train Loss: 10442.4755859, Time: 57.3583s, Throughput: 87.031842 im/s
Epoch 22, Iter 39, Loss: 9615.8886719, Throughput: 86.320985 im/s
Epoch 22, Train Loss: 10419.6718750, Time: 57.8309s, Throughput: 86.320649 im/s
Epoch 23, Iter 39, Loss: 10057.7402344, Throughput: 87.133442 im/s
Epoch 23, Train Loss: 10342.0771484, Time: 57.2917s, Throughput: 87.133102 im/s
Epoch 24, Iter 39, Loss: 9952.3603516, Throughput: 87.057582 im/s
Epoch 24, Train Loss: 10314.4394531, Time: 57.3415s, Throughput: 87.057300 im/s
Epoch 25, Iter 39, Loss: 9672.2207031, Throughput: 86.318158 im/s
Epoch 25, Train Loss: 10284.8769531, Time: 57.8328s, Throughput: 86.317851 im/s
Epoch 26, Iter 39, Loss: 9969.1855469, Throughput: 85.788400 im/s
Epoch 26, Train Loss: 10119.9277344, Time: 58.1899s, Throughput: 85.788101 im/s
Epoch 27, Iter 39, Loss: 9980.3095703, Throughput: 85.852605 im/s
Epoch 27, Train Loss: 10098.9912109, Time: 58.1464s, Throughput: 85.852269 im/s
Epoch 28, Iter 39, Loss: 10389.7441406, Throughput: 85.785226 im/s
Epoch 28, Train Loss: 10044.3369141, Time: 58.1920s, Throughput: 85.784924 im/s
Epoch 29, Iter 39, Loss: 10047.6953125, Throughput: 85.947704 im/s
Epoch 29, Train Loss: 10000.0615234, Time: 58.0820s, Throughput: 85.947436 im/s
Epoch 30, Iter 39, Loss: 9846.4472656, Throughput: 85.868023 im/s
Epoch 30, Train Loss: 10023.5234375, Time: 58.1359s, Throughput: 85.867727 im/s
Epoch 31, Iter 39, Loss: 9652.4970703, Throughput: 85.555469 im/s
Epoch 31, Train Loss: 9909.6250000, Time: 58.3483s, Throughput: 85.555151 im/s
Epoch 32, Iter 39, Loss: 10536.7753906, Throughput: 85.983132 im/s
Epoch 32, Train Loss: 9824.1826172, Time: 58.0581s, Throughput: 85.982880 im/s
Epoch 33, Iter 39, Loss: 10120.4765625, Throughput: 86.007795 im/s
Epoch 33, Train Loss: 9837.7705078, Time: 58.0415s, Throughput: 86.007497 im/s
Epoch 34, Iter 39, Loss: 9636.3574219, Throughput: 86.273641 im/s
Epoch 34, Train Loss: 9748.7744141, Time: 57.8626s, Throughput: 86.273361 im/s
Epoch 35, Iter 39, Loss: 9733.9863281, Throughput: 85.209122 im/s
Epoch 35, Train Loss: 9719.0097656, Time: 58.5855s, Throughput: 85.208831 im/s
Epoch 36, Iter 39, Loss: 9507.2304688, Throughput: 86.153803 im/s
Epoch 36, Train Loss: 9768.2382812, Time: 57.9431s, Throughput: 86.153469 im/s
Epoch 37, Iter 39, Loss: 10028.2324219, Throughput: 86.459629 im/s
Epoch 37, Train Loss: 9687.2636719, Time: 57.7381s, Throughput: 86.459356 im/s
Epoch 38, Iter 39, Loss: 9145.8945312, Throughput: 86.613266 im/s
Epoch 38, Train Loss: 9650.6035156, Time: 57.6357s, Throughput: 86.612962 im/s
Epoch 39, Iter 39, Loss: 9138.0546875, Throughput: 86.463584 im/s
Epoch 39, Train Loss: 9609.9843750, Time: 57.7355s, Throughput: 86.463307 im/s
Epoch 40, Iter 39, Loss: 9217.8691406, Throughput: 86.983116 im/s
Epoch 40, Train Loss: 9510.4873047, Time: 57.3907s, Throughput: 86.982772 im/s
Epoch 41, Iter 39, Loss: 9620.6289062, Throughput: 86.360381 im/s
Epoch 41, Train Loss: 9514.8378906, Time: 57.8045s, Throughput: 86.360066 im/s
Epoch 42, Iter 39, Loss: 9387.6943359, Throughput: 87.282358 im/s
Epoch 42, Train Loss: 9504.2255859, Time: 57.1939s, Throughput: 87.282100 im/s
Epoch 43, Iter 39, Loss: 9777.2236328, Throughput: 87.575223 im/s
Epoch 43, Train Loss: 9500.8037109, Time: 57.0026s, Throughput: 87.574925 im/s
Epoch 44, Iter 39, Loss: 9638.4541016, Throughput: 86.693662 im/s
Epoch 44, Train Loss: 9448.9306641, Time: 57.5823s, Throughput: 86.693363 im/s
Epoch 45, Iter 39, Loss: 10012.1181641, Throughput: 87.151410 im/s
Epoch 45, Train Loss: 9473.8183594, Time: 57.2798s, Throughput: 87.151135 im/s
Epoch 46, Iter 39, Loss: 9218.8496094, Throughput: 86.731771 im/s
Epoch 46, Train Loss: 9414.3193359, Time: 57.5570s, Throughput: 86.731485 im/s
Epoch 47, Iter 39, Loss: 10087.4013672, Throughput: 86.281647 im/s
Epoch 47, Train Loss: 9422.3017578, Time: 57.8572s, Throughput: 86.281381 im/s
Epoch 48, Iter 39, Loss: 9622.4365234, Throughput: 86.428351 im/s
Epoch 48, Train Loss: 9316.2353516, Time: 57.7590s, Throughput: 86.428086 im/s
Epoch 49, Iter 39, Loss: 9555.9140625, Throughput: 86.395085 im/s
Epoch 49, Train Loss: 9318.3281250, Time: 57.7813s, Throughput: 86.394797 im/s
Epoch 50, Iter 39, Loss: 9385.7744141, Throughput: 86.396110 im/s
Epoch 50, Train Loss: 9327.7919922, Time: 57.7806s, Throughput: 86.395850 im/s

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
Julia Version 1.12.6
Commit 15346901f00 (2026-04-09 19:20 UTC)
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
