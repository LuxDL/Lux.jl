---
url: /previews/PR1685/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23871.9824219, Throughput: 5.038499 im/s
Epoch 1, Train Loss: 39802.3984375, Time: 991.0750s, Throughput: 5.036955 im/s
Epoch 2, Iter 39, Loss: 18348.4042969, Throughput: 72.799176 im/s
Epoch 2, Train Loss: 20263.9746094, Time: 68.5724s, Throughput: 72.798988 im/s
Epoch 3, Iter 39, Loss: 15634.3847656, Throughput: 72.566385 im/s
Epoch 3, Train Loss: 16643.6406250, Time: 68.7924s, Throughput: 72.566182 im/s
Epoch 4, Iter 39, Loss: 14031.2548828, Throughput: 72.664956 im/s
Epoch 4, Train Loss: 14936.2421875, Time: 68.6990s, Throughput: 72.664781 im/s
Epoch 5, Iter 39, Loss: 13431.5097656, Throughput: 72.640921 im/s
Epoch 5, Train Loss: 14108.4306641, Time: 68.7218s, Throughput: 72.640722 im/s
Epoch 6, Iter 39, Loss: 13362.9189453, Throughput: 72.664076 im/s
Epoch 6, Train Loss: 13494.0576172, Time: 68.6999s, Throughput: 72.663873 im/s
Epoch 7, Iter 39, Loss: 12794.0703125, Throughput: 72.673117 im/s
Epoch 7, Train Loss: 12940.7568359, Time: 68.6913s, Throughput: 72.672949 im/s
Epoch 8, Iter 39, Loss: 12369.3457031, Throughput: 72.029706 im/s
Epoch 8, Train Loss: 12579.8925781, Time: 69.3049s, Throughput: 72.029531 im/s
Epoch 9, Iter 39, Loss: 11668.6474609, Throughput: 71.759823 im/s
Epoch 9, Train Loss: 12200.8681641, Time: 69.5656s, Throughput: 71.759651 im/s
Epoch 10, Iter 39, Loss: 11706.0722656, Throughput: 71.973902 im/s
Epoch 10, Train Loss: 11985.7294922, Time: 69.3587s, Throughput: 71.973710 im/s
Epoch 11, Iter 39, Loss: 11612.3027344, Throughput: 72.181418 im/s
Epoch 11, Train Loss: 11802.9042969, Time: 69.1592s, Throughput: 72.181251 im/s
Epoch 12, Iter 39, Loss: 11238.8222656, Throughput: 72.403108 im/s
Epoch 12, Train Loss: 11590.4316406, Time: 68.9475s, Throughput: 72.402927 im/s
Epoch 13, Iter 39, Loss: 11649.6005859, Throughput: 71.410975 im/s
Epoch 13, Train Loss: 11413.6337891, Time: 69.9054s, Throughput: 71.410827 im/s
Epoch 14, Iter 39, Loss: 11681.1035156, Throughput: 72.157356 im/s
Epoch 14, Train Loss: 11194.0556641, Time: 69.1823s, Throughput: 72.157194 im/s
Epoch 15, Iter 39, Loss: 10711.3369141, Throughput: 72.322994 im/s
Epoch 15, Train Loss: 11083.7587891, Time: 69.0239s, Throughput: 72.322780 im/s
Epoch 16, Iter 39, Loss: 11261.2792969, Throughput: 72.284212 im/s
Epoch 16, Train Loss: 10989.1044922, Time: 69.0609s, Throughput: 72.284010 im/s
Epoch 17, Iter 39, Loss: 11076.6250000, Throughput: 72.158564 im/s
Epoch 17, Train Loss: 10873.7001953, Time: 69.1811s, Throughput: 72.158386 im/s
Epoch 18, Iter 39, Loss: 11014.6396484, Throughput: 72.920492 im/s
Epoch 18, Train Loss: 10711.2949219, Time: 68.4583s, Throughput: 72.920317 im/s
Epoch 19, Iter 39, Loss: 10544.2929688, Throughput: 72.573901 im/s
Epoch 19, Train Loss: 10722.7890625, Time: 68.7852s, Throughput: 72.573723 im/s
Epoch 20, Iter 39, Loss: 11654.6005859, Throughput: 72.469201 im/s
Epoch 20, Train Loss: 10634.7089844, Time: 68.8846s, Throughput: 72.469032 im/s
Epoch 21, Iter 39, Loss: 10625.8378906, Throughput: 72.207319 im/s
Epoch 21, Train Loss: 10617.6767578, Time: 69.1344s, Throughput: 72.207163 im/s
Epoch 22, Iter 39, Loss: 10984.5537109, Throughput: 72.766474 im/s
Epoch 22, Train Loss: 10452.4882812, Time: 68.6032s, Throughput: 72.766271 im/s
Epoch 23, Iter 39, Loss: 9795.2480469, Throughput: 72.576424 im/s
Epoch 23, Train Loss: 10303.0058594, Time: 68.7828s, Throughput: 72.576262 im/s
Epoch 24, Iter 39, Loss: 10442.5117188, Throughput: 72.340302 im/s
Epoch 24, Train Loss: 10335.9208984, Time: 69.0074s, Throughput: 72.340100 im/s
Epoch 25, Iter 39, Loss: 10386.6152344, Throughput: 72.088338 im/s
Epoch 25, Train Loss: 10184.0869141, Time: 69.2485s, Throughput: 72.088160 im/s
Epoch 26, Iter 39, Loss: 10050.9970703, Throughput: 73.137113 im/s
Epoch 26, Train Loss: 10198.3017578, Time: 68.2555s, Throughput: 73.136914 im/s
Epoch 27, Iter 39, Loss: 10559.9833984, Throughput: 72.591960 im/s
Epoch 27, Train Loss: 10119.1669922, Time: 68.7681s, Throughput: 72.591762 im/s
Epoch 28, Iter 39, Loss: 9546.8203125, Throughput: 72.473467 im/s
Epoch 28, Train Loss: 10181.9277344, Time: 68.8805s, Throughput: 72.473310 im/s
Epoch 29, Iter 39, Loss: 10002.2890625, Throughput: 72.267898 im/s
Epoch 29, Train Loss: 10124.6289062, Time: 69.0765s, Throughput: 72.267700 im/s
Epoch 30, Iter 39, Loss: 9572.2353516, Throughput: 72.437398 im/s
Epoch 30, Train Loss: 10009.5253906, Time: 68.9149s, Throughput: 72.437201 im/s
Epoch 31, Iter 39, Loss: 11188.9648438, Throughput: 72.440195 im/s
Epoch 31, Train Loss: 9944.7294922, Time: 68.9122s, Throughput: 72.440016 im/s
Epoch 32, Iter 39, Loss: 10043.4062500, Throughput: 72.377383 im/s
Epoch 32, Train Loss: 9961.5468750, Time: 68.9720s, Throughput: 72.377223 im/s
Epoch 33, Iter 39, Loss: 9964.6601562, Throughput: 72.078600 im/s
Epoch 33, Train Loss: 9845.6875000, Time: 69.2579s, Throughput: 72.078429 im/s
Epoch 34, Iter 39, Loss: 9783.5263672, Throughput: 71.969378 im/s
Epoch 34, Train Loss: 9769.4765625, Time: 69.3630s, Throughput: 71.969184 im/s
Epoch 35, Iter 39, Loss: 9641.8906250, Throughput: 72.310824 im/s
Epoch 35, Train Loss: 9755.6376953, Time: 69.0355s, Throughput: 72.310661 im/s
Epoch 36, Iter 39, Loss: 9892.1855469, Throughput: 72.275652 im/s
Epoch 36, Train Loss: 9673.9716797, Time: 69.0691s, Throughput: 72.275487 im/s
Epoch 37, Iter 39, Loss: 9947.6679688, Throughput: 72.376273 im/s
Epoch 37, Train Loss: 9685.5605469, Time: 68.9731s, Throughput: 72.376092 im/s
Epoch 38, Iter 39, Loss: 10346.7822266, Throughput: 72.449043 im/s
Epoch 38, Train Loss: 9737.1757812, Time: 68.9038s, Throughput: 72.448868 im/s
Epoch 39, Iter 39, Loss: 10021.9453125, Throughput: 72.335843 im/s
Epoch 39, Train Loss: 9568.1357422, Time: 69.0116s, Throughput: 72.335682 im/s
Epoch 40, Iter 39, Loss: 10059.2060547, Throughput: 72.342436 im/s
Epoch 40, Train Loss: 9653.4472656, Time: 69.0053s, Throughput: 72.342276 im/s
Epoch 41, Iter 39, Loss: 9950.3945312, Throughput: 73.016618 im/s
Epoch 41, Train Loss: 9569.4208984, Time: 68.3682s, Throughput: 73.016452 im/s
Epoch 42, Iter 39, Loss: 9203.9091797, Throughput: 72.445058 im/s
Epoch 42, Train Loss: 9486.4931641, Time: 68.9076s, Throughput: 72.444843 im/s
Epoch 43, Iter 39, Loss: 9279.3544922, Throughput: 72.543577 im/s
Epoch 43, Train Loss: 9475.3027344, Time: 68.8140s, Throughput: 72.543366 im/s
Epoch 44, Iter 39, Loss: 9503.6376953, Throughput: 72.341632 im/s
Epoch 44, Train Loss: 9441.3710938, Time: 69.0061s, Throughput: 72.341457 im/s
Epoch 45, Iter 39, Loss: 8838.0644531, Throughput: 72.179412 im/s
Epoch 45, Train Loss: 9457.8398438, Time: 69.1612s, Throughput: 72.179215 im/s
Epoch 46, Iter 39, Loss: 9140.8339844, Throughput: 72.444645 im/s
Epoch 46, Train Loss: 9377.9589844, Time: 68.9080s, Throughput: 72.444463 im/s
Epoch 47, Iter 39, Loss: 9443.5820312, Throughput: 72.195613 im/s
Epoch 47, Train Loss: 9394.3525391, Time: 69.1457s, Throughput: 72.195417 im/s
Epoch 48, Iter 39, Loss: 9088.8896484, Throughput: 72.196972 im/s
Epoch 48, Train Loss: 9353.4755859, Time: 69.1443s, Throughput: 72.196800 im/s
Epoch 49, Iter 39, Loss: 9966.8652344, Throughput: 72.094322 im/s
Epoch 49, Train Loss: 9368.7714844, Time: 69.2428s, Throughput: 72.094129 im/s
Epoch 50, Iter 39, Loss: 9322.5761719, Throughput: 72.412005 im/s
Epoch 50, Train Loss: 9319.2031250, Time: 68.9390s, Throughput: 72.411808 im/s

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
