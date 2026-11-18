---
url: /previews/PR1855/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24010.9472656, Throughput: 15.818083 im/s
Epoch 1, Train Loss: 39792.3515625, Time: 315.8684s, Throughput: 15.804052 im/s
Epoch 2, Iter 39, Loss: 17857.8710938, Throughput: 173.655125 im/s
Epoch 2, Train Loss: 20383.9433594, Time: 28.7468s, Throughput: 173.653990 im/s
Epoch 3, Iter 39, Loss: 15288.4628906, Throughput: 173.925694 im/s
Epoch 3, Train Loss: 16633.8828125, Time: 28.7021s, Throughput: 173.924725 im/s
Epoch 4, Iter 39, Loss: 14452.1601562, Throughput: 173.955581 im/s
Epoch 4, Train Loss: 15123.4052734, Time: 28.6972s, Throughput: 173.954478 im/s
Epoch 5, Iter 39, Loss: 13526.4550781, Throughput: 173.806353 im/s
Epoch 5, Train Loss: 14100.2968750, Time: 28.7218s, Throughput: 173.805402 im/s
Epoch 6, Iter 39, Loss: 12952.8603516, Throughput: 174.307658 im/s
Epoch 6, Train Loss: 13379.9169922, Time: 28.6392s, Throughput: 174.306715 im/s
Epoch 7, Iter 39, Loss: 11969.3212891, Throughput: 173.468451 im/s
Epoch 7, Train Loss: 12973.9765625, Time: 28.7777s, Throughput: 173.467390 im/s
Epoch 8, Iter 39, Loss: 12427.0859375, Throughput: 174.166922 im/s
Epoch 8, Train Loss: 12477.4990234, Time: 28.6623s, Throughput: 174.165986 im/s
Epoch 9, Iter 39, Loss: 11860.0488281, Throughput: 173.194038 im/s
Epoch 9, Train Loss: 12200.0556641, Time: 28.8233s, Throughput: 173.192927 im/s
Epoch 10, Iter 39, Loss: 12649.6953125, Throughput: 174.044195 im/s
Epoch 10, Train Loss: 11906.5830078, Time: 28.6825s, Throughput: 174.043255 im/s
Epoch 11, Iter 39, Loss: 11896.4453125, Throughput: 173.839327 im/s
Epoch 11, Train Loss: 11810.8222656, Time: 28.7163s, Throughput: 173.838322 im/s
Epoch 12, Iter 39, Loss: 10806.6289062, Throughput: 174.157436 im/s
Epoch 12, Train Loss: 11556.2500000, Time: 28.6639s, Throughput: 174.156403 im/s
Epoch 13, Iter 39, Loss: 11923.8144531, Throughput: 173.394413 im/s
Epoch 13, Train Loss: 11432.1455078, Time: 28.7900s, Throughput: 173.393449 im/s
Epoch 14, Iter 39, Loss: 10866.5761719, Throughput: 173.950974 im/s
Epoch 14, Train Loss: 11218.1406250, Time: 28.6979s, Throughput: 173.950052 im/s
Epoch 15, Iter 39, Loss: 11896.6962891, Throughput: 173.259104 im/s
Epoch 15, Train Loss: 11042.3847656, Time: 28.8125s, Throughput: 173.258104 im/s
Epoch 16, Iter 39, Loss: 10680.6015625, Throughput: 174.472335 im/s
Epoch 16, Train Loss: 10997.5087891, Time: 28.6121s, Throughput: 174.471396 im/s
Epoch 17, Iter 39, Loss: 10630.1064453, Throughput: 173.895110 im/s
Epoch 17, Train Loss: 10901.3261719, Time: 28.7071s, Throughput: 173.894215 im/s
Epoch 18, Iter 39, Loss: 11190.2031250, Throughput: 174.298682 im/s
Epoch 18, Train Loss: 10831.1621094, Time: 28.6407s, Throughput: 174.297695 im/s
Epoch 19, Iter 39, Loss: 10549.3457031, Throughput: 171.571131 im/s
Epoch 19, Train Loss: 10688.6894531, Time: 29.0960s, Throughput: 171.570193 im/s
Epoch 20, Iter 39, Loss: 10149.1806641, Throughput: 173.731304 im/s
Epoch 20, Train Loss: 10622.1513672, Time: 28.7342s, Throughput: 173.730427 im/s
Epoch 21, Iter 39, Loss: 10788.1269531, Throughput: 173.617717 im/s
Epoch 21, Train Loss: 10546.0517578, Time: 28.7530s, Throughput: 173.616800 im/s
Epoch 22, Iter 39, Loss: 10195.4238281, Throughput: 174.274950 im/s
Epoch 22, Train Loss: 10429.5566406, Time: 28.6446s, Throughput: 174.273954 im/s
Epoch 23, Iter 39, Loss: 11098.3691406, Throughput: 173.944289 im/s
Epoch 23, Train Loss: 10378.1093750, Time: 28.6990s, Throughput: 173.943295 im/s
Epoch 24, Iter 39, Loss: 11115.6230469, Throughput: 173.707075 im/s
Epoch 24, Train Loss: 10202.3896484, Time: 28.7382s, Throughput: 173.706017 im/s
Epoch 25, Iter 39, Loss: 9678.8193359, Throughput: 173.739117 im/s
Epoch 25, Train Loss: 10178.4482422, Time: 28.7329s, Throughput: 173.738107 im/s
Epoch 26, Iter 39, Loss: 9998.6171875, Throughput: 173.771406 im/s
Epoch 26, Train Loss: 10196.4755859, Time: 28.7275s, Throughput: 173.770524 im/s
Epoch 27, Iter 39, Loss: 10584.8964844, Throughput: 173.560143 im/s
Epoch 27, Train Loss: 10135.8251953, Time: 28.7625s, Throughput: 173.559220 im/s
Epoch 28, Iter 39, Loss: 9980.8847656, Throughput: 173.570855 im/s
Epoch 28, Train Loss: 10093.0703125, Time: 28.7608s, Throughput: 173.569835 im/s
Epoch 29, Iter 39, Loss: 10130.4599609, Throughput: 174.142910 im/s
Epoch 29, Train Loss: 10047.8173828, Time: 28.6662s, Throughput: 174.142066 im/s
Epoch 30, Iter 39, Loss: 9871.1953125, Throughput: 173.315202 im/s
Epoch 30, Train Loss: 9969.7304688, Time: 28.8032s, Throughput: 173.314185 im/s
Epoch 31, Iter 39, Loss: 9728.5566406, Throughput: 173.713754 im/s
Epoch 31, Train Loss: 9932.4521484, Time: 28.7371s, Throughput: 173.712769 im/s
Epoch 32, Iter 39, Loss: 10312.6621094, Throughput: 173.839532 im/s
Epoch 32, Train Loss: 9939.7011719, Time: 28.7163s, Throughput: 173.838491 im/s
Epoch 33, Iter 39, Loss: 10338.8398438, Throughput: 174.271440 im/s
Epoch 33, Train Loss: 9871.9853516, Time: 28.6451s, Throughput: 174.270473 im/s
Epoch 34, Iter 39, Loss: 10179.4277344, Throughput: 174.057775 im/s
Epoch 34, Train Loss: 9824.6347656, Time: 28.6803s, Throughput: 174.056841 im/s
Epoch 35, Iter 39, Loss: 10033.3232422, Throughput: 174.350578 im/s
Epoch 35, Train Loss: 9783.7080078, Time: 28.6321s, Throughput: 174.349651 im/s
Epoch 36, Iter 39, Loss: 9555.6132812, Throughput: 173.598500 im/s
Epoch 36, Train Loss: 9662.3251953, Time: 28.7562s, Throughput: 173.597479 im/s
Epoch 37, Iter 39, Loss: 9577.5722656, Throughput: 174.513951 im/s
Epoch 37, Train Loss: 9688.7177734, Time: 28.6053s, Throughput: 174.512945 im/s
Epoch 38, Iter 39, Loss: 9503.0312500, Throughput: 173.262031 im/s
Epoch 38, Train Loss: 9684.0927734, Time: 28.8120s, Throughput: 173.261124 im/s
Epoch 39, Iter 39, Loss: 10005.8789062, Throughput: 173.452943 im/s
Epoch 39, Train Loss: 9569.2851562, Time: 28.7803s, Throughput: 173.452057 im/s
Epoch 40, Iter 39, Loss: 10117.7480469, Throughput: 173.459579 im/s
Epoch 40, Train Loss: 9537.3066406, Time: 28.7792s, Throughput: 173.458723 im/s
Epoch 41, Iter 39, Loss: 9401.6611328, Throughput: 173.892053 im/s
Epoch 41, Train Loss: 9510.6484375, Time: 28.7076s, Throughput: 173.891137 im/s
Epoch 42, Iter 39, Loss: 9679.4941406, Throughput: 171.692571 im/s
Epoch 42, Train Loss: 9535.6367188, Time: 29.0754s, Throughput: 171.691686 im/s
Epoch 43, Iter 39, Loss: 8989.3144531, Throughput: 173.978116 im/s
Epoch 43, Train Loss: 9460.3818359, Time: 28.6934s, Throughput: 173.977102 im/s
Epoch 44, Iter 39, Loss: 9959.0175781, Throughput: 173.630126 im/s
Epoch 44, Train Loss: 9430.2568359, Time: 28.7509s, Throughput: 173.629215 im/s
Epoch 45, Iter 39, Loss: 9229.4648438, Throughput: 174.571671 im/s
Epoch 45, Train Loss: 9496.4931641, Time: 28.5959s, Throughput: 174.570725 im/s
Epoch 46, Iter 39, Loss: 9461.5644531, Throughput: 174.084768 im/s
Epoch 46, Train Loss: 9398.3359375, Time: 28.6759s, Throughput: 174.083749 im/s
Epoch 47, Iter 39, Loss: 9414.0507812, Throughput: 174.595086 im/s
Epoch 47, Train Loss: 9367.4892578, Time: 28.5920s, Throughput: 174.594078 im/s
Epoch 48, Iter 39, Loss: 10290.9550781, Throughput: 174.091800 im/s
Epoch 48, Train Loss: 9406.5791016, Time: 28.6747s, Throughput: 174.090646 im/s
Epoch 49, Iter 39, Loss: 9606.2636719, Throughput: 174.027249 im/s
Epoch 49, Train Loss: 9339.6630859, Time: 28.6853s, Throughput: 174.026248 im/s
Epoch 50, Iter 39, Loss: 9890.5166016, Throughput: 174.095083 im/s
Epoch 50, Train Loss: 9332.0068359, Time: 28.6741s, Throughput: 174.094216 im/s

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
