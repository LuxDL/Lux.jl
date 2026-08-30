---
url: /v1/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24033.4199219, Throughput: 3.968688 im/s
Epoch 1, Train Loss: 39783.6679688, Time: 1258.1358s, Throughput: 3.967775 im/s
Epoch 2, Iter 39, Loss: 17866.2343750, Throughput: 82.696187 im/s
Epoch 2, Train Loss: 20161.1953125, Time: 60.3657s, Throughput: 82.695906 im/s
Epoch 3, Iter 39, Loss: 16421.8769531, Throughput: 84.241595 im/s
Epoch 3, Train Loss: 16492.4804688, Time: 59.2584s, Throughput: 84.241283 im/s
Epoch 4, Iter 39, Loss: 13964.6171875, Throughput: 83.545357 im/s
Epoch 4, Train Loss: 14918.0703125, Time: 59.7522s, Throughput: 83.545085 im/s
Epoch 5, Iter 39, Loss: 14150.0097656, Throughput: 83.114429 im/s
Epoch 5, Train Loss: 14138.6953125, Time: 60.0620s, Throughput: 83.114144 im/s
Epoch 6, Iter 39, Loss: 13694.8515625, Throughput: 83.794336 im/s
Epoch 6, Train Loss: 13314.6250000, Time: 59.5747s, Throughput: 83.794016 im/s
Epoch 7, Iter 39, Loss: 12338.8457031, Throughput: 84.534896 im/s
Epoch 7, Train Loss: 12816.1406250, Time: 59.0527s, Throughput: 84.534603 im/s
Epoch 8, Iter 39, Loss: 12689.6621094, Throughput: 84.738794 im/s
Epoch 8, Train Loss: 12512.3857422, Time: 58.9106s, Throughput: 84.738508 im/s
Epoch 9, Iter 39, Loss: 11879.7402344, Throughput: 84.410283 im/s
Epoch 9, Train Loss: 12142.3427734, Time: 59.1399s, Throughput: 84.409985 im/s
Epoch 10, Iter 39, Loss: 11897.7636719, Throughput: 84.310030 im/s
Epoch 10, Train Loss: 11958.5195312, Time: 59.2102s, Throughput: 84.309758 im/s
Epoch 11, Iter 39, Loss: 12180.3417969, Throughput: 83.866759 im/s
Epoch 11, Train Loss: 11803.3623047, Time: 59.5232s, Throughput: 83.866495 im/s
Epoch 12, Iter 39, Loss: 11223.1933594, Throughput: 84.758522 im/s
Epoch 12, Train Loss: 11505.2500000, Time: 58.8969s, Throughput: 84.758252 im/s
Epoch 13, Iter 39, Loss: 11612.3525391, Throughput: 85.342592 im/s
Epoch 13, Train Loss: 11385.8037109, Time: 58.4938s, Throughput: 85.342333 im/s
Epoch 14, Iter 39, Loss: 11838.5947266, Throughput: 84.253007 im/s
Epoch 14, Train Loss: 11254.1748047, Time: 59.2503s, Throughput: 84.252708 im/s
Epoch 15, Iter 39, Loss: 10921.6484375, Throughput: 84.681953 im/s
Epoch 15, Train Loss: 11087.0644531, Time: 58.9502s, Throughput: 84.681687 im/s
Epoch 16, Iter 39, Loss: 10993.3300781, Throughput: 84.770938 im/s
Epoch 16, Train Loss: 10984.8271484, Time: 58.8883s, Throughput: 84.770666 im/s
Epoch 17, Iter 39, Loss: 10536.5820312, Throughput: 84.339951 im/s
Epoch 17, Train Loss: 10818.9345703, Time: 59.1892s, Throughput: 84.339666 im/s
Epoch 18, Iter 39, Loss: 10528.5644531, Throughput: 84.535117 im/s
Epoch 18, Train Loss: 10748.1738281, Time: 59.0526s, Throughput: 84.534860 im/s
Epoch 19, Iter 39, Loss: 10833.9960938, Throughput: 85.385654 im/s
Epoch 19, Train Loss: 10644.8544922, Time: 58.4644s, Throughput: 85.385354 im/s
Epoch 20, Iter 39, Loss: 10398.4951172, Throughput: 85.767083 im/s
Epoch 20, Train Loss: 10635.7568359, Time: 58.2044s, Throughput: 85.766782 im/s
Epoch 21, Iter 39, Loss: 10073.4794922, Throughput: 86.487386 im/s
Epoch 21, Train Loss: 10494.4042969, Time: 57.7196s, Throughput: 86.487098 im/s
Epoch 22, Iter 39, Loss: 10364.0820312, Throughput: 85.490929 im/s
Epoch 22, Train Loss: 10444.0791016, Time: 58.3924s, Throughput: 85.490632 im/s
Epoch 23, Iter 39, Loss: 9802.7539062, Throughput: 85.779733 im/s
Epoch 23, Train Loss: 10311.7177734, Time: 58.1958s, Throughput: 85.779404 im/s
Epoch 24, Iter 39, Loss: 10442.4433594, Throughput: 85.977925 im/s
Epoch 24, Train Loss: 10216.1767578, Time: 58.0616s, Throughput: 85.977657 im/s
Epoch 25, Iter 39, Loss: 10356.3974609, Throughput: 85.954751 im/s
Epoch 25, Train Loss: 10225.5312500, Time: 58.0773s, Throughput: 85.954437 im/s
Epoch 26, Iter 39, Loss: 10154.7597656, Throughput: 85.296532 im/s
Epoch 26, Train Loss: 10188.0351562, Time: 58.5254s, Throughput: 85.296251 im/s
Epoch 27, Iter 39, Loss: 9993.8398438, Throughput: 85.160392 im/s
Epoch 27, Train Loss: 10164.6835938, Time: 58.6190s, Throughput: 85.160141 im/s
Epoch 28, Iter 39, Loss: 10103.2714844, Throughput: 84.543930 im/s
Epoch 28, Train Loss: 10039.0605469, Time: 59.0464s, Throughput: 84.543657 im/s
Epoch 29, Iter 39, Loss: 9507.7080078, Throughput: 84.835466 im/s
Epoch 29, Train Loss: 10075.3320312, Time: 58.8435s, Throughput: 84.835178 im/s
Epoch 30, Iter 39, Loss: 10592.9746094, Throughput: 85.387231 im/s
Epoch 30, Train Loss: 9931.8935547, Time: 58.4633s, Throughput: 85.386946 im/s
Epoch 31, Iter 39, Loss: 9433.5312500, Throughput: 85.437751 im/s
Epoch 31, Train Loss: 9944.2460938, Time: 58.4287s, Throughput: 85.437452 im/s
Epoch 32, Iter 39, Loss: 9930.5849609, Throughput: 85.026354 im/s
Epoch 32, Train Loss: 9884.2265625, Time: 58.7114s, Throughput: 85.026067 im/s
Epoch 33, Iter 39, Loss: 8999.9257812, Throughput: 85.467258 im/s
Epoch 33, Train Loss: 9801.2294922, Time: 58.4085s, Throughput: 85.466964 im/s
Epoch 34, Iter 39, Loss: 9563.0107422, Throughput: 85.679159 im/s
Epoch 34, Train Loss: 9774.2304688, Time: 58.2641s, Throughput: 85.678861 im/s
Epoch 35, Iter 39, Loss: 9351.7988281, Throughput: 85.405746 im/s
Epoch 35, Train Loss: 9763.3632812, Time: 58.4506s, Throughput: 85.405467 im/s
Epoch 36, Iter 39, Loss: 9980.6699219, Throughput: 85.677409 im/s
Epoch 36, Train Loss: 9696.9863281, Time: 58.2652s, Throughput: 85.677145 im/s
Epoch 37, Iter 39, Loss: 9642.3837891, Throughput: 85.493597 im/s
Epoch 37, Train Loss: 9674.8955078, Time: 58.3906s, Throughput: 85.493270 im/s
Epoch 38, Iter 39, Loss: 9826.5722656, Throughput: 84.972125 im/s
Epoch 38, Train Loss: 9655.5312500, Time: 58.7489s, Throughput: 84.971856 im/s
Epoch 39, Iter 39, Loss: 9592.7666016, Throughput: 84.651665 im/s
Epoch 39, Train Loss: 9685.4267578, Time: 58.9713s, Throughput: 84.651368 im/s
Epoch 40, Iter 39, Loss: 10690.4326172, Throughput: 85.294159 im/s
Epoch 40, Train Loss: 9610.7226562, Time: 58.5271s, Throughput: 85.293871 im/s
Epoch 41, Iter 39, Loss: 9545.5195312, Throughput: 85.436339 im/s
Epoch 41, Train Loss: 9606.9316406, Time: 58.4296s, Throughput: 85.436082 im/s
Epoch 42, Iter 39, Loss: 10022.4296875, Throughput: 85.131522 im/s
Epoch 42, Train Loss: 9531.6611328, Time: 58.6389s, Throughput: 85.131229 im/s
Epoch 43, Iter 39, Loss: 9605.7910156, Throughput: 85.199023 im/s
Epoch 43, Train Loss: 9458.9716797, Time: 58.5924s, Throughput: 85.198747 im/s
Epoch 44, Iter 39, Loss: 9351.2783203, Throughput: 85.108748 im/s
Epoch 44, Train Loss: 9481.1601562, Time: 58.6546s, Throughput: 85.108454 im/s
Epoch 45, Iter 39, Loss: 8716.9316406, Throughput: 84.909664 im/s
Epoch 45, Train Loss: 9377.9550781, Time: 58.7921s, Throughput: 84.909386 im/s
Epoch 46, Iter 39, Loss: 10351.7861328, Throughput: 85.238331 im/s
Epoch 46, Train Loss: 9371.8535156, Time: 58.5654s, Throughput: 85.238052 im/s
Epoch 47, Iter 39, Loss: 9485.9316406, Throughput: 85.453885 im/s
Epoch 47, Train Loss: 9344.7646484, Time: 58.4177s, Throughput: 85.453602 im/s
Epoch 48, Iter 39, Loss: 9347.0546875, Throughput: 84.863636 im/s
Epoch 48, Train Loss: 9318.9980469, Time: 58.8240s, Throughput: 84.863380 im/s
Epoch 49, Iter 39, Loss: 9231.2763672, Throughput: 85.066592 im/s
Epoch 49, Train Loss: 9287.2138672, Time: 58.6837s, Throughput: 85.066256 im/s
Epoch 50, Iter 39, Loss: 9152.1484375, Throughput: 85.002344 im/s
Epoch 50, Train Loss: 9272.3076172, Time: 58.7280s, Throughput: 85.002064 im/s

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
