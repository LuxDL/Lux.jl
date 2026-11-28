---
url: /previews/PR1866/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23958.5156250, Throughput: 15.689494 im/s
Epoch 1, Train Loss: 40087.8554688, Time: 318.4473s, Throughput: 15.676062 im/s
Epoch 2, Iter 39, Loss: 17759.6367188, Throughput: 158.074435 im/s
Epoch 2, Train Loss: 20554.2519531, Time: 31.5803s, Throughput: 158.073399 im/s
Epoch 3, Iter 39, Loss: 15607.2949219, Throughput: 158.260307 im/s
Epoch 3, Train Loss: 17113.1484375, Time: 31.5432s, Throughput: 158.259365 im/s
Epoch 4, Iter 39, Loss: 14564.1806641, Throughput: 157.687020 im/s
Epoch 4, Train Loss: 15495.0722656, Time: 31.6578s, Throughput: 157.686069 im/s
Epoch 5, Iter 39, Loss: 14061.7988281, Throughput: 158.304869 im/s
Epoch 5, Train Loss: 14478.2294922, Time: 31.5342s, Throughput: 158.304107 im/s
Epoch 6, Iter 39, Loss: 13751.4882812, Throughput: 157.967395 im/s
Epoch 6, Train Loss: 13811.9697266, Time: 31.6016s, Throughput: 157.966484 im/s
Epoch 7, Iter 39, Loss: 13215.5039062, Throughput: 158.377820 im/s
Epoch 7, Train Loss: 13231.8554688, Time: 31.5197s, Throughput: 158.377016 im/s
Epoch 8, Iter 39, Loss: 12793.7363281, Throughput: 158.342098 im/s
Epoch 8, Train Loss: 12923.7988281, Time: 31.5269s, Throughput: 158.341219 im/s
Epoch 9, Iter 39, Loss: 12146.9550781, Throughput: 158.555931 im/s
Epoch 9, Train Loss: 12573.3945312, Time: 31.4843s, Throughput: 158.555080 im/s
Epoch 10, Iter 39, Loss: 11841.2236328, Throughput: 158.003649 im/s
Epoch 10, Train Loss: 12476.0078125, Time: 31.5944s, Throughput: 158.002780 im/s
Epoch 11, Iter 39, Loss: 11722.0175781, Throughput: 157.482138 im/s
Epoch 11, Train Loss: 12149.3730469, Time: 31.6990s, Throughput: 157.481359 im/s
Epoch 12, Iter 39, Loss: 11439.2861328, Throughput: 156.221053 im/s
Epoch 12, Train Loss: 11904.8017578, Time: 31.9551s, Throughput: 156.219420 im/s
Epoch 13, Iter 39, Loss: 11488.9892578, Throughput: 158.487400 im/s
Epoch 13, Train Loss: 11707.6572266, Time: 31.4979s, Throughput: 158.486651 im/s
Epoch 14, Iter 39, Loss: 11183.9082031, Throughput: 158.166290 im/s
Epoch 14, Train Loss: 11663.3017578, Time: 31.5619s, Throughput: 158.165363 im/s
Epoch 15, Iter 39, Loss: 12159.0273438, Throughput: 158.292095 im/s
Epoch 15, Train Loss: 11386.9599609, Time: 31.5368s, Throughput: 158.291146 im/s
Epoch 16, Iter 39, Loss: 12295.5507812, Throughput: 158.258482 im/s
Epoch 16, Train Loss: 11361.4101562, Time: 31.5435s, Throughput: 158.257560 im/s
Epoch 17, Iter 39, Loss: 10816.4296875, Throughput: 158.882208 im/s
Epoch 17, Train Loss: 11338.8437500, Time: 31.4197s, Throughput: 158.881434 im/s
Epoch 18, Iter 39, Loss: 11357.3837891, Throughput: 157.830634 im/s
Epoch 18, Train Loss: 11222.0507812, Time: 31.6290s, Throughput: 157.829690 im/s
Epoch 19, Iter 39, Loss: 11330.4902344, Throughput: 158.111855 im/s
Epoch 19, Train Loss: 11019.0957031, Time: 31.5728s, Throughput: 158.110935 im/s
Epoch 20, Iter 39, Loss: 10839.1445312, Throughput: 158.306922 im/s
Epoch 20, Train Loss: 10924.4619141, Time: 31.5339s, Throughput: 158.305963 im/s
Epoch 21, Iter 39, Loss: 10677.6679688, Throughput: 157.998693 im/s
Epoch 21, Train Loss: 10861.4169922, Time: 31.5954s, Throughput: 157.997767 im/s
Epoch 22, Iter 39, Loss: 10491.6015625, Throughput: 158.208006 im/s
Epoch 22, Train Loss: 10757.0468750, Time: 31.5536s, Throughput: 158.207138 im/s
Epoch 23, Iter 39, Loss: 10881.7841797, Throughput: 157.834816 im/s
Epoch 23, Train Loss: 10716.7880859, Time: 31.6282s, Throughput: 157.833907 im/s
Epoch 24, Iter 39, Loss: 11447.7373047, Throughput: 159.231784 im/s
Epoch 24, Train Loss: 10668.6767578, Time: 31.3507s, Throughput: 159.230782 im/s
Epoch 25, Iter 39, Loss: 9662.0585938, Throughput: 158.282111 im/s
Epoch 25, Train Loss: 10575.0478516, Time: 31.5388s, Throughput: 158.281239 im/s
Epoch 26, Iter 39, Loss: 10947.8515625, Throughput: 158.047426 im/s
Epoch 26, Train Loss: 10468.1582031, Time: 31.5856s, Throughput: 158.046540 im/s
Epoch 27, Iter 39, Loss: 10546.2128906, Throughput: 157.874013 im/s
Epoch 27, Train Loss: 10439.6074219, Time: 31.6203s, Throughput: 157.873194 im/s
Epoch 28, Iter 39, Loss: 10092.7861328, Throughput: 158.081498 im/s
Epoch 28, Train Loss: 10363.2333984, Time: 31.5788s, Throughput: 158.080577 im/s
Epoch 29, Iter 39, Loss: 10310.4121094, Throughput: 158.005780 im/s
Epoch 29, Train Loss: 10317.8847656, Time: 31.5940s, Throughput: 158.004934 im/s
Epoch 30, Iter 39, Loss: 10683.6289062, Throughput: 157.204293 im/s
Epoch 30, Train Loss: 10355.0498047, Time: 31.7550s, Throughput: 157.203451 im/s
Epoch 31, Iter 39, Loss: 10212.8740234, Throughput: 157.763849 im/s
Epoch 31, Train Loss: 10228.6582031, Time: 31.6424s, Throughput: 157.763106 im/s
Epoch 32, Iter 39, Loss: 10174.0976562, Throughput: 158.346728 im/s
Epoch 32, Train Loss: 10255.0488281, Time: 31.5259s, Throughput: 158.345779 im/s
Epoch 33, Iter 39, Loss: 9919.4980469, Throughput: 157.617694 im/s
Epoch 33, Train Loss: 10116.7480469, Time: 31.6717s, Throughput: 157.616903 im/s
Epoch 34, Iter 39, Loss: 9888.2929688, Throughput: 158.283502 im/s
Epoch 34, Train Loss: 10040.2773438, Time: 31.5385s, Throughput: 158.282523 im/s
Epoch 35, Iter 39, Loss: 9432.1728516, Throughput: 157.990593 im/s
Epoch 35, Train Loss: 10034.8369141, Time: 31.5970s, Throughput: 157.989798 im/s
Epoch 36, Iter 39, Loss: 10194.1328125, Throughput: 158.690922 im/s
Epoch 36, Train Loss: 9999.0917969, Time: 31.4576s, Throughput: 158.689984 im/s
Epoch 37, Iter 39, Loss: 9727.5351562, Throughput: 158.117299 im/s
Epoch 37, Train Loss: 9946.5732422, Time: 31.5717s, Throughput: 158.116422 im/s
Epoch 38, Iter 39, Loss: 10088.8710938, Throughput: 157.713986 im/s
Epoch 38, Train Loss: 10027.5361328, Time: 31.6524s, Throughput: 157.713189 im/s
Epoch 39, Iter 39, Loss: 10296.4248047, Throughput: 158.763993 im/s
Epoch 39, Train Loss: 9896.5283203, Time: 31.4431s, Throughput: 158.763063 im/s
Epoch 40, Iter 39, Loss: 9624.3964844, Throughput: 158.006940 im/s
Epoch 40, Train Loss: 9848.6669922, Time: 31.5937s, Throughput: 158.006094 im/s
Epoch 41, Iter 39, Loss: 10045.0644531, Throughput: 159.226567 im/s
Epoch 41, Train Loss: 9847.5332031, Time: 31.3517s, Throughput: 159.225694 im/s
Epoch 42, Iter 39, Loss: 11032.3388672, Throughput: 158.042522 im/s
Epoch 42, Train Loss: 9797.4619141, Time: 31.5866s, Throughput: 158.041756 im/s
Epoch 43, Iter 39, Loss: 9541.7714844, Throughput: 159.314721 im/s
Epoch 43, Train Loss: 9754.0996094, Time: 31.3344s, Throughput: 159.313974 im/s
Epoch 44, Iter 39, Loss: 9395.7734375, Throughput: 158.413539 im/s
Epoch 44, Train Loss: 9833.6914062, Time: 31.5126s, Throughput: 158.412670 im/s
Epoch 45, Iter 39, Loss: 9409.2197266, Throughput: 158.187176 im/s
Epoch 45, Train Loss: 9713.1669922, Time: 31.5577s, Throughput: 158.186324 im/s
Epoch 46, Iter 39, Loss: 9442.8603516, Throughput: 158.387000 im/s
Epoch 46, Train Loss: 9699.9394531, Time: 31.5179s, Throughput: 158.386046 im/s
Epoch 47, Iter 39, Loss: 9344.1542969, Throughput: 158.710008 im/s
Epoch 47, Train Loss: 9743.2490234, Time: 31.4537s, Throughput: 158.709256 im/s
Epoch 48, Iter 39, Loss: 9498.6689453, Throughput: 158.162563 im/s
Epoch 48, Train Loss: 9634.8720703, Time: 31.5627s, Throughput: 158.161540 im/s
Epoch 49, Iter 39, Loss: 9703.4345703, Throughput: 157.158211 im/s
Epoch 49, Train Loss: 9563.3437500, Time: 31.7644s, Throughput: 157.157266 im/s
Epoch 50, Iter 39, Loss: 10065.6513672, Throughput: 158.151372 im/s
Epoch 50, Train Loss: 9587.8652344, Time: 31.5649s, Throughput: 158.150525 im/s

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
