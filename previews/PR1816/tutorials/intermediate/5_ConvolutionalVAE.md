---
url: /previews/PR1816/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23569.7343750, Throughput: 16.682343 im/s
Epoch 1, Train Loss: 39744.0703125, Time: 299.5005s, Throughput: 16.667750 im/s
Epoch 2, Iter 39, Loss: 18807.6777344, Throughput: 173.010129 im/s
Epoch 2, Train Loss: 20229.7695312, Time: 28.8540s, Throughput: 173.009128 im/s
Epoch 3, Iter 39, Loss: 16584.2636719, Throughput: 174.538778 im/s
Epoch 3, Train Loss: 16701.7714844, Time: 28.6023s, Throughput: 174.531681 im/s
Epoch 4, Iter 39, Loss: 14302.1269531, Throughput: 173.748375 im/s
Epoch 4, Train Loss: 15009.0126953, Time: 28.7314s, Throughput: 173.747449 im/s
Epoch 5, Iter 39, Loss: 14093.3906250, Throughput: 174.656709 im/s
Epoch 5, Train Loss: 14003.6503906, Time: 28.5819s, Throughput: 174.655787 im/s
Epoch 6, Iter 39, Loss: 12702.4375000, Throughput: 173.881477 im/s
Epoch 6, Train Loss: 13329.9570312, Time: 28.7093s, Throughput: 173.880653 im/s
Epoch 7, Iter 39, Loss: 11528.4912109, Throughput: 174.243916 im/s
Epoch 7, Train Loss: 12845.5654297, Time: 28.6497s, Throughput: 174.242887 im/s
Epoch 8, Iter 39, Loss: 12509.2080078, Throughput: 174.330960 im/s
Epoch 8, Train Loss: 12454.3115234, Time: 28.6353s, Throughput: 174.329986 im/s
Epoch 9, Iter 39, Loss: 12009.1982422, Throughput: 173.436812 im/s
Epoch 9, Train Loss: 12215.5371094, Time: 28.7830s, Throughput: 173.435859 im/s
Epoch 10, Iter 39, Loss: 11545.6279297, Throughput: 174.536379 im/s
Epoch 10, Train Loss: 11942.6582031, Time: 28.6016s, Throughput: 174.535397 im/s
Epoch 11, Iter 39, Loss: 11139.5800781, Throughput: 174.149065 im/s
Epoch 11, Train Loss: 11722.7978516, Time: 28.6653s, Throughput: 174.148117 im/s
Epoch 12, Iter 39, Loss: 12658.1386719, Throughput: 174.192472 im/s
Epoch 12, Train Loss: 11500.4931641, Time: 28.6581s, Throughput: 174.191383 im/s
Epoch 13, Iter 39, Loss: 11051.5117188, Throughput: 171.539230 im/s
Epoch 13, Train Loss: 11382.2763672, Time: 29.1014s, Throughput: 171.538199 im/s
Epoch 14, Iter 39, Loss: 11207.5976562, Throughput: 174.370335 im/s
Epoch 14, Train Loss: 11271.8281250, Time: 28.6289s, Throughput: 174.369384 im/s
Epoch 15, Iter 39, Loss: 10532.5429688, Throughput: 173.753000 im/s
Epoch 15, Train Loss: 11045.4345703, Time: 28.7306s, Throughput: 173.752063 im/s
Epoch 16, Iter 39, Loss: 11825.7050781, Throughput: 174.323571 im/s
Epoch 16, Train Loss: 10885.3105469, Time: 28.6366s, Throughput: 174.322603 im/s
Epoch 17, Iter 39, Loss: 10751.3398438, Throughput: 173.803375 im/s
Epoch 17, Train Loss: 10992.3232422, Time: 28.7223s, Throughput: 173.802461 im/s
Epoch 18, Iter 39, Loss: 10773.5878906, Throughput: 174.245753 im/s
Epoch 18, Train Loss: 10899.4765625, Time: 28.6494s, Throughput: 174.244786 im/s
Epoch 19, Iter 39, Loss: 10416.2070312, Throughput: 174.090378 im/s
Epoch 19, Train Loss: 10680.3457031, Time: 28.6749s, Throughput: 174.089274 im/s
Epoch 20, Iter 39, Loss: 10873.5820312, Throughput: 174.617078 im/s
Epoch 20, Train Loss: 10570.2255859, Time: 28.5884s, Throughput: 174.616028 im/s
Epoch 21, Iter 39, Loss: 10691.2500000, Throughput: 173.898630 im/s
Epoch 21, Train Loss: 10439.2412109, Time: 28.7065s, Throughput: 173.897655 im/s
Epoch 22, Iter 39, Loss: 10354.2343750, Throughput: 174.283699 im/s
Epoch 22, Train Loss: 10400.2246094, Time: 28.6431s, Throughput: 174.282835 im/s
Epoch 23, Iter 39, Loss: 10361.4638672, Throughput: 173.492043 im/s
Epoch 23, Train Loss: 10453.4052734, Time: 28.7738s, Throughput: 173.490982 im/s
Epoch 24, Iter 39, Loss: 9868.8125000, Throughput: 174.050014 im/s
Epoch 24, Train Loss: 10228.6406250, Time: 28.6816s, Throughput: 174.048964 im/s
Epoch 25, Iter 39, Loss: 9856.3916016, Throughput: 174.370455 im/s
Epoch 25, Train Loss: 10199.0761719, Time: 28.6289s, Throughput: 174.369536 im/s
Epoch 26, Iter 39, Loss: 10595.6601562, Throughput: 174.619094 im/s
Epoch 26, Train Loss: 10219.8691406, Time: 28.5881s, Throughput: 174.618128 im/s
Epoch 27, Iter 39, Loss: 10236.2343750, Throughput: 173.809777 im/s
Epoch 27, Train Loss: 10084.7050781, Time: 28.7212s, Throughput: 173.808657 im/s
Epoch 28, Iter 39, Loss: 9806.7343750, Throughput: 174.416691 im/s
Epoch 28, Train Loss: 10078.7119141, Time: 28.6213s, Throughput: 174.415741 im/s
Epoch 29, Iter 39, Loss: 9388.2851562, Throughput: 174.688828 im/s
Epoch 29, Train Loss: 10006.8867188, Time: 28.5767s, Throughput: 174.687770 im/s
Epoch 30, Iter 39, Loss: 9641.0849609, Throughput: 174.192799 im/s
Epoch 30, Train Loss: 10116.4951172, Time: 28.6580s, Throughput: 174.191960 im/s
Epoch 31, Iter 39, Loss: 9892.0078125, Throughput: 174.097289 im/s
Epoch 31, Train Loss: 9863.2216797, Time: 28.6738s, Throughput: 174.096141 im/s
Epoch 32, Iter 39, Loss: 9676.5087891, Throughput: 173.913074 im/s
Epoch 32, Train Loss: 9885.8076172, Time: 28.7042s, Throughput: 173.912105 im/s
Epoch 33, Iter 39, Loss: 9724.3095703, Throughput: 173.775206 im/s
Epoch 33, Train Loss: 9889.8007812, Time: 28.7269s, Throughput: 173.774201 im/s
Epoch 34, Iter 39, Loss: 9701.3046875, Throughput: 174.384923 im/s
Epoch 34, Train Loss: 9750.8623047, Time: 28.6265s, Throughput: 174.384051 im/s
Epoch 35, Iter 39, Loss: 10051.2529297, Throughput: 174.439729 im/s
Epoch 35, Train Loss: 9773.9892578, Time: 28.6175s, Throughput: 174.438698 im/s
Epoch 36, Iter 39, Loss: 10194.6738281, Throughput: 174.098327 im/s
Epoch 36, Train Loss: 9786.0820312, Time: 28.6736s, Throughput: 174.097379 im/s
Epoch 37, Iter 39, Loss: 9736.9960938, Throughput: 174.409251 im/s
Epoch 37, Train Loss: 9649.0273438, Time: 28.6225s, Throughput: 174.408240 im/s
Epoch 38, Iter 39, Loss: 10002.0000000, Throughput: 174.087423 im/s
Epoch 38, Train Loss: 9644.9443359, Time: 28.6754s, Throughput: 174.086499 im/s
Epoch 39, Iter 39, Loss: 9454.7802734, Throughput: 174.039499 im/s
Epoch 39, Train Loss: 9629.6289062, Time: 28.6833s, Throughput: 174.038401 im/s
Epoch 40, Iter 39, Loss: 9602.7753906, Throughput: 172.556803 im/s
Epoch 40, Train Loss: 9539.7685547, Time: 28.9298s, Throughput: 172.555849 im/s
Epoch 41, Iter 39, Loss: 9453.4277344, Throughput: 174.954488 im/s
Epoch 41, Train Loss: 9548.6416016, Time: 28.5333s, Throughput: 174.953629 im/s
Epoch 42, Iter 39, Loss: 9661.6015625, Throughput: 174.018957 im/s
Epoch 42, Train Loss: 9440.4462891, Time: 28.6867s, Throughput: 174.017937 im/s
Epoch 43, Iter 39, Loss: 9568.9257812, Throughput: 174.040427 im/s
Epoch 43, Train Loss: 9486.2470703, Time: 28.6832s, Throughput: 174.039262 im/s
Epoch 44, Iter 39, Loss: 9548.9648438, Throughput: 173.968348 im/s
Epoch 44, Train Loss: 9486.9394531, Time: 28.6950s, Throughput: 173.967481 im/s
Epoch 45, Iter 39, Loss: 8981.8789062, Throughput: 174.543283 im/s
Epoch 45, Train Loss: 9393.3496094, Time: 28.6006s, Throughput: 174.542062 im/s
Epoch 46, Iter 39, Loss: 9112.8261719, Throughput: 174.118651 im/s
Epoch 46, Train Loss: 9371.5937500, Time: 28.6703s, Throughput: 174.117727 im/s
Epoch 47, Iter 39, Loss: 9199.7832031, Throughput: 174.070158 im/s
Epoch 47, Train Loss: 9399.3964844, Time: 28.6783s, Throughput: 174.069077 im/s
Epoch 48, Iter 39, Loss: 9777.3496094, Throughput: 173.609221 im/s
Epoch 48, Train Loss: 9297.7431641, Time: 28.7544s, Throughput: 173.608056 im/s
Epoch 49, Iter 39, Loss: 8823.5791016, Throughput: 174.669965 im/s
Epoch 49, Train Loss: 9334.4082031, Time: 28.5798s, Throughput: 174.668987 im/s
Epoch 50, Iter 39, Loss: 9833.5312500, Throughput: 173.580330 im/s
Epoch 50, Train Loss: 9328.7373047, Time: 28.7592s, Throughput: 173.579280 im/s

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
