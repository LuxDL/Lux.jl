---
url: /previews/PR1818/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23075.8261719, Throughput: 24.009121 im/s
Epoch 1, Train Loss: 39511.6132812, Time: 208.0785s, Throughput: 23.990948 im/s
Epoch 2, Iter 39, Loss: 17620.0195312, Throughput: 128.446166 im/s
Epoch 2, Train Loss: 20174.3339844, Time: 38.8647s, Throughput: 128.445661 im/s
Epoch 3, Iter 39, Loss: 15795.6074219, Throughput: 129.729751 im/s
Epoch 3, Train Loss: 16579.4726562, Time: 38.4802s, Throughput: 129.729204 im/s
Epoch 4, Iter 39, Loss: 14426.9140625, Throughput: 128.765401 im/s
Epoch 4, Train Loss: 15151.9824219, Time: 38.7683s, Throughput: 128.764860 im/s
Epoch 5, Iter 39, Loss: 13409.5097656, Throughput: 128.136535 im/s
Epoch 5, Train Loss: 14161.7832031, Time: 38.9586s, Throughput: 128.135969 im/s
Epoch 6, Iter 39, Loss: 13897.3183594, Throughput: 127.352810 im/s
Epoch 6, Train Loss: 13548.3974609, Time: 39.1983s, Throughput: 127.352309 im/s
Epoch 7, Iter 39, Loss: 13831.0351562, Throughput: 128.097850 im/s
Epoch 7, Train Loss: 13076.2900391, Time: 38.9704s, Throughput: 128.097206 im/s
Epoch 8, Iter 39, Loss: 12811.0644531, Throughput: 127.794958 im/s
Epoch 8, Train Loss: 12687.8300781, Time: 39.0628s, Throughput: 127.794353 im/s
Epoch 9, Iter 39, Loss: 12236.5244141, Throughput: 128.747567 im/s
Epoch 9, Train Loss: 12268.3769531, Time: 38.7737s, Throughput: 128.747100 im/s
Epoch 10, Iter 39, Loss: 11390.8535156, Throughput: 129.048786 im/s
Epoch 10, Train Loss: 12019.1884766, Time: 38.6832s, Throughput: 129.048166 im/s
Epoch 11, Iter 39, Loss: 12167.8183594, Throughput: 127.730786 im/s
Epoch 11, Train Loss: 11985.4902344, Time: 39.0824s, Throughput: 127.730276 im/s
Epoch 12, Iter 39, Loss: 11638.8447266, Throughput: 128.882298 im/s
Epoch 12, Train Loss: 11791.1171875, Time: 38.7332s, Throughput: 128.881819 im/s
Epoch 13, Iter 39, Loss: 11447.2187500, Throughput: 129.377700 im/s
Epoch 13, Train Loss: 11552.8017578, Time: 38.5849s, Throughput: 129.377100 im/s
Epoch 14, Iter 39, Loss: 10839.1601562, Throughput: 129.404594 im/s
Epoch 14, Train Loss: 11325.3125000, Time: 38.5768s, Throughput: 129.404044 im/s
Epoch 15, Iter 39, Loss: 11212.6708984, Throughput: 129.343521 im/s
Epoch 15, Train Loss: 11245.5664062, Time: 38.5950s, Throughput: 129.343022 im/s
Epoch 16, Iter 39, Loss: 11447.3623047, Throughput: 129.033432 im/s
Epoch 16, Train Loss: 11120.5380859, Time: 38.6878s, Throughput: 129.032804 im/s
Epoch 17, Iter 39, Loss: 10939.2988281, Throughput: 129.170373 im/s
Epoch 17, Train Loss: 10958.0585938, Time: 38.6468s, Throughput: 129.169831 im/s
Epoch 18, Iter 39, Loss: 11680.7138672, Throughput: 129.553987 im/s
Epoch 18, Train Loss: 10941.2578125, Time: 38.5324s, Throughput: 129.553449 im/s
Epoch 19, Iter 39, Loss: 10018.9482422, Throughput: 129.022504 im/s
Epoch 19, Train Loss: 10845.3681641, Time: 38.6911s, Throughput: 129.021893 im/s
Epoch 20, Iter 39, Loss: 10424.8417969, Throughput: 128.165745 im/s
Epoch 20, Train Loss: 10818.9541016, Time: 38.9498s, Throughput: 128.165100 im/s
Epoch 21, Iter 39, Loss: 10681.3437500, Throughput: 128.082650 im/s
Epoch 21, Train Loss: 10736.8222656, Time: 38.9750s, Throughput: 128.082025 im/s
Epoch 22, Iter 39, Loss: 10255.5234375, Throughput: 124.864489 im/s
Epoch 22, Train Loss: 10627.6953125, Time: 39.9796s, Throughput: 124.863821 im/s
Epoch 23, Iter 39, Loss: 9756.5605469, Throughput: 120.748966 im/s
Epoch 23, Train Loss: 10529.5966797, Time: 41.3422s, Throughput: 120.748372 im/s
Epoch 24, Iter 39, Loss: 10882.3632812, Throughput: 120.501062 im/s
Epoch 24, Train Loss: 10430.9843750, Time: 41.4272s, Throughput: 120.500561 im/s
Epoch 25, Iter 39, Loss: 10583.8701172, Throughput: 120.414939 im/s
Epoch 25, Train Loss: 10457.0263672, Time: 41.4568s, Throughput: 120.414478 im/s
Epoch 26, Iter 39, Loss: 11175.3417969, Throughput: 120.092047 im/s
Epoch 26, Train Loss: 10355.6210938, Time: 41.5683s, Throughput: 120.091518 im/s
Epoch 27, Iter 39, Loss: 9653.6103516, Throughput: 122.624290 im/s
Epoch 27, Train Loss: 10255.5292969, Time: 40.7099s, Throughput: 122.623853 im/s
Epoch 28, Iter 39, Loss: 10356.3378906, Throughput: 128.601273 im/s
Epoch 28, Train Loss: 10227.4375000, Time: 38.8178s, Throughput: 128.600754 im/s
Epoch 29, Iter 39, Loss: 10321.9228516, Throughput: 126.886132 im/s
Epoch 29, Train Loss: 10220.5419922, Time: 39.3426s, Throughput: 126.885484 im/s
Epoch 30, Iter 39, Loss: 10355.4492188, Throughput: 129.574897 im/s
Epoch 30, Train Loss: 10141.0371094, Time: 38.5261s, Throughput: 129.574368 im/s
Epoch 31, Iter 39, Loss: 9919.6728516, Throughput: 130.297843 im/s
Epoch 31, Train Loss: 10089.4082031, Time: 38.3124s, Throughput: 130.297291 im/s
Epoch 32, Iter 39, Loss: 9653.0761719, Throughput: 130.133061 im/s
Epoch 32, Train Loss: 10062.1025391, Time: 38.3609s, Throughput: 130.132423 im/s
Epoch 33, Iter 39, Loss: 10295.9316406, Throughput: 129.754278 im/s
Epoch 33, Train Loss: 9976.6591797, Time: 38.4729s, Throughput: 129.753739 im/s
Epoch 34, Iter 39, Loss: 10666.7939453, Throughput: 130.149841 im/s
Epoch 34, Train Loss: 9946.7988281, Time: 38.3560s, Throughput: 130.149298 im/s
Epoch 35, Iter 39, Loss: 9868.8457031, Throughput: 129.346802 im/s
Epoch 35, Train Loss: 9901.2714844, Time: 38.5941s, Throughput: 129.346219 im/s
Epoch 36, Iter 39, Loss: 9844.1064453, Throughput: 130.137315 im/s
Epoch 36, Train Loss: 9826.8076172, Time: 38.3596s, Throughput: 130.136799 im/s
Epoch 37, Iter 39, Loss: 9328.1337891, Throughput: 129.423788 im/s
Epoch 37, Train Loss: 9826.1855469, Time: 38.5711s, Throughput: 129.423258 im/s
Epoch 38, Iter 39, Loss: 9313.9199219, Throughput: 129.962051 im/s
Epoch 38, Train Loss: 9728.5019531, Time: 38.4114s, Throughput: 129.961533 im/s
Epoch 39, Iter 39, Loss: 10000.6621094, Throughput: 130.104246 im/s
Epoch 39, Train Loss: 9811.0009766, Time: 38.3694s, Throughput: 130.103724 im/s
Epoch 40, Iter 39, Loss: 10107.0126953, Throughput: 130.324576 im/s
Epoch 40, Train Loss: 9723.5253906, Time: 38.3045s, Throughput: 130.324052 im/s
Epoch 41, Iter 39, Loss: 9627.0927734, Throughput: 130.064195 im/s
Epoch 41, Train Loss: 9663.5869141, Time: 38.3812s, Throughput: 130.063612 im/s
Epoch 42, Iter 39, Loss: 9784.9003906, Throughput: 130.128777 im/s
Epoch 42, Train Loss: 9621.4482422, Time: 38.3622s, Throughput: 130.128261 im/s
Epoch 43, Iter 39, Loss: 9568.9316406, Throughput: 130.149512 im/s
Epoch 43, Train Loss: 9621.8339844, Time: 38.3560s, Throughput: 130.148969 im/s
Epoch 44, Iter 39, Loss: 9646.5410156, Throughput: 129.228700 im/s
Epoch 44, Train Loss: 9642.2529297, Time: 38.6293s, Throughput: 129.228168 im/s
Epoch 45, Iter 39, Loss: 10045.7226562, Throughput: 129.443467 im/s
Epoch 45, Train Loss: 9609.3447266, Time: 38.5653s, Throughput: 129.442920 im/s
Epoch 46, Iter 39, Loss: 9514.6162109, Throughput: 129.352154 im/s
Epoch 46, Train Loss: 9531.0615234, Time: 38.5925s, Throughput: 129.351621 im/s
Epoch 47, Iter 39, Loss: 9275.0332031, Throughput: 129.425553 im/s
Epoch 47, Train Loss: 9467.7763672, Time: 38.5706s, Throughput: 129.424962 im/s
Epoch 48, Iter 39, Loss: 10190.7070312, Throughput: 129.160941 im/s
Epoch 48, Train Loss: 9548.8955078, Time: 38.6496s, Throughput: 129.160320 im/s
Epoch 49, Iter 39, Loss: 9078.2265625, Throughput: 129.894129 im/s
Epoch 49, Train Loss: 9518.3837891, Time: 38.4314s, Throughput: 129.893653 im/s
Epoch 50, Iter 39, Loss: 9066.8056641, Throughput: 129.918406 im/s
Epoch 50, Train Loss: 9445.8017578, Time: 38.4243s, Throughput: 129.917834 im/s

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
  CPU: 4 × AMD EPYC 9V45 96-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver5)
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
