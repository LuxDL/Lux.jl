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
Epoch 1, Iter 39, Loss: 24229.5136719, Throughput: 12.958160 im/s
Epoch 1, Train Loss: 39572.2812500, Time: 385.4965s, Throughput: 12.949534 im/s
Epoch 2, Iter 39, Loss: 17256.6406250, Throughput: 99.101885 im/s
Epoch 2, Train Loss: 20099.3027344, Time: 50.3726s, Throughput: 99.101494 im/s
Epoch 3, Iter 39, Loss: 14777.5371094, Throughput: 98.839697 im/s
Epoch 3, Train Loss: 16503.5781250, Time: 50.5062s, Throughput: 98.839278 im/s
Epoch 4, Iter 39, Loss: 14410.7871094, Throughput: 98.605164 im/s
Epoch 4, Train Loss: 15009.2470703, Time: 50.6264s, Throughput: 98.604747 im/s
Epoch 5, Iter 39, Loss: 14552.4033203, Throughput: 98.483663 im/s
Epoch 5, Train Loss: 14081.3417969, Time: 50.6888s, Throughput: 98.483224 im/s
Epoch 6, Iter 39, Loss: 12536.7441406, Throughput: 98.610198 im/s
Epoch 6, Train Loss: 13290.9619141, Time: 50.6238s, Throughput: 98.609649 im/s
Epoch 7, Iter 39, Loss: 12301.2617188, Throughput: 98.715355 im/s
Epoch 7, Train Loss: 12828.6943359, Time: 50.5699s, Throughput: 98.714942 im/s
Epoch 8, Iter 39, Loss: 12263.0712891, Throughput: 99.201238 im/s
Epoch 8, Train Loss: 12491.3134766, Time: 50.3222s, Throughput: 99.200817 im/s
Epoch 9, Iter 39, Loss: 12219.6123047, Throughput: 98.359367 im/s
Epoch 9, Train Loss: 12236.3759766, Time: 50.7529s, Throughput: 98.358896 im/s
Epoch 10, Iter 39, Loss: 11941.0263672, Throughput: 97.530101 im/s
Epoch 10, Train Loss: 11909.1806641, Time: 51.1844s, Throughput: 97.529706 im/s
Epoch 11, Iter 39, Loss: 11271.3261719, Throughput: 97.283290 im/s
Epoch 11, Train Loss: 11804.5966797, Time: 51.3143s, Throughput: 97.282917 im/s
Epoch 12, Iter 39, Loss: 11528.5488281, Throughput: 97.807829 im/s
Epoch 12, Train Loss: 11533.3632812, Time: 51.0391s, Throughput: 97.807404 im/s
Epoch 13, Iter 39, Loss: 11777.8886719, Throughput: 97.669103 im/s
Epoch 13, Train Loss: 11318.3916016, Time: 51.1116s, Throughput: 97.668643 im/s
Epoch 14, Iter 39, Loss: 11570.7900391, Throughput: 97.383499 im/s
Epoch 14, Train Loss: 11182.4931641, Time: 51.2615s, Throughput: 97.383089 im/s
Epoch 15, Iter 39, Loss: 10858.4511719, Throughput: 97.544539 im/s
Epoch 15, Train Loss: 11079.4189453, Time: 51.1768s, Throughput: 97.544171 im/s
Epoch 16, Iter 39, Loss: 11222.0996094, Throughput: 97.353060 im/s
Epoch 16, Train Loss: 10974.4482422, Time: 51.2775s, Throughput: 97.352628 im/s
Epoch 17, Iter 39, Loss: 11016.3710938, Throughput: 97.438123 im/s
Epoch 17, Train Loss: 10927.1220703, Time: 51.2327s, Throughput: 97.437687 im/s
Epoch 18, Iter 39, Loss: 10606.3886719, Throughput: 97.158614 im/s
Epoch 18, Train Loss: 10741.6748047, Time: 51.3801s, Throughput: 97.158227 im/s
Epoch 19, Iter 39, Loss: 11337.1347656, Throughput: 97.774519 im/s
Epoch 19, Train Loss: 10674.8974609, Time: 51.0565s, Throughput: 97.774128 im/s
Epoch 20, Iter 39, Loss: 10190.8125000, Throughput: 98.583617 im/s
Epoch 20, Train Loss: 10614.8886719, Time: 50.6374s, Throughput: 98.583255 im/s
Epoch 21, Iter 39, Loss: 9972.5859375, Throughput: 98.236610 im/s
Epoch 21, Train Loss: 10465.0224609, Time: 50.8163s, Throughput: 98.236215 im/s
Epoch 22, Iter 39, Loss: 9916.8759766, Throughput: 97.326656 im/s
Epoch 22, Train Loss: 10413.7861328, Time: 51.2914s, Throughput: 97.326253 im/s
Epoch 23, Iter 39, Loss: 10620.9550781, Throughput: 97.098179 im/s
Epoch 23, Train Loss: 10430.7968750, Time: 51.4121s, Throughput: 97.097814 im/s
Epoch 24, Iter 39, Loss: 10316.2636719, Throughput: 97.349851 im/s
Epoch 24, Train Loss: 10363.3710938, Time: 51.2792s, Throughput: 97.349462 im/s
Epoch 25, Iter 39, Loss: 10142.3222656, Throughput: 97.312381 im/s
Epoch 25, Train Loss: 10242.8007812, Time: 51.2989s, Throughput: 97.312008 im/s
Epoch 26, Iter 39, Loss: 10514.8183594, Throughput: 97.351113 im/s
Epoch 26, Train Loss: 10164.4707031, Time: 51.2785s, Throughput: 97.350740 im/s
Epoch 27, Iter 39, Loss: 9635.9062500, Throughput: 98.015039 im/s
Epoch 27, Train Loss: 10157.4951172, Time: 50.9311s, Throughput: 98.014686 im/s
Epoch 28, Iter 39, Loss: 10041.7958984, Throughput: 97.630152 im/s
Epoch 28, Train Loss: 10048.6835938, Time: 51.1320s, Throughput: 97.629718 im/s
Epoch 29, Iter 39, Loss: 10166.3681641, Throughput: 97.746285 im/s
Epoch 29, Train Loss: 10044.4560547, Time: 51.0712s, Throughput: 97.745910 im/s
Epoch 30, Iter 39, Loss: 9988.2353516, Throughput: 96.421535 im/s
Epoch 30, Train Loss: 9946.4785156, Time: 51.7729s, Throughput: 96.421177 im/s
Epoch 31, Iter 39, Loss: 10104.6621094, Throughput: 96.610029 im/s
Epoch 31, Train Loss: 9946.2119141, Time: 51.6719s, Throughput: 96.609632 im/s
Epoch 32, Iter 39, Loss: 10011.8681641, Throughput: 93.358670 im/s
Epoch 32, Train Loss: 9884.3222656, Time: 53.4714s, Throughput: 93.358348 im/s
Epoch 33, Iter 39, Loss: 9824.2324219, Throughput: 93.205072 im/s
Epoch 33, Train Loss: 9810.3730469, Time: 53.5595s, Throughput: 93.204720 im/s
Epoch 34, Iter 39, Loss: 9957.2324219, Throughput: 94.629812 im/s
Epoch 34, Train Loss: 9775.2792969, Time: 52.7531s, Throughput: 94.629494 im/s
Epoch 35, Iter 39, Loss: 9131.6240234, Throughput: 96.677267 im/s
Epoch 35, Train Loss: 9744.8837891, Time: 51.6359s, Throughput: 96.676917 im/s
Epoch 36, Iter 39, Loss: 9632.5097656, Throughput: 95.546675 im/s
Epoch 36, Train Loss: 9772.2119141, Time: 52.2470s, Throughput: 95.546200 im/s
Epoch 37, Iter 39, Loss: 9601.5732422, Throughput: 94.697127 im/s
Epoch 37, Train Loss: 9661.4248047, Time: 52.7156s, Throughput: 94.696799 im/s
Epoch 38, Iter 39, Loss: 10350.1708984, Throughput: 94.895847 im/s
Epoch 38, Train Loss: 9656.5371094, Time: 52.6052s, Throughput: 94.895489 im/s
Epoch 39, Iter 39, Loss: 9331.7695312, Throughput: 94.390157 im/s
Epoch 39, Train Loss: 9590.8017578, Time: 52.8871s, Throughput: 94.389795 im/s
Epoch 40, Iter 39, Loss: 9566.7343750, Throughput: 93.451544 im/s
Epoch 40, Train Loss: 9567.4101562, Time: 53.4183s, Throughput: 93.451170 im/s
Epoch 41, Iter 39, Loss: 9871.8476562, Throughput: 94.658787 im/s
Epoch 41, Train Loss: 9522.9462891, Time: 52.7370s, Throughput: 94.658435 im/s
Epoch 42, Iter 39, Loss: 10030.2236328, Throughput: 95.980403 im/s
Epoch 42, Train Loss: 9549.0966797, Time: 52.0108s, Throughput: 95.980034 im/s
Epoch 43, Iter 39, Loss: 9246.1425781, Throughput: 97.049327 im/s
Epoch 43, Train Loss: 9475.3291016, Time: 51.4379s, Throughput: 97.048982 im/s
Epoch 44, Iter 39, Loss: 9492.1044922, Throughput: 95.942119 im/s
Epoch 44, Train Loss: 9424.0869141, Time: 52.0316s, Throughput: 95.941780 im/s
Epoch 45, Iter 39, Loss: 9805.0507812, Throughput: 94.348304 im/s
Epoch 45, Train Loss: 9457.3955078, Time: 52.9105s, Throughput: 94.347944 im/s
Epoch 46, Iter 39, Loss: 9639.4472656, Throughput: 93.571768 im/s
Epoch 46, Train Loss: 9414.0332031, Time: 53.3496s, Throughput: 93.571389 im/s
Epoch 47, Iter 39, Loss: 9221.5761719, Throughput: 95.962931 im/s
Epoch 47, Train Loss: 9365.6806641, Time: 52.0203s, Throughput: 95.962556 im/s
Epoch 48, Iter 39, Loss: 9377.5126953, Throughput: 97.141213 im/s
Epoch 48, Train Loss: 9412.6855469, Time: 51.3893s, Throughput: 97.140865 im/s
Epoch 49, Iter 39, Loss: 8934.1396484, Throughput: 96.945030 im/s
Epoch 49, Train Loss: 9405.9228516, Time: 51.4933s, Throughput: 96.944636 im/s
Epoch 50, Iter 39, Loss: 9049.1835938, Throughput: 96.482075 im/s
Epoch 50, Train Loss: 9343.9814453, Time: 51.7404s, Throughput: 96.481745 im/s

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
  CPU: 4 × INTEL(R) XEON(R) PLATINUM 8573C
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, sapphirerapids)
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
