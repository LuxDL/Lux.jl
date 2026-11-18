---
url: /previews/PR1854/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24024.5742188, Throughput: 20.411474 im/s
Epoch 1, Train Loss: 39524.8984375, Time: 244.7861s, Throughput: 20.393317 im/s
Epoch 2, Iter 39, Loss: 17150.9335938, Throughput: 203.784914 im/s
Epoch 2, Train Loss: 20005.4667969, Time: 24.4966s, Throughput: 203.783575 im/s
Epoch 3, Iter 39, Loss: 15528.8564453, Throughput: 204.296998 im/s
Epoch 3, Train Loss: 16490.3105469, Time: 24.4352s, Throughput: 204.295594 im/s
Epoch 4, Iter 39, Loss: 14038.0419922, Throughput: 204.044979 im/s
Epoch 4, Train Loss: 14945.7207031, Time: 24.4654s, Throughput: 204.043504 im/s
Epoch 5, Iter 39, Loss: 13440.7050781, Throughput: 204.538368 im/s
Epoch 5, Train Loss: 13986.9648438, Time: 24.4063s, Throughput: 204.537135 im/s
Epoch 6, Iter 39, Loss: 12669.6191406, Throughput: 204.349818 im/s
Epoch 6, Train Loss: 13463.0878906, Time: 24.4289s, Throughput: 204.348422 im/s
Epoch 7, Iter 39, Loss: 13488.0087891, Throughput: 204.621084 im/s
Epoch 7, Train Loss: 12836.3886719, Time: 24.3965s, Throughput: 204.619818 im/s
Epoch 8, Iter 39, Loss: 12183.1074219, Throughput: 204.442177 im/s
Epoch 8, Train Loss: 12480.6669922, Time: 24.4178s, Throughput: 204.440779 im/s
Epoch 9, Iter 39, Loss: 12077.7275391, Throughput: 204.222696 im/s
Epoch 9, Train Loss: 12203.5341797, Time: 24.4441s, Throughput: 204.221419 im/s
Epoch 10, Iter 39, Loss: 11832.5068359, Throughput: 204.465022 im/s
Epoch 10, Train Loss: 11915.2138672, Time: 24.4151s, Throughput: 204.463514 im/s
Epoch 11, Iter 39, Loss: 12133.6582031, Throughput: 204.116740 im/s
Epoch 11, Train Loss: 11838.8417969, Time: 24.4568s, Throughput: 204.115387 im/s
Epoch 12, Iter 39, Loss: 11769.7558594, Throughput: 204.027266 im/s
Epoch 12, Train Loss: 11573.8779297, Time: 24.4675s, Throughput: 204.025973 im/s
Epoch 13, Iter 39, Loss: 11830.5957031, Throughput: 204.258646 im/s
Epoch 13, Train Loss: 11310.4990234, Time: 24.4398s, Throughput: 204.257385 im/s
Epoch 14, Iter 39, Loss: 11572.5703125, Throughput: 203.427924 im/s
Epoch 14, Train Loss: 11192.3017578, Time: 24.5396s, Throughput: 203.426456 im/s
Epoch 15, Iter 39, Loss: 11442.3359375, Throughput: 203.974437 im/s
Epoch 15, Train Loss: 11066.0107422, Time: 24.4738s, Throughput: 203.973070 im/s
Epoch 16, Iter 39, Loss: 11102.3515625, Throughput: 204.419289 im/s
Epoch 16, Train Loss: 11112.2050781, Time: 24.4206s, Throughput: 204.417966 im/s
Epoch 17, Iter 39, Loss: 11374.8671875, Throughput: 203.445210 im/s
Epoch 17, Train Loss: 10910.9560547, Time: 24.5375s, Throughput: 203.443941 im/s
Epoch 18, Iter 39, Loss: 10989.5097656, Throughput: 203.736217 im/s
Epoch 18, Train Loss: 10735.6025391, Time: 24.5024s, Throughput: 203.734895 im/s
Epoch 19, Iter 39, Loss: 10988.8164062, Throughput: 200.024788 im/s
Epoch 19, Train Loss: 10616.4931641, Time: 24.9571s, Throughput: 200.023562 im/s
Epoch 20, Iter 39, Loss: 10268.0722656, Throughput: 203.989689 im/s
Epoch 20, Train Loss: 10594.6386719, Time: 24.4720s, Throughput: 203.988364 im/s
Epoch 21, Iter 39, Loss: 10639.3945312, Throughput: 204.276498 im/s
Epoch 21, Train Loss: 10476.3330078, Time: 24.4376s, Throughput: 204.275077 im/s
Epoch 22, Iter 39, Loss: 10482.9257812, Throughput: 203.724637 im/s
Epoch 22, Train Loss: 10375.0781250, Time: 24.5038s, Throughput: 203.723239 im/s
Epoch 23, Iter 39, Loss: 10300.1054688, Throughput: 204.495434 im/s
Epoch 23, Train Loss: 10330.8906250, Time: 24.4115s, Throughput: 204.493936 im/s
Epoch 24, Iter 39, Loss: 10556.1464844, Throughput: 203.769407 im/s
Epoch 24, Train Loss: 10306.7255859, Time: 24.4985s, Throughput: 203.767835 im/s
Epoch 25, Iter 39, Loss: 10050.5898438, Throughput: 204.124485 im/s
Epoch 25, Train Loss: 10199.1162109, Time: 24.4558s, Throughput: 204.123016 im/s
Epoch 26, Iter 39, Loss: 9721.2636719, Throughput: 203.148368 im/s
Epoch 26, Train Loss: 10234.0761719, Time: 24.5733s, Throughput: 203.147012 im/s
Epoch 27, Iter 39, Loss: 10105.6748047, Throughput: 203.994576 im/s
Epoch 27, Train Loss: 10101.8281250, Time: 24.4714s, Throughput: 203.993201 im/s
Epoch 28, Iter 39, Loss: 10108.8671875, Throughput: 204.538477 im/s
Epoch 28, Train Loss: 10029.7558594, Time: 24.4063s, Throughput: 204.537111 im/s
Epoch 29, Iter 39, Loss: 10610.9853516, Throughput: 204.140520 im/s
Epoch 29, Train Loss: 10038.7060547, Time: 24.4539s, Throughput: 204.139216 im/s
Epoch 30, Iter 39, Loss: 10616.6015625, Throughput: 204.004928 im/s
Epoch 30, Train Loss: 10139.3740234, Time: 24.4701s, Throughput: 204.003661 im/s
Epoch 31, Iter 39, Loss: 9558.0917969, Throughput: 203.143971 im/s
Epoch 31, Train Loss: 9917.4892578, Time: 24.5739s, Throughput: 203.142715 im/s
Epoch 32, Iter 39, Loss: 10224.6542969, Throughput: 204.061626 im/s
Epoch 32, Train Loss: 9948.7304688, Time: 24.4633s, Throughput: 204.060393 im/s
Epoch 33, Iter 39, Loss: 10841.8691406, Throughput: 204.062320 im/s
Epoch 33, Train Loss: 9890.9746094, Time: 24.4633s, Throughput: 204.060886 im/s
Epoch 34, Iter 39, Loss: 9639.7773438, Throughput: 204.313779 im/s
Epoch 34, Train Loss: 9728.3876953, Time: 24.4332s, Throughput: 204.312374 im/s
Epoch 35, Iter 39, Loss: 10144.6650391, Throughput: 203.900505 im/s
Epoch 35, Train Loss: 9763.4902344, Time: 24.4827s, Throughput: 203.899238 im/s
Epoch 36, Iter 39, Loss: 10132.1279297, Throughput: 204.062253 im/s
Epoch 36, Train Loss: 9782.5126953, Time: 24.4633s, Throughput: 204.061043 im/s
Epoch 37, Iter 39, Loss: 9833.2597656, Throughput: 201.506935 im/s
Epoch 37, Train Loss: 9656.2285156, Time: 24.7735s, Throughput: 201.505634 im/s
Epoch 38, Iter 39, Loss: 9808.9912109, Throughput: 203.299214 im/s
Epoch 38, Train Loss: 9607.4003906, Time: 24.5551s, Throughput: 203.297890 im/s
Epoch 39, Iter 39, Loss: 9770.7929688, Throughput: 203.772852 im/s
Epoch 39, Train Loss: 9643.1015625, Time: 24.4980s, Throughput: 203.771620 im/s
Epoch 40, Iter 39, Loss: 10170.5058594, Throughput: 202.904131 im/s
Epoch 40, Train Loss: 9669.0546875, Time: 24.6029s, Throughput: 202.902860 im/s
Epoch 41, Iter 39, Loss: 9029.8798828, Throughput: 204.197987 im/s
Epoch 41, Train Loss: 9584.8310547, Time: 24.4470s, Throughput: 204.196577 im/s
Epoch 42, Iter 39, Loss: 9240.1328125, Throughput: 204.466252 im/s
Epoch 42, Train Loss: 9484.2656250, Time: 24.4149s, Throughput: 204.465004 im/s
Epoch 43, Iter 39, Loss: 9309.0390625, Throughput: 204.191338 im/s
Epoch 43, Train Loss: 9466.8593750, Time: 24.4478s, Throughput: 204.190053 im/s
Epoch 44, Iter 39, Loss: 9573.7597656, Throughput: 204.627417 im/s
Epoch 44, Train Loss: 9430.9150391, Time: 24.3957s, Throughput: 204.626167 im/s
Epoch 45, Iter 39, Loss: 9452.2167969, Throughput: 204.341721 im/s
Epoch 45, Train Loss: 9449.3642578, Time: 24.4298s, Throughput: 204.340508 im/s
Epoch 46, Iter 39, Loss: 9864.5664062, Throughput: 204.148450 im/s
Epoch 46, Train Loss: 9336.9160156, Time: 24.4529s, Throughput: 204.147214 im/s
Epoch 47, Iter 39, Loss: 9197.7773438, Throughput: 204.057173 im/s
Epoch 47, Train Loss: 9347.9765625, Time: 24.4639s, Throughput: 204.055789 im/s
Epoch 48, Iter 39, Loss: 9442.2597656, Throughput: 203.531601 im/s
Epoch 48, Train Loss: 9294.4863281, Time: 24.5271s, Throughput: 203.530107 im/s
Epoch 49, Iter 39, Loss: 9386.8447266, Throughput: 204.137097 im/s
Epoch 49, Train Loss: 9306.9130859, Time: 24.4543s, Throughput: 204.135894 im/s
Epoch 50, Iter 39, Loss: 9083.4199219, Throughput: 204.144542 im/s
Epoch 50, Train Loss: 9295.8906250, Time: 24.4534s, Throughput: 204.142982 im/s

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
