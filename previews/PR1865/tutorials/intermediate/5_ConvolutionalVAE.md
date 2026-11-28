---
url: /previews/PR1865/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24462.5136719, Throughput: 18.745375 im/s
Epoch 1, Train Loss: 39931.5078125, Time: 266.5635s, Throughput: 18.727242 im/s
Epoch 2, Iter 39, Loss: 18683.2187500, Throughput: 147.568843 im/s
Epoch 2, Train Loss: 20577.1601562, Time: 33.8285s, Throughput: 147.567997 im/s
Epoch 3, Iter 39, Loss: 15885.8886719, Throughput: 147.217523 im/s
Epoch 3, Train Loss: 17026.7226562, Time: 33.9092s, Throughput: 147.216889 im/s
Epoch 4, Iter 39, Loss: 14652.2187500, Throughput: 147.881364 im/s
Epoch 4, Train Loss: 15394.3144531, Time: 33.7569s, Throughput: 147.880681 im/s
Epoch 5, Iter 39, Loss: 13224.7353516, Throughput: 148.044561 im/s
Epoch 5, Train Loss: 14466.7724609, Time: 33.7197s, Throughput: 148.043851 im/s
Epoch 6, Iter 39, Loss: 13738.1250000, Throughput: 147.421258 im/s
Epoch 6, Train Loss: 13679.6923828, Time: 33.8623s, Throughput: 147.420456 im/s
Epoch 7, Iter 39, Loss: 12613.6455078, Throughput: 148.088571 im/s
Epoch 7, Train Loss: 13181.7880859, Time: 33.7097s, Throughput: 148.087742 im/s
Epoch 8, Iter 39, Loss: 12872.3740234, Throughput: 147.675880 im/s
Epoch 8, Train Loss: 12796.2548828, Time: 33.8039s, Throughput: 147.675080 im/s
Epoch 9, Iter 39, Loss: 12000.7636719, Throughput: 147.966817 im/s
Epoch 9, Train Loss: 12569.8417969, Time: 33.7374s, Throughput: 147.966181 im/s
Epoch 10, Iter 39, Loss: 11963.6640625, Throughput: 147.721074 im/s
Epoch 10, Train Loss: 12209.2167969, Time: 33.7936s, Throughput: 147.720357 im/s
Epoch 11, Iter 39, Loss: 11068.1181641, Throughput: 147.551562 im/s
Epoch 11, Train Loss: 12029.4111328, Time: 33.8324s, Throughput: 147.550864 im/s
Epoch 12, Iter 39, Loss: 11946.1738281, Throughput: 147.615861 im/s
Epoch 12, Train Loss: 11858.0302734, Time: 33.8177s, Throughput: 147.615119 im/s
Epoch 13, Iter 39, Loss: 12251.5039062, Throughput: 147.828244 im/s
Epoch 13, Train Loss: 11722.4697266, Time: 33.7691s, Throughput: 147.827557 im/s
Epoch 14, Iter 39, Loss: 12350.7695312, Throughput: 148.123017 im/s
Epoch 14, Train Loss: 11562.6855469, Time: 33.7019s, Throughput: 148.122248 im/s
Epoch 15, Iter 39, Loss: 11408.1962891, Throughput: 147.657638 im/s
Epoch 15, Train Loss: 11389.1113281, Time: 33.8081s, Throughput: 147.656909 im/s
Epoch 16, Iter 39, Loss: 11161.0595703, Throughput: 147.784358 im/s
Epoch 16, Train Loss: 11331.8671875, Time: 33.7791s, Throughput: 147.783632 im/s
Epoch 17, Iter 39, Loss: 11651.3906250, Throughput: 147.867412 im/s
Epoch 17, Train Loss: 11279.3271484, Time: 33.7601s, Throughput: 147.866734 im/s
Epoch 18, Iter 39, Loss: 10572.1123047, Throughput: 147.328119 im/s
Epoch 18, Train Loss: 11101.4179688, Time: 33.8837s, Throughput: 147.327410 im/s
Epoch 19, Iter 39, Loss: 11131.5732422, Throughput: 147.943128 im/s
Epoch 19, Train Loss: 11015.9902344, Time: 33.7429s, Throughput: 147.942422 im/s
Epoch 20, Iter 39, Loss: 11390.1142578, Throughput: 147.863138 im/s
Epoch 20, Train Loss: 10913.1933594, Time: 33.7611s, Throughput: 147.862306 im/s
Epoch 21, Iter 39, Loss: 11027.9023438, Throughput: 148.044157 im/s
Epoch 21, Train Loss: 10861.2646484, Time: 33.7199s, Throughput: 148.043345 im/s
Epoch 22, Iter 39, Loss: 10358.8291016, Throughput: 148.092715 im/s
Epoch 22, Train Loss: 10782.0126953, Time: 33.7088s, Throughput: 148.091805 im/s
Epoch 23, Iter 39, Loss: 10708.7929688, Throughput: 148.122389 im/s
Epoch 23, Train Loss: 10696.9003906, Time: 33.7020s, Throughput: 148.121576 im/s
Epoch 24, Iter 39, Loss: 10677.5380859, Throughput: 148.196597 im/s
Epoch 24, Train Loss: 10679.4287109, Time: 33.6852s, Throughput: 148.195841 im/s
Epoch 25, Iter 39, Loss: 10732.5947266, Throughput: 147.858487 im/s
Epoch 25, Train Loss: 10571.1835938, Time: 33.7622s, Throughput: 147.857826 im/s
Epoch 26, Iter 39, Loss: 10725.7089844, Throughput: 147.324337 im/s
Epoch 26, Train Loss: 10485.0244141, Time: 33.8846s, Throughput: 147.323550 im/s
Epoch 27, Iter 39, Loss: 10294.0039062, Throughput: 148.279834 im/s
Epoch 27, Train Loss: 10490.0820312, Time: 33.6663s, Throughput: 148.279055 im/s
Epoch 28, Iter 39, Loss: 10576.2968750, Throughput: 148.606218 im/s
Epoch 28, Train Loss: 10496.5175781, Time: 33.5923s, Throughput: 148.605506 im/s
Epoch 29, Iter 39, Loss: 9997.8398438, Throughput: 148.436771 im/s
Epoch 29, Train Loss: 10362.9033203, Time: 33.6306s, Throughput: 148.436131 im/s
Epoch 30, Iter 39, Loss: 10429.8359375, Throughput: 148.496928 im/s
Epoch 30, Train Loss: 10334.4501953, Time: 33.6170s, Throughput: 148.496209 im/s
Epoch 31, Iter 39, Loss: 10604.2890625, Throughput: 148.018579 im/s
Epoch 31, Train Loss: 10332.9931641, Time: 33.7257s, Throughput: 148.017798 im/s
Epoch 32, Iter 39, Loss: 10485.5644531, Throughput: 148.434842 im/s
Epoch 32, Train Loss: 10300.7158203, Time: 33.6311s, Throughput: 148.434096 im/s
Epoch 33, Iter 39, Loss: 11214.8750000, Throughput: 147.876834 im/s
Epoch 33, Train Loss: 10185.8046875, Time: 33.7580s, Throughput: 147.876080 im/s
Epoch 34, Iter 39, Loss: 9671.9550781, Throughput: 148.128585 im/s
Epoch 34, Train Loss: 10141.6298828, Time: 33.7006s, Throughput: 148.127821 im/s
Epoch 35, Iter 39, Loss: 10295.5292969, Throughput: 148.066894 im/s
Epoch 35, Train Loss: 10048.5517578, Time: 33.7147s, Throughput: 148.066152 im/s
Epoch 36, Iter 39, Loss: 9764.4384766, Throughput: 147.566340 im/s
Epoch 36, Train Loss: 10023.6718750, Time: 33.8290s, Throughput: 147.565672 im/s
Epoch 37, Iter 39, Loss: 9479.0371094, Throughput: 147.718211 im/s
Epoch 37, Train Loss: 10038.4824219, Time: 33.7942s, Throughput: 147.717468 im/s
Epoch 38, Iter 39, Loss: 10145.7802734, Throughput: 147.327259 im/s
Epoch 38, Train Loss: 9984.9257812, Time: 33.8839s, Throughput: 147.326572 im/s
Epoch 39, Iter 39, Loss: 9447.0742188, Throughput: 147.849072 im/s
Epoch 39, Train Loss: 9973.8476562, Time: 33.7643s, Throughput: 147.848354 im/s
Epoch 40, Iter 39, Loss: 9669.6503906, Throughput: 148.143778 im/s
Epoch 40, Train Loss: 9925.2939453, Time: 33.6972s, Throughput: 148.143048 im/s
Epoch 41, Iter 39, Loss: 9851.8183594, Throughput: 148.111380 im/s
Epoch 41, Train Loss: 9877.6855469, Time: 33.7045s, Throughput: 148.110659 im/s
Epoch 42, Iter 39, Loss: 10062.1347656, Throughput: 148.178007 im/s
Epoch 42, Train Loss: 9874.4345703, Time: 33.6894s, Throughput: 148.177234 im/s
Epoch 43, Iter 39, Loss: 10204.3066406, Throughput: 147.773806 im/s
Epoch 43, Train Loss: 9847.7333984, Time: 33.7815s, Throughput: 147.773107 im/s
Epoch 44, Iter 39, Loss: 10038.2011719, Throughput: 148.604360 im/s
Epoch 44, Train Loss: 9830.8212891, Time: 33.5927s, Throughput: 148.603626 im/s
Epoch 45, Iter 39, Loss: 9494.3017578, Throughput: 147.667550 im/s
Epoch 45, Train Loss: 9793.4609375, Time: 33.8058s, Throughput: 147.666916 im/s
Epoch 46, Iter 39, Loss: 9400.4980469, Throughput: 148.183480 im/s
Epoch 46, Train Loss: 9687.8144531, Time: 33.6881s, Throughput: 148.182762 im/s
Epoch 47, Iter 39, Loss: 10381.3173828, Throughput: 146.857977 im/s
Epoch 47, Train Loss: 9670.1113281, Time: 33.9922s, Throughput: 146.857260 im/s
Epoch 48, Iter 39, Loss: 9712.3330078, Throughput: 147.975146 im/s
Epoch 48, Train Loss: 9627.8183594, Time: 33.7356s, Throughput: 147.974313 im/s
Epoch 49, Iter 39, Loss: 9226.3134766, Throughput: 147.968544 im/s
Epoch 49, Train Loss: 9595.6835938, Time: 33.7371s, Throughput: 147.967821 im/s
Epoch 50, Iter 39, Loss: 9236.7089844, Throughput: 148.080423 im/s
Epoch 50, Train Loss: 9630.0380859, Time: 33.7116s, Throughput: 148.079620 im/s

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
Julia Version 1.13.4
Commit c1ac32ab733 (2026-11-13 17:30 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × AMD EPYC 7763 64-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-20.1.8 (ORCJIT, znver3)
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
