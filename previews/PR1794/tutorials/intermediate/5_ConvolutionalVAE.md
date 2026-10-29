---
url: /previews/PR1794/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24411.0371094, Throughput: 21.112510 im/s
Epoch 1, Train Loss: 39546.1796875, Time: 236.6400s, Throughput: 21.095334 im/s
Epoch 2, Iter 39, Loss: 17580.2832031, Throughput: 128.815069 im/s
Epoch 2, Train Loss: 19801.3105469, Time: 38.7534s, Throughput: 128.814478 im/s
Epoch 3, Iter 39, Loss: 15775.1972656, Throughput: 129.883913 im/s
Epoch 3, Train Loss: 16312.6396484, Time: 38.4345s, Throughput: 129.883291 im/s
Epoch 4, Iter 39, Loss: 13927.8339844, Throughput: 128.935564 im/s
Epoch 4, Train Loss: 14875.6025391, Time: 38.7172s, Throughput: 128.935001 im/s
Epoch 5, Iter 39, Loss: 14255.9912109, Throughput: 128.347215 im/s
Epoch 5, Train Loss: 13956.2001953, Time: 38.8947s, Throughput: 128.346635 im/s
Epoch 6, Iter 39, Loss: 13008.1347656, Throughput: 128.164668 im/s
Epoch 6, Train Loss: 13395.5595703, Time: 38.9501s, Throughput: 128.164096 im/s
Epoch 7, Iter 39, Loss: 12948.6562500, Throughput: 129.392875 im/s
Epoch 7, Train Loss: 12849.8759766, Time: 38.5803s, Throughput: 129.392358 im/s
Epoch 8, Iter 39, Loss: 13477.1933594, Throughput: 129.583864 im/s
Epoch 8, Train Loss: 12486.7128906, Time: 38.5235s, Throughput: 129.583140 im/s
Epoch 9, Iter 39, Loss: 12781.8769531, Throughput: 128.882359 im/s
Epoch 9, Train Loss: 12237.2421875, Time: 38.7332s, Throughput: 128.881766 im/s
Epoch 10, Iter 39, Loss: 11386.6474609, Throughput: 129.582915 im/s
Epoch 10, Train Loss: 11971.3984375, Time: 38.5238s, Throughput: 129.582266 im/s
Epoch 11, Iter 39, Loss: 11693.3203125, Throughput: 128.956254 im/s
Epoch 11, Train Loss: 11826.4980469, Time: 38.7110s, Throughput: 128.955588 im/s
Epoch 12, Iter 39, Loss: 11324.9492188, Throughput: 128.495122 im/s
Epoch 12, Train Loss: 11647.0488281, Time: 38.8499s, Throughput: 128.494450 im/s
Epoch 13, Iter 39, Loss: 11266.4843750, Throughput: 128.349819 im/s
Epoch 13, Train Loss: 11387.5556641, Time: 38.8939s, Throughput: 128.349229 im/s
Epoch 14, Iter 39, Loss: 11865.0957031, Throughput: 129.122238 im/s
Epoch 14, Train Loss: 11297.2109375, Time: 38.6612s, Throughput: 129.121577 im/s
Epoch 15, Iter 39, Loss: 11197.9736328, Throughput: 128.357377 im/s
Epoch 15, Train Loss: 11138.7988281, Time: 38.8916s, Throughput: 128.356793 im/s
Epoch 16, Iter 39, Loss: 10363.2597656, Throughput: 127.891794 im/s
Epoch 16, Train Loss: 11003.1660156, Time: 39.0332s, Throughput: 127.891201 im/s
Epoch 17, Iter 39, Loss: 10842.3056641, Throughput: 127.268302 im/s
Epoch 17, Train Loss: 10930.2939453, Time: 39.2244s, Throughput: 127.267663 im/s
Epoch 18, Iter 39, Loss: 10301.6357422, Throughput: 127.998638 im/s
Epoch 18, Train Loss: 10913.3369141, Time: 39.0006s, Throughput: 127.998084 im/s
Epoch 19, Iter 39, Loss: 10619.3037109, Throughput: 128.055872 im/s
Epoch 19, Train Loss: 10703.5546875, Time: 38.9832s, Throughput: 128.055316 im/s
Epoch 20, Iter 39, Loss: 10454.1572266, Throughput: 128.156114 im/s
Epoch 20, Train Loss: 10576.2744141, Time: 38.9527s, Throughput: 128.155403 im/s
Epoch 21, Iter 39, Loss: 10399.8750000, Throughput: 127.211730 im/s
Epoch 21, Train Loss: 10490.2832031, Time: 39.2419s, Throughput: 127.211098 im/s
Epoch 22, Iter 39, Loss: 10363.8027344, Throughput: 127.913747 im/s
Epoch 22, Train Loss: 10532.4130859, Time: 39.0265s, Throughput: 127.913203 im/s
Epoch 23, Iter 39, Loss: 10400.7050781, Throughput: 127.303977 im/s
Epoch 23, Train Loss: 10408.1679688, Time: 39.2134s, Throughput: 127.303390 im/s
Epoch 24, Iter 39, Loss: 10239.3359375, Throughput: 127.087492 im/s
Epoch 24, Train Loss: 10360.2529297, Time: 39.2802s, Throughput: 127.086891 im/s
Epoch 25, Iter 39, Loss: 9458.3144531, Throughput: 127.649645 im/s
Epoch 25, Train Loss: 10261.8232422, Time: 39.1072s, Throughput: 127.649077 im/s
Epoch 26, Iter 39, Loss: 10162.1367188, Throughput: 127.368195 im/s
Epoch 26, Train Loss: 10215.8076172, Time: 39.1936s, Throughput: 127.367590 im/s
Epoch 27, Iter 39, Loss: 10899.1123047, Throughput: 128.071178 im/s
Epoch 27, Train Loss: 10176.9873047, Time: 38.9785s, Throughput: 128.070534 im/s
Epoch 28, Iter 39, Loss: 9691.4375000, Throughput: 126.139300 im/s
Epoch 28, Train Loss: 10065.6464844, Time: 39.5755s, Throughput: 126.138707 im/s
Epoch 29, Iter 39, Loss: 10205.1250000, Throughput: 127.087531 im/s
Epoch 29, Train Loss: 9985.9687500, Time: 39.2802s, Throughput: 127.086987 im/s
Epoch 30, Iter 39, Loss: 9883.0703125, Throughput: 128.219067 im/s
Epoch 30, Train Loss: 10030.1025391, Time: 38.9336s, Throughput: 128.218402 im/s
Epoch 31, Iter 39, Loss: 9870.8398438, Throughput: 129.076802 im/s
Epoch 31, Train Loss: 9979.9785156, Time: 38.6748s, Throughput: 129.076221 im/s
Epoch 32, Iter 39, Loss: 9664.7421875, Throughput: 127.682960 im/s
Epoch 32, Train Loss: 9984.8144531, Time: 39.0970s, Throughput: 127.682428 im/s
Epoch 33, Iter 39, Loss: 10154.3046875, Throughput: 126.986210 im/s
Epoch 33, Train Loss: 9897.4687500, Time: 39.3116s, Throughput: 126.985545 im/s
Epoch 34, Iter 39, Loss: 10649.8505859, Throughput: 126.543882 im/s
Epoch 34, Train Loss: 9791.5273438, Time: 39.4490s, Throughput: 126.543279 im/s
Epoch 35, Iter 39, Loss: 9781.6796875, Throughput: 126.988737 im/s
Epoch 35, Train Loss: 9771.3310547, Time: 39.3107s, Throughput: 126.988194 im/s
Epoch 36, Iter 39, Loss: 9327.6796875, Throughput: 126.286982 im/s
Epoch 36, Train Loss: 9775.7304688, Time: 39.5292s, Throughput: 126.286276 im/s
Epoch 37, Iter 39, Loss: 10025.9589844, Throughput: 126.593357 im/s
Epoch 37, Train Loss: 9720.3945312, Time: 39.4335s, Throughput: 126.592757 im/s
Epoch 38, Iter 39, Loss: 9949.4843750, Throughput: 125.969647 im/s
Epoch 38, Train Loss: 9682.0156250, Time: 39.6288s, Throughput: 125.969123 im/s
Epoch 39, Iter 39, Loss: 9750.3300781, Throughput: 125.663493 im/s
Epoch 39, Train Loss: 9657.3505859, Time: 39.7254s, Throughput: 125.662817 im/s
Epoch 40, Iter 39, Loss: 9316.5605469, Throughput: 126.294831 im/s
Epoch 40, Train Loss: 9647.2177734, Time: 39.5268s, Throughput: 126.294202 im/s
Epoch 41, Iter 39, Loss: 9794.5859375, Throughput: 126.799353 im/s
Epoch 41, Train Loss: 9614.6269531, Time: 39.3694s, Throughput: 126.798838 im/s
Epoch 42, Iter 39, Loss: 9650.3251953, Throughput: 127.535564 im/s
Epoch 42, Train Loss: 9531.9707031, Time: 39.1422s, Throughput: 127.534919 im/s
Epoch 43, Iter 39, Loss: 9640.1787109, Throughput: 125.866502 im/s
Epoch 43, Train Loss: 9495.2158203, Time: 39.6613s, Throughput: 125.865889 im/s
Epoch 44, Iter 39, Loss: 9856.7675781, Throughput: 127.187503 im/s
Epoch 44, Train Loss: 9510.2460938, Time: 39.2493s, Throughput: 127.187005 im/s
Epoch 45, Iter 39, Loss: 9506.8955078, Throughput: 126.016523 im/s
Epoch 45, Train Loss: 9447.9218750, Time: 39.6140s, Throughput: 126.015986 im/s
Epoch 46, Iter 39, Loss: 9665.2802734, Throughput: 126.646799 im/s
Epoch 46, Train Loss: 9369.0380859, Time: 39.4169s, Throughput: 126.646159 im/s
Epoch 47, Iter 39, Loss: 9873.0507812, Throughput: 126.902240 im/s
Epoch 47, Train Loss: 9385.6953125, Time: 39.3375s, Throughput: 126.901702 im/s
Epoch 48, Iter 39, Loss: 9631.8447266, Throughput: 127.500529 im/s
Epoch 48, Train Loss: 9372.5332031, Time: 39.1530s, Throughput: 127.499913 im/s
Epoch 49, Iter 39, Loss: 10032.6718750, Throughput: 127.074575 im/s
Epoch 49, Train Loss: 9341.2148438, Time: 39.2843s, Throughput: 127.073811 im/s
Epoch 50, Iter 39, Loss: 9254.6523438, Throughput: 126.979172 im/s
Epoch 50, Train Loss: 9344.1943359, Time: 39.3137s, Throughput: 126.978620 im/s

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
