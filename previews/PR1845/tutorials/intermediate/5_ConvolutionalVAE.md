---
url: /previews/PR1845/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24160.5410156, Throughput: 17.969734 im/s
Epoch 1, Train Loss: 40225.3828125, Time: 278.0041s, Throughput: 17.956572 im/s
Epoch 2, Iter 39, Loss: 18597.0820312, Throughput: 187.537588 im/s
Epoch 2, Train Loss: 20763.3710938, Time: 26.6189s, Throughput: 187.536221 im/s
Epoch 3, Iter 39, Loss: 15475.2109375, Throughput: 185.649005 im/s
Epoch 3, Train Loss: 16986.6679688, Time: 26.8896s, Throughput: 185.647783 im/s
Epoch 4, Iter 39, Loss: 14780.5722656, Throughput: 187.267532 im/s
Epoch 4, Train Loss: 15380.5751953, Time: 26.6572s, Throughput: 187.266318 im/s
Epoch 5, Iter 39, Loss: 13942.3642578, Throughput: 187.733920 im/s
Epoch 5, Train Loss: 14437.5126953, Time: 26.5910s, Throughput: 187.732649 im/s
Epoch 6, Iter 39, Loss: 13364.3076172, Throughput: 186.731452 im/s
Epoch 6, Train Loss: 13809.8027344, Time: 26.7338s, Throughput: 186.730194 im/s
Epoch 7, Iter 39, Loss: 13149.5429688, Throughput: 186.381508 im/s
Epoch 7, Train Loss: 13356.7187500, Time: 26.7839s, Throughput: 186.380312 im/s
Epoch 8, Iter 39, Loss: 13114.2949219, Throughput: 187.636281 im/s
Epoch 8, Train Loss: 12919.9472656, Time: 26.6048s, Throughput: 187.635095 im/s
Epoch 9, Iter 39, Loss: 12678.9140625, Throughput: 187.628768 im/s
Epoch 9, Train Loss: 12669.9140625, Time: 26.6059s, Throughput: 187.627717 im/s
Epoch 10, Iter 39, Loss: 12422.7880859, Throughput: 187.529858 im/s
Epoch 10, Train Loss: 12367.1445312, Time: 26.6199s, Throughput: 187.528751 im/s
Epoch 11, Iter 39, Loss: 11711.6132812, Throughput: 187.386402 im/s
Epoch 11, Train Loss: 12216.6582031, Time: 26.6404s, Throughput: 187.384911 im/s
Epoch 12, Iter 39, Loss: 11983.2031250, Throughput: 186.701610 im/s
Epoch 12, Train Loss: 11949.6855469, Time: 26.7380s, Throughput: 186.700277 im/s
Epoch 13, Iter 39, Loss: 11398.4941406, Throughput: 187.613743 im/s
Epoch 13, Train Loss: 11781.1054688, Time: 26.6080s, Throughput: 187.612474 im/s
Epoch 14, Iter 39, Loss: 12322.3496094, Throughput: 186.826220 im/s
Epoch 14, Train Loss: 11660.7744141, Time: 26.7202s, Throughput: 186.824990 im/s
Epoch 15, Iter 39, Loss: 12214.3759766, Throughput: 187.695302 im/s
Epoch 15, Train Loss: 11516.2060547, Time: 26.5965s, Throughput: 187.694096 im/s
Epoch 16, Iter 39, Loss: 10874.2558594, Throughput: 187.114548 im/s
Epoch 16, Train Loss: 11408.3544922, Time: 26.6790s, Throughput: 187.113467 im/s
Epoch 17, Iter 39, Loss: 11141.3603516, Throughput: 187.745882 im/s
Epoch 17, Train Loss: 11396.5253906, Time: 26.5893s, Throughput: 187.744660 im/s
Epoch 18, Iter 39, Loss: 10963.9296875, Throughput: 187.589362 im/s
Epoch 18, Train Loss: 11136.1718750, Time: 26.6115s, Throughput: 187.588128 im/s
Epoch 19, Iter 39, Loss: 10658.6025391, Throughput: 184.788794 im/s
Epoch 19, Train Loss: 11099.3193359, Time: 27.0148s, Throughput: 184.787646 im/s
Epoch 20, Iter 39, Loss: 10926.2431641, Throughput: 186.691143 im/s
Epoch 20, Train Loss: 11037.2734375, Time: 26.7395s, Throughput: 186.689963 im/s
Epoch 21, Iter 39, Loss: 11466.2480469, Throughput: 187.090480 im/s
Epoch 21, Train Loss: 11002.8281250, Time: 26.6824s, Throughput: 187.089273 im/s
Epoch 22, Iter 39, Loss: 10825.8212891, Throughput: 187.434901 im/s
Epoch 22, Train Loss: 10788.7294922, Time: 26.6334s, Throughput: 187.433705 im/s
Epoch 23, Iter 39, Loss: 10391.7304688, Throughput: 186.784999 im/s
Epoch 23, Train Loss: 10719.6894531, Time: 26.7261s, Throughput: 186.783531 im/s
Epoch 24, Iter 39, Loss: 10768.8652344, Throughput: 187.084450 im/s
Epoch 24, Train Loss: 10775.3876953, Time: 26.6833s, Throughput: 187.083118 im/s
Epoch 25, Iter 39, Loss: 10590.9179688, Throughput: 186.859125 im/s
Epoch 25, Train Loss: 10712.9072266, Time: 26.7155s, Throughput: 186.857859 im/s
Epoch 26, Iter 39, Loss: 10272.8779297, Throughput: 187.481424 im/s
Epoch 26, Train Loss: 10493.1767578, Time: 26.6268s, Throughput: 187.480388 im/s
Epoch 27, Iter 39, Loss: 11546.5986328, Throughput: 187.489148 im/s
Epoch 27, Train Loss: 10472.7841797, Time: 26.6257s, Throughput: 187.487936 im/s
Epoch 28, Iter 39, Loss: 10104.6142578, Throughput: 187.238685 im/s
Epoch 28, Train Loss: 10526.3095703, Time: 26.6613s, Throughput: 187.237632 im/s
Epoch 29, Iter 39, Loss: 9926.4042969, Throughput: 186.062454 im/s
Epoch 29, Train Loss: 10456.6767578, Time: 26.8299s, Throughput: 186.061282 im/s
Epoch 30, Iter 39, Loss: 10625.9316406, Throughput: 186.921060 im/s
Epoch 30, Train Loss: 10340.5527344, Time: 26.7066s, Throughput: 186.919989 im/s
Epoch 31, Iter 39, Loss: 10232.5546875, Throughput: 186.075514 im/s
Epoch 31, Train Loss: 10318.3662109, Time: 26.8280s, Throughput: 186.074224 im/s
Epoch 32, Iter 39, Loss: 10086.4394531, Throughput: 186.980137 im/s
Epoch 32, Train Loss: 10286.9238281, Time: 26.6982s, Throughput: 186.978953 im/s
Epoch 33, Iter 39, Loss: 10289.7089844, Throughput: 186.753225 im/s
Epoch 33, Train Loss: 10159.2822266, Time: 26.7306s, Throughput: 186.751961 im/s
Epoch 34, Iter 39, Loss: 10063.7773438, Throughput: 187.099447 im/s
Epoch 34, Train Loss: 10119.5146484, Time: 26.6812s, Throughput: 187.098332 im/s
Epoch 35, Iter 39, Loss: 9920.2324219, Throughput: 186.607460 im/s
Epoch 35, Train Loss: 10085.3906250, Time: 26.7515s, Throughput: 186.606163 im/s
Epoch 36, Iter 39, Loss: 9913.5312500, Throughput: 186.855825 im/s
Epoch 36, Train Loss: 10152.8593750, Time: 26.7159s, Throughput: 186.854761 im/s
Epoch 37, Iter 39, Loss: 10196.1601562, Throughput: 186.597968 im/s
Epoch 37, Train Loss: 10013.1777344, Time: 26.7529s, Throughput: 186.596629 im/s
Epoch 38, Iter 39, Loss: 10574.8457031, Throughput: 186.710226 im/s
Epoch 38, Train Loss: 9972.5761719, Time: 26.7368s, Throughput: 186.709096 im/s
Epoch 39, Iter 39, Loss: 10193.3183594, Throughput: 186.868114 im/s
Epoch 39, Train Loss: 9967.1152344, Time: 26.7142s, Throughput: 186.866638 im/s
Epoch 40, Iter 39, Loss: 10506.3730469, Throughput: 184.930148 im/s
Epoch 40, Train Loss: 9986.6533203, Time: 26.9942s, Throughput: 184.928930 im/s
Epoch 41, Iter 39, Loss: 9527.8261719, Throughput: 186.716588 im/s
Epoch 41, Train Loss: 9907.8671875, Time: 26.7359s, Throughput: 186.715576 im/s
Epoch 42, Iter 39, Loss: 10171.9677734, Throughput: 187.003693 im/s
Epoch 42, Train Loss: 9878.1044922, Time: 26.6949s, Throughput: 187.002285 im/s
Epoch 43, Iter 39, Loss: 10876.5585938, Throughput: 187.346065 im/s
Epoch 43, Train Loss: 9848.9785156, Time: 26.6460s, Throughput: 187.344997 im/s
Epoch 44, Iter 39, Loss: 9875.6054688, Throughput: 186.321759 im/s
Epoch 44, Train Loss: 9868.2080078, Time: 26.7925s, Throughput: 186.320605 im/s
Epoch 45, Iter 39, Loss: 10409.3447266, Throughput: 185.918546 im/s
Epoch 45, Train Loss: 9832.3300781, Time: 26.8506s, Throughput: 185.917362 im/s
Epoch 46, Iter 39, Loss: 10423.7128906, Throughput: 185.163301 im/s
Epoch 46, Train Loss: 9675.7666016, Time: 26.9602s, Throughput: 185.161935 im/s
Epoch 47, Iter 39, Loss: 9327.8388672, Throughput: 186.592297 im/s
Epoch 47, Train Loss: 9685.3388672, Time: 26.7537s, Throughput: 186.591195 im/s
Epoch 48, Iter 39, Loss: 9018.8417969, Throughput: 186.294445 im/s
Epoch 48, Train Loss: 9785.3203125, Time: 26.7965s, Throughput: 186.292985 im/s
Epoch 49, Iter 39, Loss: 9128.2314453, Throughput: 186.850802 im/s
Epoch 49, Train Loss: 9649.6835938, Time: 26.7167s, Throughput: 186.849677 im/s
Epoch 50, Iter 39, Loss: 9801.8759766, Throughput: 186.811350 im/s
Epoch 50, Train Loss: 9700.4912109, Time: 26.7223s, Throughput: 186.810120 im/s

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
