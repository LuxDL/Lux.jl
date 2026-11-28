---
url: /previews/PR1848/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24606.4394531, Throughput: 22.780896 im/s
Epoch 1, Train Loss: 39820.1328125, Time: 219.3427s, Throughput: 22.758910 im/s
Epoch 2, Iter 39, Loss: 17319.9511719, Throughput: 173.238942 im/s
Epoch 2, Train Loss: 20387.2285156, Time: 28.8158s, Throughput: 173.237999 im/s
Epoch 3, Iter 39, Loss: 16089.3583984, Throughput: 173.441609 im/s
Epoch 3, Train Loss: 16948.0683594, Time: 28.7822s, Throughput: 173.440645 im/s
Epoch 4, Iter 39, Loss: 15810.4667969, Throughput: 173.133298 im/s
Epoch 4, Train Loss: 15497.7500000, Time: 28.8335s, Throughput: 173.132236 im/s
Epoch 5, Iter 39, Loss: 13568.4042969, Throughput: 173.353751 im/s
Epoch 5, Train Loss: 14618.6923828, Time: 28.7968s, Throughput: 173.352859 im/s
Epoch 6, Iter 39, Loss: 13829.2041016, Throughput: 173.359772 im/s
Epoch 6, Train Loss: 13852.4726562, Time: 28.7958s, Throughput: 173.358705 im/s
Epoch 7, Iter 39, Loss: 12531.5644531, Throughput: 173.487431 im/s
Epoch 7, Train Loss: 13359.7070312, Time: 28.7746s, Throughput: 173.486418 im/s
Epoch 8, Iter 39, Loss: 12347.5927734, Throughput: 173.901824 im/s
Epoch 8, Train Loss: 13089.2578125, Time: 28.7060s, Throughput: 173.900806 im/s
Epoch 9, Iter 39, Loss: 13357.4199219, Throughput: 173.888381 im/s
Epoch 9, Train Loss: 12664.6816406, Time: 28.7082s, Throughput: 173.887418 im/s
Epoch 10, Iter 39, Loss: 12985.4697266, Throughput: 172.757384 im/s
Epoch 10, Train Loss: 12391.1357422, Time: 28.8962s, Throughput: 172.756458 im/s
Epoch 11, Iter 39, Loss: 12518.8857422, Throughput: 173.218485 im/s
Epoch 11, Train Loss: 12287.8417969, Time: 28.8193s, Throughput: 173.217373 im/s
Epoch 12, Iter 39, Loss: 12418.7021484, Throughput: 173.804912 im/s
Epoch 12, Train Loss: 12136.3466797, Time: 28.7220s, Throughput: 173.803775 im/s
Epoch 13, Iter 39, Loss: 12436.5566406, Throughput: 173.317042 im/s
Epoch 13, Train Loss: 11847.9570312, Time: 28.8029s, Throughput: 173.315898 im/s
Epoch 14, Iter 39, Loss: 12771.9472656, Throughput: 173.429888 im/s
Epoch 14, Train Loss: 11669.1103516, Time: 28.7841s, Throughput: 173.428971 im/s
Epoch 15, Iter 39, Loss: 11553.3437500, Throughput: 170.882434 im/s
Epoch 15, Train Loss: 11641.6230469, Time: 29.2132s, Throughput: 170.881463 im/s
Epoch 16, Iter 39, Loss: 11681.6162109, Throughput: 173.425610 im/s
Epoch 16, Train Loss: 11568.7324219, Time: 28.7848s, Throughput: 173.424657 im/s
Epoch 17, Iter 39, Loss: 10571.0820312, Throughput: 173.326509 im/s
Epoch 17, Train Loss: 11330.8076172, Time: 28.8013s, Throughput: 173.325564 im/s
Epoch 18, Iter 39, Loss: 11683.5205078, Throughput: 173.260979 im/s
Epoch 18, Train Loss: 11251.1621094, Time: 28.8122s, Throughput: 173.260107 im/s
Epoch 19, Iter 39, Loss: 11317.9218750, Throughput: 173.528728 im/s
Epoch 19, Train Loss: 11177.4082031, Time: 28.7677s, Throughput: 173.527769 im/s
Epoch 20, Iter 39, Loss: 10833.0224609, Throughput: 173.124273 im/s
Epoch 20, Train Loss: 11196.8593750, Time: 28.8349s, Throughput: 173.123301 im/s
Epoch 21, Iter 39, Loss: 11083.7949219, Throughput: 173.033198 im/s
Epoch 21, Train Loss: 11039.5087891, Time: 28.8501s, Throughput: 173.032143 im/s
Epoch 22, Iter 39, Loss: 10967.9511719, Throughput: 173.471321 im/s
Epoch 22, Train Loss: 11021.7167969, Time: 28.7773s, Throughput: 173.470284 im/s
Epoch 23, Iter 39, Loss: 11263.1562500, Throughput: 173.258050 im/s
Epoch 23, Train Loss: 10830.7314453, Time: 28.8127s, Throughput: 173.256986 im/s
Epoch 24, Iter 39, Loss: 10399.7851562, Throughput: 172.383649 im/s
Epoch 24, Train Loss: 10764.0888672, Time: 28.9589s, Throughput: 172.382507 im/s
Epoch 25, Iter 39, Loss: 9758.1523438, Throughput: 173.666834 im/s
Epoch 25, Train Loss: 10700.4316406, Time: 28.7449s, Throughput: 173.665862 im/s
Epoch 26, Iter 39, Loss: 10635.9140625, Throughput: 173.159465 im/s
Epoch 26, Train Loss: 10726.5048828, Time: 28.8291s, Throughput: 173.158539 im/s
Epoch 27, Iter 39, Loss: 10971.9589844, Throughput: 173.279599 im/s
Epoch 27, Train Loss: 10648.1455078, Time: 28.8091s, Throughput: 173.278636 im/s
Epoch 28, Iter 39, Loss: 10276.9658203, Throughput: 172.886463 im/s
Epoch 28, Train Loss: 10562.6767578, Time: 28.8746s, Throughput: 172.885361 im/s
Epoch 29, Iter 39, Loss: 10070.5781250, Throughput: 172.419157 im/s
Epoch 29, Train Loss: 10509.2001953, Time: 28.9529s, Throughput: 172.418003 im/s
Epoch 30, Iter 39, Loss: 9909.0019531, Throughput: 172.976605 im/s
Epoch 30, Train Loss: 10498.4023438, Time: 28.8596s, Throughput: 172.975383 im/s
Epoch 31, Iter 39, Loss: 10638.7363281, Throughput: 172.889422 im/s
Epoch 31, Train Loss: 10396.5644531, Time: 28.8741s, Throughput: 172.888277 im/s
Epoch 32, Iter 39, Loss: 10286.5800781, Throughput: 173.274902 im/s
Epoch 32, Train Loss: 10358.8251953, Time: 28.8099s, Throughput: 173.273855 im/s
Epoch 33, Iter 39, Loss: 10487.8798828, Throughput: 173.130032 im/s
Epoch 33, Train Loss: 10287.6767578, Time: 28.8340s, Throughput: 173.129065 im/s
Epoch 34, Iter 39, Loss: 9861.5058594, Throughput: 172.631681 im/s
Epoch 34, Train Loss: 10267.2285156, Time: 28.9172s, Throughput: 172.630673 im/s
Epoch 35, Iter 39, Loss: 10331.3896484, Throughput: 173.383879 im/s
Epoch 35, Train Loss: 10144.0888672, Time: 28.7918s, Throughput: 173.382784 im/s
Epoch 36, Iter 39, Loss: 9551.8535156, Throughput: 172.974195 im/s
Epoch 36, Train Loss: 10183.0234375, Time: 28.8600s, Throughput: 172.973254 im/s
Epoch 37, Iter 39, Loss: 10552.5625000, Throughput: 173.279050 im/s
Epoch 37, Train Loss: 10152.2500000, Time: 28.8092s, Throughput: 173.277992 im/s
Epoch 38, Iter 39, Loss: 9819.7187500, Throughput: 172.680917 im/s
Epoch 38, Train Loss: 10095.1230469, Time: 28.9090s, Throughput: 172.679866 im/s
Epoch 39, Iter 39, Loss: 10004.6816406, Throughput: 172.302148 im/s
Epoch 39, Train Loss: 10117.6572266, Time: 28.9725s, Throughput: 172.301202 im/s
Epoch 40, Iter 39, Loss: 9705.8339844, Throughput: 173.018511 im/s
Epoch 40, Train Loss: 10032.4033203, Time: 28.8526s, Throughput: 173.017426 im/s
Epoch 41, Iter 39, Loss: 9863.1250000, Throughput: 173.369223 im/s
Epoch 41, Train Loss: 10005.1318359, Time: 28.7942s, Throughput: 173.368224 im/s
Epoch 42, Iter 39, Loss: 9838.2636719, Throughput: 173.521936 im/s
Epoch 42, Train Loss: 10026.5292969, Time: 28.7689s, Throughput: 173.520916 im/s
Epoch 43, Iter 39, Loss: 9859.5273438, Throughput: 173.395099 im/s
Epoch 43, Train Loss: 9912.3212891, Time: 28.7899s, Throughput: 173.394034 im/s
Epoch 44, Iter 39, Loss: 10002.6894531, Throughput: 173.559635 im/s
Epoch 44, Train Loss: 9878.7343750, Time: 28.7626s, Throughput: 173.558610 im/s
Epoch 45, Iter 39, Loss: 10149.4355469, Throughput: 173.234554 im/s
Epoch 45, Train Loss: 9819.6191406, Time: 28.8166s, Throughput: 173.233537 im/s
Epoch 46, Iter 39, Loss: 9517.7949219, Throughput: 173.154918 im/s
Epoch 46, Train Loss: 9810.4716797, Time: 28.8298s, Throughput: 173.153933 im/s
Epoch 47, Iter 39, Loss: 9799.0683594, Throughput: 173.135232 im/s
Epoch 47, Train Loss: 9848.4580078, Time: 28.8331s, Throughput: 173.134253 im/s
Epoch 48, Iter 39, Loss: 10156.3974609, Throughput: 173.883578 im/s
Epoch 48, Train Loss: 9849.1562500, Time: 28.7091s, Throughput: 173.882428 im/s
Epoch 49, Iter 39, Loss: 9838.6875000, Throughput: 173.931265 im/s
Epoch 49, Train Loss: 9738.0986328, Time: 28.7011s, Throughput: 173.930344 im/s
Epoch 50, Iter 39, Loss: 9452.2822266, Throughput: 173.146528 im/s
Epoch 50, Train Loss: 9675.9277344, Time: 28.8312s, Throughput: 173.145477 im/s

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
  CPU: 4 × AMD EPYC 9V74 80-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-20.1.8 (ORCJIT, znver4)
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
