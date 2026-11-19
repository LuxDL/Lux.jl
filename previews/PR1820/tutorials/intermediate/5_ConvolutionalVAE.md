---
url: /previews/PR1820/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24283.8203125, Throughput: 15.256651 im/s
Epoch 1, Train Loss: 39662.7773438, Time: 327.4628s, Throughput: 15.244481 im/s
Epoch 2, Iter 39, Loss: 17122.7890625, Throughput: 97.611872 im/s
Epoch 2, Train Loss: 20084.0410156, Time: 51.1415s, Throughput: 97.611531 im/s
Epoch 3, Iter 39, Loss: 15288.4628906, Throughput: 98.138724 im/s
Epoch 3, Train Loss: 16676.2714844, Time: 50.8669s, Throughput: 98.138394 im/s
Epoch 4, Iter 39, Loss: 14766.0185547, Throughput: 98.255468 im/s
Epoch 4, Train Loss: 15153.8554688, Time: 50.8065s, Throughput: 98.255100 im/s
Epoch 5, Iter 39, Loss: 14430.5810547, Throughput: 98.291308 im/s
Epoch 5, Train Loss: 14225.1601562, Time: 50.7880s, Throughput: 98.290964 im/s
Epoch 6, Iter 39, Loss: 13136.7724609, Throughput: 98.283989 im/s
Epoch 6, Train Loss: 13505.3271484, Time: 50.7918s, Throughput: 98.283631 im/s
Epoch 7, Iter 39, Loss: 13388.9394531, Throughput: 98.196294 im/s
Epoch 7, Train Loss: 13229.3857422, Time: 50.8371s, Throughput: 98.195941 im/s
Epoch 8, Iter 39, Loss: 12517.1230469, Throughput: 97.899116 im/s
Epoch 8, Train Loss: 12725.6933594, Time: 50.9914s, Throughput: 97.898809 im/s
Epoch 9, Iter 39, Loss: 12117.2500000, Throughput: 97.864428 im/s
Epoch 9, Train Loss: 12351.9541016, Time: 51.0095s, Throughput: 97.864073 im/s
Epoch 10, Iter 39, Loss: 12057.8281250, Throughput: 98.134698 im/s
Epoch 10, Train Loss: 12028.8505859, Time: 50.8691s, Throughput: 98.134327 im/s
Epoch 11, Iter 39, Loss: 11303.1660156, Throughput: 98.116349 im/s
Epoch 11, Train Loss: 11871.6142578, Time: 50.8786s, Throughput: 98.115969 im/s
Epoch 12, Iter 39, Loss: 11598.2324219, Throughput: 98.409271 im/s
Epoch 12, Train Loss: 11612.4218750, Time: 50.7271s, Throughput: 98.408889 im/s
Epoch 13, Iter 39, Loss: 11408.8232422, Throughput: 98.015937 im/s
Epoch 13, Train Loss: 11491.3134766, Time: 50.9307s, Throughput: 98.015616 im/s
Epoch 14, Iter 39, Loss: 11616.3339844, Throughput: 98.250710 im/s
Epoch 14, Train Loss: 11259.6943359, Time: 50.8089s, Throughput: 98.250410 im/s
Epoch 15, Iter 39, Loss: 11046.8554688, Throughput: 98.080882 im/s
Epoch 15, Train Loss: 11128.3105469, Time: 50.8969s, Throughput: 98.080597 im/s
Epoch 16, Iter 39, Loss: 11157.6152344, Throughput: 98.139513 im/s
Epoch 16, Train Loss: 11120.2207031, Time: 50.8665s, Throughput: 98.139175 im/s
Epoch 17, Iter 39, Loss: 10287.5429688, Throughput: 98.185631 im/s
Epoch 17, Train Loss: 10858.4814453, Time: 50.8426s, Throughput: 98.185313 im/s
Epoch 18, Iter 39, Loss: 10726.9970703, Throughput: 98.426662 im/s
Epoch 18, Train Loss: 10796.8906250, Time: 50.7182s, Throughput: 98.426284 im/s
Epoch 19, Iter 39, Loss: 10624.5800781, Throughput: 97.696375 im/s
Epoch 19, Train Loss: 10768.5263672, Time: 51.0972s, Throughput: 97.696073 im/s
Epoch 20, Iter 39, Loss: 10598.0195312, Throughput: 98.414874 im/s
Epoch 20, Train Loss: 10594.2021484, Time: 50.7242s, Throughput: 98.414556 im/s
Epoch 21, Iter 39, Loss: 9703.4248047, Throughput: 98.319162 im/s
Epoch 21, Train Loss: 10555.7597656, Time: 50.7736s, Throughput: 98.318817 im/s
Epoch 22, Iter 39, Loss: 9524.4990234, Throughput: 98.159853 im/s
Epoch 22, Train Loss: 10450.3847656, Time: 50.8560s, Throughput: 98.159529 im/s
Epoch 23, Iter 39, Loss: 9738.3867188, Throughput: 98.266452 im/s
Epoch 23, Train Loss: 10369.2500000, Time: 50.8008s, Throughput: 98.266096 im/s
Epoch 24, Iter 39, Loss: 10100.5927734, Throughput: 98.501353 im/s
Epoch 24, Train Loss: 10224.5810547, Time: 50.6797s, Throughput: 98.501022 im/s
Epoch 25, Iter 39, Loss: 10099.2919922, Throughput: 98.127790 im/s
Epoch 25, Train Loss: 10225.1171875, Time: 50.8727s, Throughput: 98.127315 im/s
Epoch 26, Iter 39, Loss: 10069.4169922, Throughput: 98.681844 im/s
Epoch 26, Train Loss: 10183.0537109, Time: 50.5870s, Throughput: 98.681520 im/s
Epoch 27, Iter 39, Loss: 9411.2949219, Throughput: 98.516893 im/s
Epoch 27, Train Loss: 10032.0917969, Time: 50.6717s, Throughput: 98.516541 im/s
Epoch 28, Iter 39, Loss: 10089.2949219, Throughput: 98.177307 im/s
Epoch 28, Train Loss: 9974.5292969, Time: 50.8469s, Throughput: 98.176998 im/s
Epoch 29, Iter 39, Loss: 10786.6679688, Throughput: 98.021965 im/s
Epoch 29, Train Loss: 10016.3642578, Time: 50.9275s, Throughput: 98.021646 im/s
Epoch 30, Iter 39, Loss: 9509.3867188, Throughput: 98.541678 im/s
Epoch 30, Train Loss: 10067.5556641, Time: 50.6590s, Throughput: 98.541314 im/s
Epoch 31, Iter 39, Loss: 9426.3837891, Throughput: 98.162294 im/s
Epoch 31, Train Loss: 9895.7753906, Time: 50.8547s, Throughput: 98.161941 im/s
Epoch 32, Iter 39, Loss: 10134.3251953, Throughput: 98.510008 im/s
Epoch 32, Train Loss: 9877.4355469, Time: 50.6752s, Throughput: 98.509705 im/s
Epoch 33, Iter 39, Loss: 9330.7978516, Throughput: 98.433325 im/s
Epoch 33, Train Loss: 9763.6972656, Time: 50.7147s, Throughput: 98.432968 im/s
Epoch 34, Iter 39, Loss: 9979.0654297, Throughput: 98.278359 im/s
Epoch 34, Train Loss: 9781.3955078, Time: 50.7947s, Throughput: 98.278003 im/s
Epoch 35, Iter 39, Loss: 9925.8574219, Throughput: 98.600552 im/s
Epoch 35, Train Loss: 9763.0332031, Time: 50.6287s, Throughput: 98.600226 im/s
Epoch 36, Iter 39, Loss: 9598.5654297, Throughput: 98.658523 im/s
Epoch 36, Train Loss: 9685.7968750, Time: 50.5990s, Throughput: 98.658154 im/s
Epoch 37, Iter 39, Loss: 9799.2236328, Throughput: 98.205331 im/s
Epoch 37, Train Loss: 9683.2666016, Time: 50.8324s, Throughput: 98.204987 im/s
Epoch 38, Iter 39, Loss: 10178.0371094, Throughput: 98.453953 im/s
Epoch 38, Train Loss: 9594.0810547, Time: 50.7041s, Throughput: 98.453631 im/s
Epoch 39, Iter 39, Loss: 9293.9609375, Throughput: 98.588924 im/s
Epoch 39, Train Loss: 9559.2607422, Time: 50.6347s, Throughput: 98.588568 im/s
Epoch 40, Iter 39, Loss: 9919.8974609, Throughput: 98.778829 im/s
Epoch 40, Train Loss: 9539.7500000, Time: 50.5373s, Throughput: 98.778497 im/s
Epoch 41, Iter 39, Loss: 9772.7587891, Throughput: 98.830529 im/s
Epoch 41, Train Loss: 9516.7626953, Time: 50.5109s, Throughput: 98.830212 im/s
Epoch 42, Iter 39, Loss: 9701.6328125, Throughput: 98.708188 im/s
Epoch 42, Train Loss: 9452.7412109, Time: 50.5735s, Throughput: 98.707853 im/s
Epoch 43, Iter 39, Loss: 10079.6113281, Throughput: 98.911751 im/s
Epoch 43, Train Loss: 9439.9414062, Time: 50.4694s, Throughput: 98.911410 im/s
Epoch 44, Iter 39, Loss: 9792.1728516, Throughput: 98.614650 im/s
Epoch 44, Train Loss: 9391.9980469, Time: 50.6215s, Throughput: 98.614315 im/s
Epoch 45, Iter 39, Loss: 9499.8320312, Throughput: 98.971374 im/s
Epoch 45, Train Loss: 9407.0957031, Time: 50.4390s, Throughput: 98.971073 im/s
Epoch 46, Iter 39, Loss: 9006.2890625, Throughput: 98.486535 im/s
Epoch 46, Train Loss: 9429.9628906, Time: 50.6873s, Throughput: 98.486218 im/s
Epoch 47, Iter 39, Loss: 9606.8437500, Throughput: 98.501582 im/s
Epoch 47, Train Loss: 9365.2880859, Time: 50.6796s, Throughput: 98.501263 im/s
Epoch 48, Iter 39, Loss: 9557.8964844, Throughput: 97.992364 im/s
Epoch 48, Train Loss: 9268.6669922, Time: 50.9429s, Throughput: 97.992083 im/s
Epoch 49, Iter 39, Loss: 9401.2451172, Throughput: 98.619249 im/s
Epoch 49, Train Loss: 9337.5087891, Time: 50.6191s, Throughput: 98.618873 im/s
Epoch 50, Iter 39, Loss: 10195.2167969, Throughput: 99.083474 im/s
Epoch 50, Train Loss: 9297.3496094, Time: 50.3819s, Throughput: 99.083154 im/s

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
  CPU: 4 × Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, icelake-server)
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
