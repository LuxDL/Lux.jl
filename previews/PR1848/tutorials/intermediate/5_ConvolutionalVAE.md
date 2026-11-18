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
Epoch 1, Iter 39, Loss: 23667.6738281, Throughput: 20.292180 im/s
Epoch 1, Train Loss: 39684.6367188, Time: 246.2130s, Throughput: 20.275126 im/s
Epoch 2, Iter 39, Loss: 17367.1406250, Throughput: 205.059507 im/s
Epoch 2, Train Loss: 20136.3046875, Time: 24.3443s, Throughput: 205.058075 im/s
Epoch 3, Iter 39, Loss: 16216.6054688, Throughput: 204.398399 im/s
Epoch 3, Train Loss: 16413.0136719, Time: 24.4231s, Throughput: 204.396985 im/s
Epoch 4, Iter 39, Loss: 14759.4707031, Throughput: 204.119252 im/s
Epoch 4, Train Loss: 14876.6181641, Time: 24.4564s, Throughput: 204.118034 im/s
Epoch 5, Iter 39, Loss: 13073.4755859, Throughput: 203.791093 im/s
Epoch 5, Train Loss: 13981.4101562, Time: 24.4958s, Throughput: 203.789704 im/s
Epoch 6, Iter 39, Loss: 12960.4619141, Throughput: 204.805737 im/s
Epoch 6, Train Loss: 13284.4082031, Time: 24.3745s, Throughput: 204.804267 im/s
Epoch 7, Iter 39, Loss: 12613.4179688, Throughput: 204.566277 im/s
Epoch 7, Train Loss: 12817.5498047, Time: 24.4030s, Throughput: 204.564962 im/s
Epoch 8, Iter 39, Loss: 11741.5136719, Throughput: 205.409360 im/s
Epoch 8, Train Loss: 12520.6416016, Time: 24.3029s, Throughput: 205.407949 im/s
Epoch 9, Iter 39, Loss: 11567.2480469, Throughput: 205.500555 im/s
Epoch 9, Train Loss: 12166.2939453, Time: 24.2921s, Throughput: 205.499220 im/s
Epoch 10, Iter 39, Loss: 12258.5810547, Throughput: 204.930265 im/s
Epoch 10, Train Loss: 11990.7158203, Time: 24.3596s, Throughput: 204.929086 im/s
Epoch 11, Iter 39, Loss: 11762.9941406, Throughput: 204.435162 im/s
Epoch 11, Train Loss: 11686.1142578, Time: 24.4187s, Throughput: 204.433839 im/s
Epoch 12, Iter 39, Loss: 11409.7880859, Throughput: 204.023280 im/s
Epoch 12, Train Loss: 11451.5576172, Time: 24.4680s, Throughput: 204.021930 im/s
Epoch 13, Iter 39, Loss: 12042.1542969, Throughput: 204.787465 im/s
Epoch 13, Train Loss: 11391.1923828, Time: 24.3766s, Throughput: 204.786155 im/s
Epoch 14, Iter 39, Loss: 11108.2490234, Throughput: 205.139743 im/s
Epoch 14, Train Loss: 11343.3642578, Time: 24.3348s, Throughput: 205.138260 im/s
Epoch 15, Iter 39, Loss: 11503.8789062, Throughput: 204.891900 im/s
Epoch 15, Train Loss: 11092.7587891, Time: 24.3642s, Throughput: 204.890613 im/s
Epoch 16, Iter 39, Loss: 10682.0332031, Throughput: 204.231923 im/s
Epoch 16, Train Loss: 11021.7392578, Time: 24.4429s, Throughput: 204.230678 im/s
Epoch 17, Iter 39, Loss: 10584.8359375, Throughput: 205.051868 im/s
Epoch 17, Train Loss: 10815.1650391, Time: 24.3452s, Throughput: 205.050494 im/s
Epoch 18, Iter 39, Loss: 9918.9287109, Throughput: 204.786339 im/s
Epoch 18, Train Loss: 10782.6181641, Time: 24.3768s, Throughput: 204.785027 im/s
Epoch 19, Iter 39, Loss: 10305.4882812, Throughput: 204.494872 im/s
Epoch 19, Train Loss: 10718.8525391, Time: 24.4115s, Throughput: 204.493648 im/s
Epoch 20, Iter 39, Loss: 11160.7890625, Throughput: 205.143876 im/s
Epoch 20, Train Loss: 10570.3369141, Time: 24.3343s, Throughput: 205.142593 im/s
Epoch 21, Iter 39, Loss: 11155.5986328, Throughput: 204.296254 im/s
Epoch 21, Train Loss: 10534.5605469, Time: 24.4352s, Throughput: 204.295040 im/s
Epoch 22, Iter 39, Loss: 10459.3671875, Throughput: 205.609887 im/s
Epoch 22, Train Loss: 10377.7734375, Time: 24.2791s, Throughput: 205.608532 im/s
Epoch 23, Iter 39, Loss: 10745.5039062, Throughput: 204.620052 im/s
Epoch 23, Train Loss: 10406.4267578, Time: 24.3966s, Throughput: 204.618870 im/s
Epoch 24, Iter 39, Loss: 10463.7382812, Throughput: 204.578627 im/s
Epoch 24, Train Loss: 10346.9335938, Time: 24.4015s, Throughput: 204.577362 im/s
Epoch 25, Iter 39, Loss: 10244.0019531, Throughput: 205.758162 im/s
Epoch 25, Train Loss: 10265.8525391, Time: 24.2616s, Throughput: 205.756856 im/s
Epoch 26, Iter 39, Loss: 11145.3847656, Throughput: 205.365941 im/s
Epoch 26, Train Loss: 10132.8232422, Time: 24.3080s, Throughput: 205.364565 im/s
Epoch 27, Iter 39, Loss: 10607.6660156, Throughput: 202.735655 im/s
Epoch 27, Train Loss: 10217.2597656, Time: 24.6234s, Throughput: 202.734279 im/s
Epoch 28, Iter 39, Loss: 9826.2675781, Throughput: 204.765473 im/s
Epoch 28, Train Loss: 10074.0429688, Time: 24.3793s, Throughput: 204.764129 im/s
Epoch 29, Iter 39, Loss: 9538.8671875, Throughput: 204.892893 im/s
Epoch 29, Train Loss: 10000.5644531, Time: 24.3641s, Throughput: 204.891497 im/s
Epoch 30, Iter 39, Loss: 9312.9755859, Throughput: 204.382565 im/s
Epoch 30, Train Loss: 10006.3876953, Time: 24.4249s, Throughput: 204.381326 im/s
Epoch 31, Iter 39, Loss: 9953.0507812, Throughput: 205.043402 im/s
Epoch 31, Train Loss: 9921.8945312, Time: 24.3462s, Throughput: 205.042097 im/s
Epoch 32, Iter 39, Loss: 9790.6757812, Throughput: 205.295650 im/s
Epoch 32, Train Loss: 9878.3574219, Time: 24.3163s, Throughput: 205.293985 im/s
Epoch 33, Iter 39, Loss: 10512.1289062, Throughput: 204.869936 im/s
Epoch 33, Train Loss: 9815.9013672, Time: 24.3668s, Throughput: 204.868549 im/s
Epoch 34, Iter 39, Loss: 9881.0839844, Throughput: 203.904851 im/s
Epoch 34, Train Loss: 9714.7900391, Time: 24.4822s, Throughput: 203.903410 im/s
Epoch 35, Iter 39, Loss: 10724.9013672, Throughput: 204.411346 im/s
Epoch 35, Train Loss: 9770.0986328, Time: 24.4215s, Throughput: 204.409981 im/s
Epoch 36, Iter 39, Loss: 9277.7714844, Throughput: 204.468798 im/s
Epoch 36, Train Loss: 9832.8994141, Time: 24.4146s, Throughput: 204.467466 im/s
Epoch 37, Iter 39, Loss: 9260.2851562, Throughput: 205.213872 im/s
Epoch 37, Train Loss: 9703.2656250, Time: 24.3260s, Throughput: 205.212614 im/s
Epoch 38, Iter 39, Loss: 8971.3300781, Throughput: 204.684722 im/s
Epoch 38, Train Loss: 9719.3906250, Time: 24.3889s, Throughput: 204.683354 im/s
Epoch 39, Iter 39, Loss: 9793.2050781, Throughput: 203.737623 im/s
Epoch 39, Train Loss: 9685.2519531, Time: 24.5022s, Throughput: 203.736426 im/s
Epoch 40, Iter 39, Loss: 9630.6523438, Throughput: 204.488523 im/s
Epoch 40, Train Loss: 9626.2695312, Time: 24.4123s, Throughput: 204.487285 im/s
Epoch 41, Iter 39, Loss: 10170.3222656, Throughput: 203.864864 im/s
Epoch 41, Train Loss: 9554.1347656, Time: 24.4870s, Throughput: 203.863517 im/s
Epoch 42, Iter 39, Loss: 10406.2285156, Throughput: 203.775456 im/s
Epoch 42, Train Loss: 9535.4707031, Time: 24.4977s, Throughput: 203.774175 im/s
Epoch 43, Iter 39, Loss: 9170.4375000, Throughput: 203.253399 im/s
Epoch 43, Train Loss: 9451.9023438, Time: 24.5606s, Throughput: 203.252140 im/s
Epoch 44, Iter 39, Loss: 9778.6562500, Throughput: 204.340024 im/s
Epoch 44, Train Loss: 9457.4394531, Time: 24.4300s, Throughput: 204.338744 im/s
Epoch 45, Iter 39, Loss: 9585.9033203, Throughput: 203.799755 im/s
Epoch 45, Train Loss: 9493.7880859, Time: 24.4948s, Throughput: 203.798424 im/s
Epoch 46, Iter 39, Loss: 9457.7382812, Throughput: 203.749138 im/s
Epoch 46, Train Loss: 9422.5468750, Time: 24.5009s, Throughput: 203.747516 im/s
Epoch 47, Iter 39, Loss: 9381.9365234, Throughput: 203.776544 im/s
Epoch 47, Train Loss: 9377.6992188, Time: 24.4976s, Throughput: 203.775206 im/s
Epoch 48, Iter 39, Loss: 10044.1367188, Throughput: 203.542871 im/s
Epoch 48, Train Loss: 9338.9951172, Time: 24.5257s, Throughput: 203.541359 im/s
Epoch 49, Iter 39, Loss: 9484.3496094, Throughput: 204.466621 im/s
Epoch 49, Train Loss: 9354.6972656, Time: 24.4149s, Throughput: 204.465357 im/s
Epoch 50, Iter 39, Loss: 9830.0068359, Throughput: 204.124409 im/s
Epoch 50, Train Loss: 9327.9218750, Time: 24.4558s, Throughput: 204.123048 im/s

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
