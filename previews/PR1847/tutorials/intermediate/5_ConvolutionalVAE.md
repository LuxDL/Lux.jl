---
url: /previews/PR1847/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23072.3359375, Throughput: 20.185728 im/s
Epoch 1, Train Loss: 39498.3750000, Time: 247.5008s, Throughput: 20.169628 im/s
Epoch 2, Iter 39, Loss: 16697.4648438, Throughput: 120.799583 im/s
Epoch 2, Train Loss: 19927.1679688, Time: 41.3248s, Throughput: 120.799009 im/s
Epoch 3, Iter 39, Loss: 14562.9667969, Throughput: 121.565909 im/s
Epoch 3, Train Loss: 16431.8281250, Time: 41.0643s, Throughput: 121.565394 im/s
Epoch 4, Iter 39, Loss: 15075.6513672, Throughput: 121.348812 im/s
Epoch 4, Train Loss: 14824.7128906, Time: 41.1378s, Throughput: 121.348252 im/s
Epoch 5, Iter 39, Loss: 13237.1660156, Throughput: 121.441197 im/s
Epoch 5, Train Loss: 13949.9746094, Time: 41.1065s, Throughput: 121.440736 im/s
Epoch 6, Iter 39, Loss: 12879.2812500, Throughput: 121.714637 im/s
Epoch 6, Train Loss: 13293.7617188, Time: 41.0142s, Throughput: 121.714035 im/s
Epoch 7, Iter 39, Loss: 12168.9941406, Throughput: 121.470611 im/s
Epoch 7, Train Loss: 12885.2294922, Time: 41.0965s, Throughput: 121.470100 im/s
Epoch 8, Iter 39, Loss: 11592.5410156, Throughput: 121.279749 im/s
Epoch 8, Train Loss: 12520.4931641, Time: 41.1612s, Throughput: 121.279260 im/s
Epoch 9, Iter 39, Loss: 12372.5615234, Throughput: 121.172837 im/s
Epoch 9, Train Loss: 12235.8759766, Time: 41.1976s, Throughput: 121.172242 im/s
Epoch 10, Iter 39, Loss: 11624.2480469, Throughput: 121.760998 im/s
Epoch 10, Train Loss: 11893.5468750, Time: 40.9985s, Throughput: 121.760520 im/s
Epoch 11, Iter 39, Loss: 12422.9882812, Throughput: 121.512039 im/s
Epoch 11, Train Loss: 11758.8115234, Time: 41.0825s, Throughput: 121.511489 im/s
Epoch 12, Iter 39, Loss: 11360.3886719, Throughput: 121.400347 im/s
Epoch 12, Train Loss: 11435.1552734, Time: 41.1203s, Throughput: 121.399786 im/s
Epoch 13, Iter 39, Loss: 12039.8691406, Throughput: 122.009397 im/s
Epoch 13, Train Loss: 11422.0917969, Time: 40.9151s, Throughput: 122.008890 im/s
Epoch 14, Iter 39, Loss: 10904.6093750, Throughput: 121.904506 im/s
Epoch 14, Train Loss: 11184.0107422, Time: 40.9503s, Throughput: 121.903941 im/s
Epoch 15, Iter 39, Loss: 11574.9375000, Throughput: 121.685256 im/s
Epoch 15, Train Loss: 11078.9843750, Time: 41.0241s, Throughput: 121.684666 im/s
Epoch 16, Iter 39, Loss: 10894.5781250, Throughput: 122.115083 im/s
Epoch 16, Train Loss: 10947.3427734, Time: 40.8796s, Throughput: 122.114569 im/s
Epoch 17, Iter 39, Loss: 10717.1484375, Throughput: 120.663126 im/s
Epoch 17, Train Loss: 10869.6015625, Time: 41.3716s, Throughput: 120.662603 im/s
Epoch 18, Iter 39, Loss: 10481.8964844, Throughput: 122.231282 im/s
Epoch 18, Train Loss: 10774.1640625, Time: 40.8408s, Throughput: 122.230725 im/s
Epoch 19, Iter 39, Loss: 11383.7431641, Throughput: 122.659020 im/s
Epoch 19, Train Loss: 10642.2783203, Time: 40.6984s, Throughput: 122.658475 im/s
Epoch 20, Iter 39, Loss: 9933.1640625, Throughput: 122.465601 im/s
Epoch 20, Train Loss: 10677.0156250, Time: 40.7627s, Throughput: 122.464964 im/s
Epoch 21, Iter 39, Loss: 11392.5888672, Throughput: 122.497978 im/s
Epoch 21, Train Loss: 10554.2626953, Time: 40.7518s, Throughput: 122.497530 im/s
Epoch 22, Iter 39, Loss: 10803.1289062, Throughput: 122.717943 im/s
Epoch 22, Train Loss: 10426.5927734, Time: 40.6788s, Throughput: 122.717345 im/s
Epoch 23, Iter 39, Loss: 10551.1347656, Throughput: 122.838644 im/s
Epoch 23, Train Loss: 10339.9697266, Time: 40.6388s, Throughput: 122.838158 im/s
Epoch 24, Iter 39, Loss: 10158.9648438, Throughput: 123.140853 im/s
Epoch 24, Train Loss: 10376.9550781, Time: 40.5402s, Throughput: 123.137159 im/s
Epoch 25, Iter 39, Loss: 10740.9306641, Throughput: 123.118184 im/s
Epoch 25, Train Loss: 10226.4394531, Time: 40.5466s, Throughput: 123.117564 im/s
Epoch 26, Iter 39, Loss: 10815.9921875, Throughput: 122.748628 im/s
Epoch 26, Train Loss: 10198.6943359, Time: 40.6686s, Throughput: 122.748121 im/s
Epoch 27, Iter 39, Loss: 10265.6025391, Throughput: 123.042288 im/s
Epoch 27, Train Loss: 10089.3359375, Time: 40.5716s, Throughput: 123.041773 im/s
Epoch 28, Iter 39, Loss: 10763.0683594, Throughput: 123.136078 im/s
Epoch 28, Train Loss: 10087.9960938, Time: 40.5407s, Throughput: 123.135610 im/s
Epoch 29, Iter 39, Loss: 10036.5468750, Throughput: 122.985317 im/s
Epoch 29, Train Loss: 10032.2080078, Time: 40.5904s, Throughput: 122.984848 im/s
Epoch 30, Iter 39, Loss: 10575.1054688, Throughput: 122.980306 im/s
Epoch 30, Train Loss: 10046.8300781, Time: 40.5920s, Throughput: 122.979767 im/s
Epoch 31, Iter 39, Loss: 9942.4931641, Throughput: 122.926147 im/s
Epoch 31, Train Loss: 9875.3544922, Time: 40.6100s, Throughput: 122.925494 im/s
Epoch 32, Iter 39, Loss: 10356.0087891, Throughput: 123.096212 im/s
Epoch 32, Train Loss: 9961.4531250, Time: 40.5538s, Throughput: 123.095782 im/s
Epoch 33, Iter 39, Loss: 9723.1894531, Throughput: 122.794891 im/s
Epoch 33, Train Loss: 9913.4843750, Time: 40.6533s, Throughput: 122.794405 im/s
Epoch 34, Iter 39, Loss: 9280.8027344, Throughput: 121.594288 im/s
Epoch 34, Train Loss: 9787.6455078, Time: 41.0547s, Throughput: 121.593749 im/s
Epoch 35, Iter 39, Loss: 9735.7998047, Throughput: 123.005670 im/s
Epoch 35, Train Loss: 9690.8037109, Time: 40.5837s, Throughput: 123.005088 im/s
Epoch 36, Iter 39, Loss: 9773.7119141, Throughput: 122.879652 im/s
Epoch 36, Train Loss: 9712.6562500, Time: 40.6253s, Throughput: 122.879020 im/s
Epoch 37, Iter 39, Loss: 9889.4755859, Throughput: 122.635508 im/s
Epoch 37, Train Loss: 9685.0986328, Time: 40.7061s, Throughput: 122.635050 im/s
Epoch 38, Iter 39, Loss: 9323.9296875, Throughput: 122.696528 im/s
Epoch 38, Train Loss: 9664.6523438, Time: 40.6859s, Throughput: 122.696015 im/s
Epoch 39, Iter 39, Loss: 9706.4003906, Throughput: 122.988081 im/s
Epoch 39, Train Loss: 9646.0156250, Time: 40.5895s, Throughput: 122.987594 im/s
Epoch 40, Iter 39, Loss: 9439.4492188, Throughput: 122.913441 im/s
Epoch 40, Train Loss: 9564.2880859, Time: 40.6141s, Throughput: 122.912852 im/s
Epoch 41, Iter 39, Loss: 8972.6972656, Throughput: 122.476441 im/s
Epoch 41, Train Loss: 9583.4384766, Time: 40.7591s, Throughput: 122.475835 im/s
Epoch 42, Iter 39, Loss: 9505.1367188, Throughput: 123.000708 im/s
Epoch 42, Train Loss: 9492.0693359, Time: 40.5853s, Throughput: 123.000117 im/s
Epoch 43, Iter 39, Loss: 8939.9824219, Throughput: 122.708232 im/s
Epoch 43, Train Loss: 9475.8007812, Time: 40.6820s, Throughput: 122.707725 im/s
Epoch 44, Iter 39, Loss: 9303.2187500, Throughput: 122.852466 im/s
Epoch 44, Train Loss: 9531.8671875, Time: 40.6343s, Throughput: 122.851915 im/s
Epoch 45, Iter 39, Loss: 9644.8437500, Throughput: 122.152023 im/s
Epoch 45, Train Loss: 9464.4394531, Time: 40.8683s, Throughput: 122.148573 im/s
Epoch 46, Iter 39, Loss: 9625.7324219, Throughput: 122.570106 im/s
Epoch 46, Train Loss: 9379.7060547, Time: 40.7279s, Throughput: 122.569544 im/s
Epoch 47, Iter 39, Loss: 9401.1289062, Throughput: 122.860309 im/s
Epoch 47, Train Loss: 9414.1708984, Time: 40.6317s, Throughput: 122.859801 im/s
Epoch 48, Iter 39, Loss: 10170.9101562, Throughput: 122.754970 im/s
Epoch 48, Train Loss: 9385.6074219, Time: 40.6666s, Throughput: 122.754251 im/s
Epoch 49, Iter 39, Loss: 9434.7773438, Throughput: 123.008195 im/s
Epoch 49, Train Loss: 9373.3496094, Time: 40.5828s, Throughput: 123.007744 im/s
Epoch 50, Iter 39, Loss: 10002.9394531, Throughput: 122.896136 im/s
Epoch 50, Train Loss: 9297.1181641, Time: 40.6198s, Throughput: 122.895582 im/s

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
  CPU: 4 × Intel(R) Xeon(R) 6975P-C
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, graniterapids)
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
