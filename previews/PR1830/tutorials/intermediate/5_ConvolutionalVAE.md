---
url: /previews/PR1830/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24410.0332031, Throughput: 16.403282 im/s
Epoch 1, Train Loss: 39636.0468750, Time: 304.5958s, Throughput: 16.388934 im/s
Epoch 2, Iter 39, Loss: 18708.5156250, Throughput: 173.735659 im/s
Epoch 2, Train Loss: 20244.9960938, Time: 28.7335s, Throughput: 173.734716 im/s
Epoch 3, Iter 39, Loss: 15086.4902344, Throughput: 173.779669 im/s
Epoch 3, Train Loss: 16725.0703125, Time: 28.7262s, Throughput: 173.778701 im/s
Epoch 4, Iter 39, Loss: 15348.2324219, Throughput: 174.373136 im/s
Epoch 4, Train Loss: 15161.0576172, Time: 28.6284s, Throughput: 174.372022 im/s
Epoch 5, Iter 39, Loss: 14157.1611328, Throughput: 173.381108 im/s
Epoch 5, Train Loss: 14276.9755859, Time: 28.7922s, Throughput: 173.380139 im/s
Epoch 6, Iter 39, Loss: 13745.2304688, Throughput: 174.405382 im/s
Epoch 6, Train Loss: 13569.2773438, Time: 28.6231s, Throughput: 174.404480 im/s
Epoch 7, Iter 39, Loss: 12039.3203125, Throughput: 172.811078 im/s
Epoch 7, Train Loss: 13010.9042969, Time: 28.8872s, Throughput: 172.810169 im/s
Epoch 8, Iter 39, Loss: 13215.0742188, Throughput: 174.211413 im/s
Epoch 8, Train Loss: 12641.3564453, Time: 28.6550s, Throughput: 174.210471 im/s
Epoch 9, Iter 39, Loss: 12737.6083984, Throughput: 173.211623 im/s
Epoch 9, Train Loss: 12432.9472656, Time: 28.8204s, Throughput: 173.210601 im/s
Epoch 10, Iter 39, Loss: 12425.6328125, Throughput: 173.946391 im/s
Epoch 10, Train Loss: 12135.4892578, Time: 28.6986s, Throughput: 173.945513 im/s
Epoch 11, Iter 39, Loss: 12140.8964844, Throughput: 173.920174 im/s
Epoch 11, Train Loss: 11915.3457031, Time: 28.7030s, Throughput: 173.919218 im/s
Epoch 12, Iter 39, Loss: 12218.4707031, Throughput: 174.938320 im/s
Epoch 12, Train Loss: 11719.1376953, Time: 28.5359s, Throughput: 174.937376 im/s
Epoch 13, Iter 39, Loss: 11350.9609375, Throughput: 174.197212 im/s
Epoch 13, Train Loss: 11493.0517578, Time: 28.6574s, Throughput: 174.196112 im/s
Epoch 14, Iter 39, Loss: 11236.0605469, Throughput: 173.754639 im/s
Epoch 14, Train Loss: 11428.4609375, Time: 28.7303s, Throughput: 173.753805 im/s
Epoch 15, Iter 39, Loss: 11045.6933594, Throughput: 173.750741 im/s
Epoch 15, Train Loss: 11348.8642578, Time: 28.7310s, Throughput: 173.749645 im/s
Epoch 16, Iter 39, Loss: 10839.2763672, Throughput: 174.177604 im/s
Epoch 16, Train Loss: 11168.7207031, Time: 28.6606s, Throughput: 174.176529 im/s
Epoch 17, Iter 39, Loss: 10244.3496094, Throughput: 174.753827 im/s
Epoch 17, Train Loss: 11039.5195312, Time: 28.5661s, Throughput: 174.752836 im/s
Epoch 18, Iter 39, Loss: 11252.6181641, Throughput: 173.691657 im/s
Epoch 18, Train Loss: 10975.8115234, Time: 28.7407s, Throughput: 173.690743 im/s
Epoch 19, Iter 39, Loss: 10697.7597656, Throughput: 172.061103 im/s
Epoch 19, Train Loss: 10907.7519531, Time: 29.0132s, Throughput: 172.059846 im/s
Epoch 20, Iter 39, Loss: 11559.2001953, Throughput: 173.928786 im/s
Epoch 20, Train Loss: 10835.5332031, Time: 28.7016s, Throughput: 173.927817 im/s
Epoch 21, Iter 39, Loss: 11258.1972656, Throughput: 174.300197 im/s
Epoch 21, Train Loss: 10622.4580078, Time: 28.6404s, Throughput: 174.299235 im/s
Epoch 22, Iter 39, Loss: 10825.7597656, Throughput: 173.171490 im/s
Epoch 22, Train Loss: 10594.1279297, Time: 28.8271s, Throughput: 173.170602 im/s
Epoch 23, Iter 39, Loss: 10158.8085938, Throughput: 175.064520 im/s
Epoch 23, Train Loss: 10538.3681641, Time: 28.5154s, Throughput: 175.063588 im/s
Epoch 24, Iter 39, Loss: 10617.5000000, Throughput: 173.452533 im/s
Epoch 24, Train Loss: 10450.3271484, Time: 28.7804s, Throughput: 173.451431 im/s
Epoch 25, Iter 39, Loss: 10682.8750000, Throughput: 174.466110 im/s
Epoch 25, Train Loss: 10456.4267578, Time: 28.6132s, Throughput: 174.465055 im/s
Epoch 26, Iter 39, Loss: 10290.1425781, Throughput: 174.389090 im/s
Epoch 26, Train Loss: 10303.8857422, Time: 28.6258s, Throughput: 174.387963 im/s
Epoch 27, Iter 39, Loss: 11017.9785156, Throughput: 174.680269 im/s
Epoch 27, Train Loss: 10304.0634766, Time: 28.5781s, Throughput: 174.679249 im/s
Epoch 28, Iter 39, Loss: 10581.3066406, Throughput: 174.319845 im/s
Epoch 28, Train Loss: 10194.8632812, Time: 28.6372s, Throughput: 174.318895 im/s
Epoch 29, Iter 39, Loss: 10738.3017578, Throughput: 174.583936 im/s
Epoch 29, Train Loss: 10147.5917969, Time: 28.5938s, Throughput: 174.583020 im/s
Epoch 30, Iter 39, Loss: 11695.2910156, Throughput: 173.449931 im/s
Epoch 30, Train Loss: 10118.1455078, Time: 28.7808s, Throughput: 173.448846 im/s
Epoch 31, Iter 39, Loss: 10288.5683594, Throughput: 174.717882 im/s
Epoch 31, Train Loss: 10107.8955078, Time: 28.5719s, Throughput: 174.716915 im/s
Epoch 32, Iter 39, Loss: 10547.5683594, Throughput: 174.037139 im/s
Epoch 32, Train Loss: 9966.4511719, Time: 28.6837s, Throughput: 174.035949 im/s
Epoch 33, Iter 39, Loss: 9120.1904297, Throughput: 174.500933 im/s
Epoch 33, Train Loss: 9982.2177734, Time: 28.6075s, Throughput: 174.499896 im/s
Epoch 34, Iter 39, Loss: 10228.7792969, Throughput: 173.929913 im/s
Epoch 34, Train Loss: 9910.4042969, Time: 28.7014s, Throughput: 173.928955 im/s
Epoch 35, Iter 39, Loss: 10638.8828125, Throughput: 174.552443 im/s
Epoch 35, Train Loss: 9826.9941406, Time: 28.5990s, Throughput: 174.551460 im/s
Epoch 36, Iter 39, Loss: 10085.7509766, Throughput: 173.725072 im/s
Epoch 36, Train Loss: 9808.4287109, Time: 28.7352s, Throughput: 173.724075 im/s
Epoch 37, Iter 39, Loss: 9877.4726562, Throughput: 174.181099 im/s
Epoch 37, Train Loss: 9758.3291016, Time: 28.6600s, Throughput: 174.180193 im/s
Epoch 38, Iter 39, Loss: 9707.7236328, Throughput: 172.198598 im/s
Epoch 38, Train Loss: 9859.6386719, Time: 28.9899s, Throughput: 172.197678 im/s
Epoch 39, Iter 39, Loss: 9448.7304688, Throughput: 174.697655 im/s
Epoch 39, Train Loss: 9762.8085938, Time: 28.5752s, Throughput: 174.696677 im/s
Epoch 40, Iter 39, Loss: 9123.9248047, Throughput: 174.380159 im/s
Epoch 40, Train Loss: 9641.8740234, Time: 28.6273s, Throughput: 174.379202 im/s
Epoch 41, Iter 39, Loss: 9037.7763672, Throughput: 173.514095 im/s
Epoch 41, Train Loss: 9625.8798828, Time: 28.7702s, Throughput: 173.513124 im/s
Epoch 42, Iter 39, Loss: 9861.6132812, Throughput: 174.319340 im/s
Epoch 42, Train Loss: 9580.6474609, Time: 28.6372s, Throughput: 174.318427 im/s
Epoch 43, Iter 39, Loss: 9583.1884766, Throughput: 174.111375 im/s
Epoch 43, Train Loss: 9638.0156250, Time: 28.6715s, Throughput: 174.110451 im/s
Epoch 44, Iter 39, Loss: 10043.9101562, Throughput: 174.682196 im/s
Epoch 44, Train Loss: 9556.5527344, Time: 28.5778s, Throughput: 174.681248 im/s
Epoch 45, Iter 39, Loss: 9756.2773438, Throughput: 174.178820 im/s
Epoch 45, Train Loss: 9697.5009766, Time: 28.6603s, Throughput: 174.177939 im/s
Epoch 46, Iter 39, Loss: 9293.6972656, Throughput: 174.356581 im/s
Epoch 46, Train Loss: 9585.7412109, Time: 28.6312s, Throughput: 174.355468 im/s
Epoch 47, Iter 39, Loss: 9458.2109375, Throughput: 173.877200 im/s
Epoch 47, Train Loss: 9448.5195312, Time: 28.7101s, Throughput: 173.876099 im/s
Epoch 48, Iter 39, Loss: 9662.8281250, Throughput: 174.139788 im/s
Epoch 48, Train Loss: 9426.6582031, Time: 28.6668s, Throughput: 174.138428 im/s
Epoch 49, Iter 39, Loss: 9233.0175781, Throughput: 174.331411 im/s
Epoch 49, Train Loss: 9342.0664062, Time: 28.6353s, Throughput: 174.330480 im/s
Epoch 50, Iter 39, Loss: 9241.0732422, Throughput: 174.494266 im/s
Epoch 50, Train Loss: 9356.9306641, Time: 28.6085s, Throughput: 174.493321 im/s

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
  CPU: 4 × AMD EPYC 7763 64-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver3)
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
