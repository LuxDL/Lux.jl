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
Epoch 1, Iter 39, Loss: 24142.0078125, Throughput: 16.407576 im/s
Epoch 1, Train Loss: 39597.8125000, Time: 304.5282s, Throughput: 16.392570 im/s
Epoch 2, Iter 39, Loss: 17447.4179688, Throughput: 174.434641 im/s
Epoch 2, Train Loss: 20273.4121094, Time: 28.6183s, Throughput: 174.433647 im/s
Epoch 3, Iter 39, Loss: 16171.2734375, Throughput: 173.329168 im/s
Epoch 3, Train Loss: 16550.8007812, Time: 28.8008s, Throughput: 173.328290 im/s
Epoch 4, Iter 39, Loss: 14555.8007812, Throughput: 174.123552 im/s
Epoch 4, Train Loss: 14968.9218750, Time: 28.6695s, Throughput: 174.122617 im/s
Epoch 5, Iter 39, Loss: 13447.1552734, Throughput: 174.209321 im/s
Epoch 5, Train Loss: 13914.6601562, Time: 28.6553s, Throughput: 174.208326 im/s
Epoch 6, Iter 39, Loss: 12499.2285156, Throughput: 174.739971 im/s
Epoch 6, Train Loss: 13452.6376953, Time: 28.5683s, Throughput: 174.738914 im/s
Epoch 7, Iter 39, Loss: 11622.3583984, Throughput: 173.978201 im/s
Epoch 7, Train Loss: 12862.4052734, Time: 28.6934s, Throughput: 173.977242 im/s
Epoch 8, Iter 39, Loss: 12752.5351562, Throughput: 174.581120 im/s
Epoch 8, Train Loss: 12447.3935547, Time: 28.5943s, Throughput: 174.580157 im/s
Epoch 9, Iter 39, Loss: 11823.6386719, Throughput: 173.799727 im/s
Epoch 9, Train Loss: 12155.6367188, Time: 28.7229s, Throughput: 173.798759 im/s
Epoch 10, Iter 39, Loss: 12304.8593750, Throughput: 174.508069 im/s
Epoch 10, Train Loss: 11990.7851562, Time: 28.6063s, Throughput: 174.507307 im/s
Epoch 11, Iter 39, Loss: 11239.1054688, Throughput: 173.944731 im/s
Epoch 11, Train Loss: 11789.0605469, Time: 28.6989s, Throughput: 173.943748 im/s
Epoch 12, Iter 39, Loss: 11446.9550781, Throughput: 174.456304 im/s
Epoch 12, Train Loss: 11499.4882812, Time: 28.6148s, Throughput: 174.455305 im/s
Epoch 13, Iter 39, Loss: 11117.1220703, Throughput: 174.424175 im/s
Epoch 13, Train Loss: 11351.8144531, Time: 28.6201s, Throughput: 174.423096 im/s
Epoch 14, Iter 39, Loss: 12049.8505859, Throughput: 174.244681 im/s
Epoch 14, Train Loss: 11234.5761719, Time: 28.6495s, Throughput: 174.243847 im/s
Epoch 15, Iter 39, Loss: 11230.8896484, Throughput: 173.841167 im/s
Epoch 15, Train Loss: 11037.2949219, Time: 28.7160s, Throughput: 173.840271 im/s
Epoch 16, Iter 39, Loss: 10565.6523438, Throughput: 174.459073 im/s
Epoch 16, Train Loss: 10886.8564453, Time: 28.6143s, Throughput: 174.458103 im/s
Epoch 17, Iter 39, Loss: 10405.8261719, Throughput: 173.797838 im/s
Epoch 17, Train Loss: 10891.6162109, Time: 28.7232s, Throughput: 173.796700 im/s
Epoch 18, Iter 39, Loss: 10006.4228516, Throughput: 174.495803 im/s
Epoch 18, Train Loss: 10846.7978516, Time: 28.6083s, Throughput: 174.494919 im/s
Epoch 19, Iter 39, Loss: 11261.8906250, Throughput: 174.396267 im/s
Epoch 19, Train Loss: 10678.8769531, Time: 28.6246s, Throughput: 174.395310 im/s
Epoch 20, Iter 39, Loss: 10084.6220703, Throughput: 173.913740 im/s
Epoch 20, Train Loss: 10557.1210938, Time: 28.7040s, Throughput: 173.912856 im/s
Epoch 21, Iter 39, Loss: 10567.0185547, Throughput: 171.523810 im/s
Epoch 21, Train Loss: 10480.5039062, Time: 29.1040s, Throughput: 171.522855 im/s
Epoch 22, Iter 39, Loss: 10767.3964844, Throughput: 174.079500 im/s
Epoch 22, Train Loss: 10398.1357422, Time: 28.6767s, Throughput: 174.078589 im/s
Epoch 23, Iter 39, Loss: 10583.1015625, Throughput: 174.307786 im/s
Epoch 23, Train Loss: 10380.0761719, Time: 28.6392s, Throughput: 174.306789 im/s
Epoch 24, Iter 39, Loss: 9938.6884766, Throughput: 174.042624 im/s
Epoch 24, Train Loss: 10282.1552734, Time: 28.6828s, Throughput: 174.041513 im/s
Epoch 25, Iter 39, Loss: 10108.7490234, Throughput: 174.485203 im/s
Epoch 25, Train Loss: 10233.1513672, Time: 28.6100s, Throughput: 174.484202 im/s
Epoch 26, Iter 39, Loss: 11202.3671875, Throughput: 174.106955 im/s
Epoch 26, Train Loss: 10237.4091797, Time: 28.6722s, Throughput: 174.105947 im/s
Epoch 27, Iter 39, Loss: 10160.1308594, Throughput: 174.730278 im/s
Epoch 27, Train Loss: 10175.7167969, Time: 28.5699s, Throughput: 174.729367 im/s
Epoch 28, Iter 39, Loss: 10351.2578125, Throughput: 173.488130 im/s
Epoch 28, Train Loss: 10058.1425781, Time: 28.7745s, Throughput: 173.487069 im/s
Epoch 29, Iter 39, Loss: 10204.4521484, Throughput: 173.458946 im/s
Epoch 29, Train Loss: 10027.8369141, Time: 28.7793s, Throughput: 173.458061 im/s
Epoch 30, Iter 39, Loss: 9103.4042969, Throughput: 173.450822 im/s
Epoch 30, Train Loss: 9962.2294922, Time: 28.7807s, Throughput: 173.449797 im/s
Epoch 31, Iter 39, Loss: 9656.0371094, Throughput: 174.230699 im/s
Epoch 31, Train Loss: 9875.2587891, Time: 28.6518s, Throughput: 174.229757 im/s
Epoch 32, Iter 39, Loss: 9930.6542969, Throughput: 174.066291 im/s
Epoch 32, Train Loss: 9952.4570312, Time: 28.6789s, Throughput: 174.065307 im/s
Epoch 33, Iter 39, Loss: 10179.3115234, Throughput: 174.465786 im/s
Epoch 33, Train Loss: 9928.4130859, Time: 28.6132s, Throughput: 174.464786 im/s
Epoch 34, Iter 39, Loss: 9498.6455078, Throughput: 173.779737 im/s
Epoch 34, Train Loss: 9797.0634766, Time: 28.7262s, Throughput: 173.778762 im/s
Epoch 35, Iter 39, Loss: 9042.5693359, Throughput: 174.251389 im/s
Epoch 35, Train Loss: 9759.7851562, Time: 28.6484s, Throughput: 174.250496 im/s
Epoch 36, Iter 39, Loss: 9992.4697266, Throughput: 174.427034 im/s
Epoch 36, Train Loss: 9642.3457031, Time: 28.6196s, Throughput: 174.426095 im/s
Epoch 37, Iter 39, Loss: 8827.7949219, Throughput: 174.836189 im/s
Epoch 37, Train Loss: 9660.4482422, Time: 28.5526s, Throughput: 174.835338 im/s
Epoch 38, Iter 39, Loss: 9951.7500000, Throughput: 173.625827 im/s
Epoch 38, Train Loss: 9658.4531250, Time: 28.7517s, Throughput: 173.624715 im/s
Epoch 39, Iter 39, Loss: 9608.7304688, Throughput: 174.662234 im/s
Epoch 39, Train Loss: 9620.5380859, Time: 28.5810s, Throughput: 174.661220 im/s
Epoch 40, Iter 39, Loss: 10544.0585938, Throughput: 174.538876 im/s
Epoch 40, Train Loss: 9631.1894531, Time: 28.6013s, Throughput: 174.537624 im/s
Epoch 41, Iter 39, Loss: 9640.0917969, Throughput: 173.843123 im/s
Epoch 41, Train Loss: 9627.2343750, Time: 28.7157s, Throughput: 173.842215 im/s
Epoch 42, Iter 39, Loss: 9913.2558594, Throughput: 174.578387 im/s
Epoch 42, Train Loss: 9577.8388672, Time: 28.5948s, Throughput: 174.577471 im/s
Epoch 43, Iter 39, Loss: 8727.6074219, Throughput: 174.281303 im/s
Epoch 43, Train Loss: 9453.7177734, Time: 28.6435s, Throughput: 174.280268 im/s
Epoch 44, Iter 39, Loss: 10056.1640625, Throughput: 174.264542 im/s
Epoch 44, Train Loss: 9485.9355469, Time: 28.6463s, Throughput: 174.263525 im/s
Epoch 45, Iter 39, Loss: 9928.4873047, Throughput: 174.053637 im/s
Epoch 45, Train Loss: 9470.4423828, Time: 28.6810s, Throughput: 174.052636 im/s
Epoch 46, Iter 39, Loss: 9447.5585938, Throughput: 174.169583 im/s
Epoch 46, Train Loss: 9407.7685547, Time: 28.6619s, Throughput: 174.168563 im/s
Epoch 47, Iter 39, Loss: 10215.7167969, Throughput: 173.631769 im/s
Epoch 47, Train Loss: 9353.5263672, Time: 28.7507s, Throughput: 173.630820 im/s
Epoch 48, Iter 39, Loss: 9458.0292969, Throughput: 174.089492 im/s
Epoch 48, Train Loss: 9368.3134766, Time: 28.6751s, Throughput: 174.088272 im/s
Epoch 49, Iter 39, Loss: 9640.6728516, Throughput: 170.925819 im/s
Epoch 49, Train Loss: 9359.5576172, Time: 29.2058s, Throughput: 170.924865 im/s
Epoch 50, Iter 39, Loss: 8639.2587891, Throughput: 174.485850 im/s
Epoch 50, Train Loss: 9339.5156250, Time: 28.6099s, Throughput: 174.484813 im/s

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
