---
url: /previews/PR1853/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24188.5781250, Throughput: 21.058128 im/s
Epoch 1, Train Loss: 39756.5625000, Time: 237.2568s, Throughput: 21.040496 im/s
Epoch 2, Iter 39, Loss: 16986.6621094, Throughput: 206.206025 im/s
Epoch 2, Train Loss: 20079.2558594, Time: 24.2090s, Throughput: 206.204535 im/s
Epoch 3, Iter 39, Loss: 15187.6816406, Throughput: 206.085474 im/s
Epoch 3, Train Loss: 16518.8886719, Time: 24.2231s, Throughput: 206.084182 im/s
Epoch 4, Iter 39, Loss: 14907.1953125, Throughput: 205.294001 im/s
Epoch 4, Train Loss: 15048.4345703, Time: 24.3165s, Throughput: 205.292743 im/s
Epoch 5, Iter 39, Loss: 14128.9052734, Throughput: 205.424650 im/s
Epoch 5, Train Loss: 14086.3281250, Time: 24.3010s, Throughput: 205.423408 im/s
Epoch 6, Iter 39, Loss: 14198.4179688, Throughput: 206.373229 im/s
Epoch 6, Train Loss: 13481.2031250, Time: 24.1893s, Throughput: 206.371846 im/s
Epoch 7, Iter 39, Loss: 12688.0000000, Throughput: 206.168740 im/s
Epoch 7, Train Loss: 13040.9980469, Time: 24.2133s, Throughput: 206.167413 im/s
Epoch 8, Iter 39, Loss: 11789.1640625, Throughput: 206.463379 im/s
Epoch 8, Train Loss: 12610.7236328, Time: 24.1788s, Throughput: 206.462131 im/s
Epoch 9, Iter 39, Loss: 11839.9804688, Throughput: 205.964954 im/s
Epoch 9, Train Loss: 12215.1259766, Time: 24.2373s, Throughput: 205.963484 im/s
Epoch 10, Iter 39, Loss: 12148.6503906, Throughput: 206.562745 im/s
Epoch 10, Train Loss: 11995.3984375, Time: 24.1671s, Throughput: 206.561488 im/s
Epoch 11, Iter 39, Loss: 12035.2714844, Throughput: 205.975237 im/s
Epoch 11, Train Loss: 11976.8876953, Time: 24.2361s, Throughput: 205.973920 im/s
Epoch 12, Iter 39, Loss: 11831.1074219, Throughput: 206.169431 im/s
Epoch 12, Train Loss: 11618.0292969, Time: 24.2133s, Throughput: 206.168042 im/s
Epoch 13, Iter 39, Loss: 11229.0507812, Throughput: 206.041846 im/s
Epoch 13, Train Loss: 11458.8857422, Time: 24.2282s, Throughput: 206.040587 im/s
Epoch 14, Iter 39, Loss: 11440.0468750, Throughput: 205.361388 im/s
Epoch 14, Train Loss: 11218.9179688, Time: 24.3085s, Throughput: 205.359926 im/s
Epoch 15, Iter 39, Loss: 10515.1464844, Throughput: 204.184380 im/s
Epoch 15, Train Loss: 11105.9619141, Time: 24.4486s, Throughput: 204.183219 im/s
Epoch 16, Iter 39, Loss: 10559.6289062, Throughput: 206.503504 im/s
Epoch 16, Train Loss: 11052.3144531, Time: 24.1741s, Throughput: 206.502213 im/s
Epoch 17, Iter 39, Loss: 10530.0087891, Throughput: 206.420138 im/s
Epoch 17, Train Loss: 10862.8623047, Time: 24.1838s, Throughput: 206.418831 im/s
Epoch 18, Iter 39, Loss: 11218.8330078, Throughput: 206.150613 im/s
Epoch 18, Train Loss: 10788.2431641, Time: 24.2155s, Throughput: 206.149319 im/s
Epoch 19, Iter 39, Loss: 10706.4746094, Throughput: 206.291308 im/s
Epoch 19, Train Loss: 10743.0068359, Time: 24.1989s, Throughput: 206.290070 im/s
Epoch 20, Iter 39, Loss: 10430.9492188, Throughput: 206.249209 im/s
Epoch 20, Train Loss: 10700.2070312, Time: 24.2039s, Throughput: 206.247771 im/s
Epoch 21, Iter 39, Loss: 10014.0722656, Throughput: 206.331969 im/s
Epoch 21, Train Loss: 10479.8271484, Time: 24.1942s, Throughput: 206.330731 im/s
Epoch 22, Iter 39, Loss: 10339.5771484, Throughput: 206.052274 im/s
Epoch 22, Train Loss: 10488.5976562, Time: 24.2270s, Throughput: 206.051049 im/s
Epoch 23, Iter 39, Loss: 10461.2421875, Throughput: 206.464002 im/s
Epoch 23, Train Loss: 10426.2294922, Time: 24.1787s, Throughput: 206.462660 im/s
Epoch 24, Iter 39, Loss: 10674.9794922, Throughput: 206.279246 im/s
Epoch 24, Train Loss: 10363.4267578, Time: 24.2004s, Throughput: 206.277567 im/s
Epoch 25, Iter 39, Loss: 9858.0478516, Throughput: 205.588515 im/s
Epoch 25, Train Loss: 10299.1591797, Time: 24.2817s, Throughput: 205.587261 im/s
Epoch 26, Iter 39, Loss: 10243.8437500, Throughput: 205.760689 im/s
Epoch 26, Train Loss: 10186.3046875, Time: 24.2613s, Throughput: 205.759468 im/s
Epoch 27, Iter 39, Loss: 10162.9414062, Throughput: 205.633281 im/s
Epoch 27, Train Loss: 10118.4160156, Time: 24.2764s, Throughput: 205.632111 im/s
Epoch 28, Iter 39, Loss: 10114.7714844, Throughput: 206.206834 im/s
Epoch 28, Train Loss: 10002.5644531, Time: 24.2089s, Throughput: 206.205581 im/s
Epoch 29, Iter 39, Loss: 9612.7177734, Throughput: 205.982947 im/s
Epoch 29, Train Loss: 10053.8144531, Time: 24.2352s, Throughput: 205.981578 im/s
Epoch 30, Iter 39, Loss: 10467.5371094, Throughput: 206.303957 im/s
Epoch 30, Train Loss: 9979.4218750, Time: 24.1975s, Throughput: 206.302678 im/s
Epoch 31, Iter 39, Loss: 9913.3945312, Throughput: 205.912298 im/s
Epoch 31, Train Loss: 9950.0117188, Time: 24.2435s, Throughput: 205.911025 im/s
Epoch 32, Iter 39, Loss: 10917.1220703, Throughput: 206.425824 im/s
Epoch 32, Train Loss: 9930.2675781, Time: 24.1832s, Throughput: 206.424491 im/s
Epoch 33, Iter 39, Loss: 9968.7265625, Throughput: 206.269197 im/s
Epoch 33, Train Loss: 9855.2148438, Time: 24.2015s, Throughput: 206.267900 im/s
Epoch 34, Iter 39, Loss: 9764.6708984, Throughput: 205.512960 im/s
Epoch 34, Train Loss: 9843.7197266, Time: 24.2906s, Throughput: 205.511588 im/s
Epoch 35, Iter 39, Loss: 9459.6669922, Throughput: 203.211019 im/s
Epoch 35, Train Loss: 9757.1601562, Time: 24.5657s, Throughput: 203.209778 im/s
Epoch 36, Iter 39, Loss: 9504.2187500, Throughput: 206.012341 im/s
Epoch 36, Train Loss: 9750.4023438, Time: 24.2317s, Throughput: 206.010871 im/s
Epoch 37, Iter 39, Loss: 10027.0605469, Throughput: 206.134237 im/s
Epoch 37, Train Loss: 9769.6601562, Time: 24.2174s, Throughput: 206.132918 im/s
Epoch 38, Iter 39, Loss: 10176.8339844, Throughput: 205.618137 im/s
Epoch 38, Train Loss: 9704.6806641, Time: 24.2782s, Throughput: 205.616841 im/s
Epoch 39, Iter 39, Loss: 9201.8339844, Throughput: 206.219186 im/s
Epoch 39, Train Loss: 9652.5117188, Time: 24.2074s, Throughput: 206.217900 im/s
Epoch 40, Iter 39, Loss: 10001.1484375, Throughput: 206.194867 im/s
Epoch 40, Train Loss: 9621.9335938, Time: 24.2103s, Throughput: 206.193589 im/s
Epoch 41, Iter 39, Loss: 8895.7578125, Throughput: 206.166483 im/s
Epoch 41, Train Loss: 9514.5917969, Time: 24.2136s, Throughput: 206.165249 im/s
Epoch 42, Iter 39, Loss: 10314.5976562, Throughput: 206.339995 im/s
Epoch 42, Train Loss: 9499.8984375, Time: 24.1932s, Throughput: 206.338580 im/s
Epoch 43, Iter 39, Loss: 9409.6992188, Throughput: 206.466683 im/s
Epoch 43, Train Loss: 9499.4521484, Time: 24.1784s, Throughput: 206.465454 im/s
Epoch 44, Iter 39, Loss: 9926.9238281, Throughput: 205.920545 im/s
Epoch 44, Train Loss: 9510.9335938, Time: 24.2425s, Throughput: 205.919289 im/s
Epoch 45, Iter 39, Loss: 9703.4326172, Throughput: 206.286635 im/s
Epoch 45, Train Loss: 9418.4375000, Time: 24.1995s, Throughput: 206.285469 im/s
Epoch 46, Iter 39, Loss: 9762.0673828, Throughput: 205.788622 im/s
Epoch 46, Train Loss: 9470.5068359, Time: 24.2581s, Throughput: 205.787341 im/s
Epoch 47, Iter 39, Loss: 9987.2675781, Throughput: 206.430637 im/s
Epoch 47, Train Loss: 9417.2197266, Time: 24.1826s, Throughput: 206.429392 im/s
Epoch 48, Iter 39, Loss: 9793.8388672, Throughput: 205.688455 im/s
Epoch 48, Train Loss: 9337.2197266, Time: 24.2699s, Throughput: 205.686762 im/s
Epoch 49, Iter 39, Loss: 9158.1269531, Throughput: 205.588329 im/s
Epoch 49, Train Loss: 9318.6181641, Time: 24.2817s, Throughput: 205.586941 im/s
Epoch 50, Iter 39, Loss: 9967.7470703, Throughput: 206.110756 im/s
Epoch 50, Train Loss: 9309.2617188, Time: 24.2201s, Throughput: 206.109479 im/s

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
