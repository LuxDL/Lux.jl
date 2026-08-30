---
url: /previews/PR1713/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23522.8710938, Throughput: 10.182568 im/s
Epoch 1, Train Loss: 39459.6328125, Time: 490.5711s, Throughput: 10.175894 im/s
Epoch 2, Iter 39, Loss: 16667.2187500, Throughput: 72.899915 im/s
Epoch 2, Train Loss: 20206.7890625, Time: 68.4776s, Throughput: 72.899703 im/s
Epoch 3, Iter 39, Loss: 15893.7753906, Throughput: 71.818690 im/s
Epoch 3, Train Loss: 16616.3632812, Time: 69.5086s, Throughput: 71.818464 im/s
Epoch 4, Iter 39, Loss: 14955.1113281, Throughput: 72.218368 im/s
Epoch 4, Train Loss: 14945.4931641, Time: 69.1239s, Throughput: 72.218154 im/s
Epoch 5, Iter 39, Loss: 12761.4609375, Throughput: 72.764532 im/s
Epoch 5, Train Loss: 14001.2744141, Time: 68.6050s, Throughput: 72.764339 im/s
Epoch 6, Iter 39, Loss: 13606.8105469, Throughput: 72.391631 im/s
Epoch 6, Train Loss: 13324.0556641, Time: 68.9584s, Throughput: 72.391431 im/s
Epoch 7, Iter 39, Loss: 13373.5859375, Throughput: 72.643125 im/s
Epoch 7, Train Loss: 13010.4902344, Time: 68.7198s, Throughput: 72.642854 im/s
Epoch 8, Iter 39, Loss: 12909.1660156, Throughput: 72.465741 im/s
Epoch 8, Train Loss: 12640.8457031, Time: 68.8879s, Throughput: 72.465537 im/s
Epoch 9, Iter 39, Loss: 12542.6054688, Throughput: 72.427281 im/s
Epoch 9, Train Loss: 12190.1054688, Time: 68.9245s, Throughput: 72.427076 im/s
Epoch 10, Iter 39, Loss: 12052.7724609, Throughput: 72.353912 im/s
Epoch 10, Train Loss: 11885.0781250, Time: 68.9944s, Throughput: 72.353672 im/s
Epoch 11, Iter 39, Loss: 11508.1845703, Throughput: 71.973056 im/s
Epoch 11, Train Loss: 11642.1904297, Time: 69.3595s, Throughput: 71.972851 im/s
Epoch 12, Iter 39, Loss: 11909.5283203, Throughput: 72.159800 im/s
Epoch 12, Train Loss: 11534.8115234, Time: 69.1800s, Throughput: 72.159602 im/s
Epoch 13, Iter 39, Loss: 11429.1601562, Throughput: 72.529945 im/s
Epoch 13, Train Loss: 11339.8144531, Time: 68.8269s, Throughput: 72.529773 im/s
Epoch 14, Iter 39, Loss: 11010.3906250, Throughput: 72.318715 im/s
Epoch 14, Train Loss: 11266.7031250, Time: 69.0280s, Throughput: 72.318521 im/s
Epoch 15, Iter 39, Loss: 10432.0390625, Throughput: 72.905045 im/s
Epoch 15, Train Loss: 11049.4697266, Time: 68.4728s, Throughput: 72.904849 im/s
Epoch 16, Iter 39, Loss: 10154.1972656, Throughput: 72.382585 im/s
Epoch 16, Train Loss: 11027.9169922, Time: 68.9671s, Throughput: 72.382382 im/s
Epoch 17, Iter 39, Loss: 10561.3710938, Throughput: 72.486581 im/s
Epoch 17, Train Loss: 10841.7402344, Time: 68.8681s, Throughput: 72.486396 im/s
Epoch 18, Iter 39, Loss: 10846.2910156, Throughput: 72.926125 im/s
Epoch 18, Train Loss: 10739.9316406, Time: 68.4530s, Throughput: 72.925944 im/s
Epoch 19, Iter 39, Loss: 10672.6044922, Throughput: 71.826436 im/s
Epoch 19, Train Loss: 10646.8740234, Time: 69.5011s, Throughput: 71.826241 im/s
Epoch 20, Iter 39, Loss: 11026.0976562, Throughput: 72.677143 im/s
Epoch 20, Train Loss: 10552.3906250, Time: 68.6875s, Throughput: 72.676938 im/s
Epoch 21, Iter 39, Loss: 10177.2148438, Throughput: 72.597310 im/s
Epoch 21, Train Loss: 10478.7597656, Time: 68.7631s, Throughput: 72.597114 im/s
Epoch 22, Iter 39, Loss: 10695.5996094, Throughput: 72.687324 im/s
Epoch 22, Train Loss: 10437.0234375, Time: 68.6779s, Throughput: 72.687121 im/s
Epoch 23, Iter 39, Loss: 10753.4824219, Throughput: 72.134447 im/s
Epoch 23, Train Loss: 10352.7431641, Time: 69.2043s, Throughput: 72.134258 im/s
Epoch 24, Iter 39, Loss: 10660.5566406, Throughput: 72.157254 im/s
Epoch 24, Train Loss: 10323.8818359, Time: 69.1824s, Throughput: 72.157029 im/s
Epoch 25, Iter 39, Loss: 10185.5966797, Throughput: 72.091483 im/s
Epoch 25, Train Loss: 10240.5917969, Time: 69.2455s, Throughput: 72.091299 im/s
Epoch 26, Iter 39, Loss: 10500.6191406, Throughput: 72.277408 im/s
Epoch 26, Train Loss: 10127.2509766, Time: 69.0674s, Throughput: 72.277227 im/s
Epoch 27, Iter 39, Loss: 10075.7031250, Throughput: 72.151310 im/s
Epoch 27, Train Loss: 10147.4570312, Time: 69.1881s, Throughput: 72.151109 im/s
Epoch 28, Iter 39, Loss: 9805.7333984, Throughput: 71.225995 im/s
Epoch 28, Train Loss: 10063.1816406, Time: 70.0870s, Throughput: 71.225813 im/s
Epoch 29, Iter 39, Loss: 9931.0488281, Throughput: 71.712875 im/s
Epoch 29, Train Loss: 10082.0996094, Time: 69.6111s, Throughput: 71.712705 im/s
Epoch 30, Iter 39, Loss: 10381.6621094, Throughput: 72.044570 im/s
Epoch 30, Train Loss: 9984.6289062, Time: 69.2906s, Throughput: 72.044389 im/s
Epoch 31, Iter 39, Loss: 9862.4843750, Throughput: 72.005747 im/s
Epoch 31, Train Loss: 9915.3398438, Time: 69.3280s, Throughput: 72.005531 im/s
Epoch 32, Iter 39, Loss: 9861.5488281, Throughput: 71.612398 im/s
Epoch 32, Train Loss: 9878.1230469, Time: 69.7088s, Throughput: 71.612220 im/s
Epoch 33, Iter 39, Loss: 9880.8017578, Throughput: 72.262298 im/s
Epoch 33, Train Loss: 9918.3505859, Time: 69.0818s, Throughput: 72.262109 im/s
Epoch 34, Iter 39, Loss: 9692.8300781, Throughput: 71.650121 im/s
Epoch 34, Train Loss: 9806.9160156, Time: 69.6721s, Throughput: 71.649947 im/s
Epoch 35, Iter 39, Loss: 10467.8134766, Throughput: 71.986989 im/s
Epoch 35, Train Loss: 9865.0468750, Time: 69.3460s, Throughput: 71.986808 im/s
Epoch 36, Iter 39, Loss: 10000.9765625, Throughput: 71.611334 im/s
Epoch 36, Train Loss: 9709.3847656, Time: 69.7098s, Throughput: 71.611159 im/s
Epoch 37, Iter 39, Loss: 10126.1386719, Throughput: 72.186554 im/s
Epoch 37, Train Loss: 9686.7080078, Time: 69.1544s, Throughput: 72.186313 im/s
Epoch 38, Iter 39, Loss: 9806.5507812, Throughput: 72.548769 im/s
Epoch 38, Train Loss: 9628.0761719, Time: 68.8091s, Throughput: 72.548574 im/s
Epoch 39, Iter 39, Loss: 9784.7207031, Throughput: 72.485572 im/s
Epoch 39, Train Loss: 9642.5156250, Time: 68.8690s, Throughput: 72.485400 im/s
Epoch 40, Iter 39, Loss: 9030.3105469, Throughput: 72.719860 im/s
Epoch 40, Train Loss: 9590.6054688, Time: 68.6472s, Throughput: 72.719670 im/s
Epoch 41, Iter 39, Loss: 9894.9785156, Throughput: 72.601332 im/s
Epoch 41, Train Loss: 9523.6435547, Time: 68.7593s, Throughput: 72.601138 im/s
Epoch 42, Iter 39, Loss: 9500.8154297, Throughput: 72.279284 im/s
Epoch 42, Train Loss: 9546.3691406, Time: 69.0656s, Throughput: 72.279092 im/s
Epoch 43, Iter 39, Loss: 8985.6240234, Throughput: 72.430226 im/s
Epoch 43, Train Loss: 9509.1230469, Time: 68.9217s, Throughput: 72.430029 im/s
Epoch 44, Iter 39, Loss: 10395.6035156, Throughput: 72.066322 im/s
Epoch 44, Train Loss: 9449.2744141, Time: 69.2697s, Throughput: 72.066127 im/s
Epoch 45, Iter 39, Loss: 9337.0839844, Throughput: 72.557978 im/s
Epoch 45, Train Loss: 9458.6376953, Time: 68.8003s, Throughput: 72.557787 im/s
Epoch 46, Iter 39, Loss: 9713.7343750, Throughput: 72.077344 im/s
Epoch 46, Train Loss: 9364.3681641, Time: 69.2591s, Throughput: 72.077146 im/s
Epoch 47, Iter 39, Loss: 9106.0976562, Throughput: 72.049485 im/s
Epoch 47, Train Loss: 9404.0371094, Time: 69.2860s, Throughput: 72.049232 im/s
Epoch 48, Iter 39, Loss: 8911.7490234, Throughput: 72.286319 im/s
Epoch 48, Train Loss: 9336.4423828, Time: 69.0589s, Throughput: 72.286138 im/s
Epoch 49, Iter 39, Loss: 9321.3789062, Throughput: 72.893791 im/s
Epoch 49, Train Loss: 9356.6884766, Time: 68.4834s, Throughput: 72.893566 im/s
Epoch 50, Iter 39, Loss: 9088.9511719, Throughput: 72.022217 im/s
Epoch 50, Train Loss: 9279.6044922, Time: 69.3121s, Throughput: 72.022032 im/s

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
Julia Version 1.12.6
Commit 15346901f00 (2026-04-09 19:20 UTC)
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
