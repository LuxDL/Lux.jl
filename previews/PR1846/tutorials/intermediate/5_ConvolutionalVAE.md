---
url: /previews/PR1846/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24383.4414062, Throughput: 16.624047 im/s
Epoch 1, Train Loss: 39816.4765625, Time: 300.5539s, Throughput: 16.609334 im/s
Epoch 2, Iter 39, Loss: 18173.5234375, Throughput: 172.697262 im/s
Epoch 2, Train Loss: 20388.8984375, Time: 28.9062s, Throughput: 172.696228 im/s
Epoch 3, Iter 39, Loss: 16931.2675781, Throughput: 174.063973 im/s
Epoch 3, Train Loss: 16847.9746094, Time: 28.6793s, Throughput: 174.062996 im/s
Epoch 4, Iter 39, Loss: 14241.0312500, Throughput: 173.548322 im/s
Epoch 4, Train Loss: 15244.3076172, Time: 28.7645s, Throughput: 173.547400 im/s
Epoch 5, Iter 39, Loss: 13896.3779297, Throughput: 173.980110 im/s
Epoch 5, Train Loss: 14306.2294922, Time: 28.6931s, Throughput: 173.979201 im/s
Epoch 6, Iter 39, Loss: 12564.5996094, Throughput: 173.659367 im/s
Epoch 6, Train Loss: 13608.0380859, Time: 28.7461s, Throughput: 173.658431 im/s
Epoch 7, Iter 39, Loss: 12303.8593750, Throughput: 174.434487 im/s
Epoch 7, Train Loss: 13062.2119141, Time: 28.6183s, Throughput: 174.433628 im/s
Epoch 8, Iter 39, Loss: 12216.2558594, Throughput: 173.567976 im/s
Epoch 8, Train Loss: 12592.5068359, Time: 28.7612s, Throughput: 173.566969 im/s
Epoch 9, Iter 39, Loss: 11594.7392578, Throughput: 174.227701 im/s
Epoch 9, Train Loss: 12379.9482422, Time: 28.6523s, Throughput: 174.226795 im/s
Epoch 10, Iter 39, Loss: 13000.7988281, Throughput: 173.606385 im/s
Epoch 10, Train Loss: 12116.0683594, Time: 28.7549s, Throughput: 173.605202 im/s
Epoch 11, Iter 39, Loss: 12189.7314453, Throughput: 174.150736 im/s
Epoch 11, Train Loss: 11877.4912109, Time: 28.6650s, Throughput: 174.149369 im/s
Epoch 12, Iter 39, Loss: 12225.6894531, Throughput: 174.205138 im/s
Epoch 12, Train Loss: 11649.7919922, Time: 28.6560s, Throughput: 174.204148 im/s
Epoch 13, Iter 39, Loss: 11743.9873047, Throughput: 173.459333 im/s
Epoch 13, Train Loss: 11539.4892578, Time: 28.7792s, Throughput: 173.458417 im/s
Epoch 14, Iter 39, Loss: 11543.8515625, Throughput: 174.005805 im/s
Epoch 14, Train Loss: 11388.1660156, Time: 28.6888s, Throughput: 174.004901 im/s
Epoch 15, Iter 39, Loss: 11613.3476562, Throughput: 173.814970 im/s
Epoch 15, Train Loss: 11137.7802734, Time: 28.7203s, Throughput: 173.814129 im/s
Epoch 16, Iter 39, Loss: 10953.4902344, Throughput: 174.038928 im/s
Epoch 16, Train Loss: 11170.2744141, Time: 28.6834s, Throughput: 174.038001 im/s
Epoch 17, Iter 39, Loss: 11064.7138672, Throughput: 172.763956 im/s
Epoch 17, Train Loss: 11070.6357422, Time: 28.8951s, Throughput: 172.763035 im/s
Epoch 18, Iter 39, Loss: 10919.4609375, Throughput: 173.877117 im/s
Epoch 18, Train Loss: 10863.4179688, Time: 28.7101s, Throughput: 173.876178 im/s
Epoch 19, Iter 39, Loss: 11622.9365234, Throughput: 171.640405 im/s
Epoch 19, Train Loss: 10807.1386719, Time: 29.0842s, Throughput: 171.639496 im/s
Epoch 20, Iter 39, Loss: 10753.4042969, Throughput: 174.109318 im/s
Epoch 20, Train Loss: 10717.7714844, Time: 28.6718s, Throughput: 174.108394 im/s
Epoch 21, Iter 39, Loss: 10959.3359375, Throughput: 173.574669 im/s
Epoch 21, Train Loss: 10654.8691406, Time: 28.7601s, Throughput: 173.573697 im/s
Epoch 22, Iter 39, Loss: 10504.0517578, Throughput: 174.169328 im/s
Epoch 22, Train Loss: 10525.8017578, Time: 28.6619s, Throughput: 174.168410 im/s
Epoch 23, Iter 39, Loss: 10621.2441406, Throughput: 173.945034 im/s
Epoch 23, Train Loss: 10414.3750000, Time: 28.6989s, Throughput: 173.944125 im/s
Epoch 24, Iter 39, Loss: 9800.4335938, Throughput: 173.990001 im/s
Epoch 24, Train Loss: 10429.0244141, Time: 28.6915s, Throughput: 173.988903 im/s
Epoch 25, Iter 39, Loss: 10375.2773438, Throughput: 173.454438 im/s
Epoch 25, Train Loss: 10326.5361328, Time: 28.7800s, Throughput: 173.453516 im/s
Epoch 26, Iter 39, Loss: 10692.6796875, Throughput: 173.896401 im/s
Epoch 26, Train Loss: 10323.2929688, Time: 28.7069s, Throughput: 173.895415 im/s
Epoch 27, Iter 39, Loss: 10590.0244141, Throughput: 173.959660 im/s
Epoch 27, Train Loss: 10251.5927734, Time: 28.6965s, Throughput: 173.958715 im/s
Epoch 28, Iter 39, Loss: 10498.9062500, Throughput: 174.217214 im/s
Epoch 28, Train Loss: 10220.7509766, Time: 28.6540s, Throughput: 174.216289 im/s
Epoch 29, Iter 39, Loss: 10277.7519531, Throughput: 174.139655 im/s
Epoch 29, Train Loss: 10189.1103516, Time: 28.6668s, Throughput: 174.138768 im/s
Epoch 30, Iter 39, Loss: 9687.2148438, Throughput: 173.542651 im/s
Epoch 30, Train Loss: 10112.9980469, Time: 28.7654s, Throughput: 173.541661 im/s
Epoch 31, Iter 39, Loss: 10196.5048828, Throughput: 172.793224 im/s
Epoch 31, Train Loss: 10064.8554688, Time: 28.8902s, Throughput: 172.792230 im/s
Epoch 32, Iter 39, Loss: 9934.4316406, Throughput: 173.887909 im/s
Epoch 32, Train Loss: 9969.1425781, Time: 28.7083s, Throughput: 173.887037 im/s
Epoch 33, Iter 39, Loss: 9703.2343750, Throughput: 173.979219 im/s
Epoch 33, Train Loss: 9920.4423828, Time: 28.6932s, Throughput: 173.978237 im/s
Epoch 34, Iter 39, Loss: 9624.5527344, Throughput: 174.358872 im/s
Epoch 34, Train Loss: 9913.3486328, Time: 28.6308s, Throughput: 174.357940 im/s
Epoch 35, Iter 39, Loss: 9898.5488281, Throughput: 174.052370 im/s
Epoch 35, Train Loss: 9824.4843750, Time: 28.6812s, Throughput: 174.051150 im/s
Epoch 36, Iter 39, Loss: 10017.8095703, Throughput: 173.929241 im/s
Epoch 36, Train Loss: 9810.0996094, Time: 28.7015s, Throughput: 173.928380 im/s
Epoch 37, Iter 39, Loss: 10257.9062500, Throughput: 174.037661 im/s
Epoch 37, Train Loss: 9824.6718750, Time: 28.6836s, Throughput: 174.036593 im/s
Epoch 38, Iter 39, Loss: 10172.4755859, Throughput: 173.812302 im/s
Epoch 38, Train Loss: 9738.2597656, Time: 28.7208s, Throughput: 173.811345 im/s
Epoch 39, Iter 39, Loss: 10028.4902344, Throughput: 174.506891 im/s
Epoch 39, Train Loss: 9698.5400391, Time: 28.6065s, Throughput: 174.505946 im/s
Epoch 40, Iter 39, Loss: 10017.6376953, Throughput: 174.020296 im/s
Epoch 40, Train Loss: 9617.3369141, Time: 28.6864s, Throughput: 174.019466 im/s
Epoch 41, Iter 39, Loss: 9691.6474609, Throughput: 172.042643 im/s
Epoch 41, Train Loss: 9637.4277344, Time: 29.0162s, Throughput: 172.041694 im/s
Epoch 42, Iter 39, Loss: 9882.3984375, Throughput: 173.316621 im/s
Epoch 42, Train Loss: 9607.1435547, Time: 28.8030s, Throughput: 173.315555 im/s
Epoch 43, Iter 39, Loss: 10115.5791016, Throughput: 173.808803 im/s
Epoch 43, Train Loss: 9570.7724609, Time: 28.7214s, Throughput: 173.807908 im/s
Epoch 44, Iter 39, Loss: 9796.5654297, Throughput: 173.687161 im/s
Epoch 44, Train Loss: 9573.0996094, Time: 28.7415s, Throughput: 173.686189 im/s
Epoch 45, Iter 39, Loss: 9555.6777344, Throughput: 173.872669 im/s
Epoch 45, Train Loss: 9591.5615234, Time: 28.7108s, Throughput: 173.871738 im/s
Epoch 46, Iter 39, Loss: 9808.4736328, Throughput: 173.693669 im/s
Epoch 46, Train Loss: 9513.1972656, Time: 28.7404s, Throughput: 173.692805 im/s
Epoch 47, Iter 39, Loss: 9925.3076172, Throughput: 173.793342 im/s
Epoch 47, Train Loss: 9478.0156250, Time: 28.7239s, Throughput: 173.792387 im/s
Epoch 48, Iter 39, Loss: 9536.7148438, Throughput: 173.281379 im/s
Epoch 48, Train Loss: 9487.9814453, Time: 28.8088s, Throughput: 173.280212 im/s
Epoch 49, Iter 39, Loss: 9810.2910156, Throughput: 174.042708 im/s
Epoch 49, Train Loss: 9428.3251953, Time: 28.6828s, Throughput: 174.041658 im/s
Epoch 50, Iter 39, Loss: 9391.1533203, Throughput: 174.117223 im/s
Epoch 50, Train Loss: 9365.7060547, Time: 28.6705s, Throughput: 174.116312 im/s

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
