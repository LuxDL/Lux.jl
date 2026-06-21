---
url: /previews/PR1698/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24011.8125000, Throughput: 4.232602 im/s
Epoch 1, Train Loss: 39791.4023438, Time: 1179.6989s, Throughput: 4.231588 im/s
Epoch 2, Iter 39, Loss: 17742.1093750, Throughput: 70.440694 im/s
Epoch 2, Train Loss: 20255.7773438, Time: 70.8683s, Throughput: 70.440502 im/s
Epoch 3, Iter 39, Loss: 14810.2851562, Throughput: 69.999224 im/s
Epoch 3, Train Loss: 16618.9335938, Time: 71.3153s, Throughput: 69.999034 im/s
Epoch 4, Iter 39, Loss: 14508.8964844, Throughput: 70.172763 im/s
Epoch 4, Train Loss: 15112.3652344, Time: 71.1389s, Throughput: 70.172598 im/s
Epoch 5, Iter 39, Loss: 13247.2929688, Throughput: 70.079320 im/s
Epoch 5, Train Loss: 14115.2392578, Time: 71.2338s, Throughput: 70.079105 im/s
Epoch 6, Iter 39, Loss: 12714.4033203, Throughput: 70.150103 im/s
Epoch 6, Train Loss: 13442.0800781, Time: 71.1619s, Throughput: 70.149930 im/s
Epoch 7, Iter 39, Loss: 13376.1914062, Throughput: 68.581702 im/s
Epoch 7, Train Loss: 12968.0107422, Time: 72.7893s, Throughput: 68.581522 im/s
Epoch 8, Iter 39, Loss: 11517.4794922, Throughput: 70.729252 im/s
Epoch 8, Train Loss: 12555.0390625, Time: 70.5792s, Throughput: 70.729077 im/s
Epoch 9, Iter 39, Loss: 12086.4980469, Throughput: 70.005396 im/s
Epoch 9, Train Loss: 12350.7080078, Time: 71.3090s, Throughput: 70.005157 im/s
Epoch 10, Iter 39, Loss: 12620.8935547, Throughput: 70.447950 im/s
Epoch 10, Train Loss: 12064.6386719, Time: 70.8610s, Throughput: 70.447782 im/s
Epoch 11, Iter 39, Loss: 11253.2343750, Throughput: 69.403147 im/s
Epoch 11, Train Loss: 11917.2480469, Time: 71.9278s, Throughput: 69.402958 im/s
Epoch 12, Iter 39, Loss: 11492.1650391, Throughput: 70.121444 im/s
Epoch 12, Train Loss: 11624.9160156, Time: 71.1910s, Throughput: 70.121236 im/s
Epoch 13, Iter 39, Loss: 11303.6679688, Throughput: 70.154639 im/s
Epoch 13, Train Loss: 11530.3300781, Time: 71.1573s, Throughput: 70.154450 im/s
Epoch 14, Iter 39, Loss: 11591.0761719, Throughput: 69.765318 im/s
Epoch 14, Train Loss: 11312.0263672, Time: 71.5544s, Throughput: 69.765146 im/s
Epoch 15, Iter 39, Loss: 11147.9335938, Throughput: 69.026017 im/s
Epoch 15, Train Loss: 11188.1806641, Time: 72.3208s, Throughput: 69.025832 im/s
Epoch 16, Iter 39, Loss: 11382.3642578, Throughput: 69.799299 im/s
Epoch 16, Train Loss: 11033.9375000, Time: 71.5195s, Throughput: 69.799119 im/s
Epoch 17, Iter 39, Loss: 10759.7949219, Throughput: 70.396174 im/s
Epoch 17, Train Loss: 10987.0214844, Time: 70.9131s, Throughput: 70.395979 im/s
Epoch 18, Iter 39, Loss: 10490.4179688, Throughput: 70.047151 im/s
Epoch 18, Train Loss: 10885.4257812, Time: 71.2665s, Throughput: 70.046966 im/s
Epoch 19, Iter 39, Loss: 10833.5615234, Throughput: 69.835358 im/s
Epoch 19, Train Loss: 10747.3369141, Time: 71.4826s, Throughput: 69.835182 im/s
Epoch 20, Iter 39, Loss: 10967.6982422, Throughput: 69.957920 im/s
Epoch 20, Train Loss: 10738.6152344, Time: 71.3574s, Throughput: 69.957716 im/s
Epoch 21, Iter 39, Loss: 10765.7304688, Throughput: 69.975108 im/s
Epoch 21, Train Loss: 10571.2568359, Time: 71.3398s, Throughput: 69.974925 im/s
Epoch 22, Iter 39, Loss: 10177.3769531, Throughput: 69.638570 im/s
Epoch 22, Train Loss: 10491.5058594, Time: 71.6846s, Throughput: 69.638396 im/s
Epoch 23, Iter 39, Loss: 10254.1533203, Throughput: 70.158286 im/s
Epoch 23, Train Loss: 10420.4931641, Time: 71.1536s, Throughput: 70.158097 im/s
Epoch 24, Iter 39, Loss: 11026.5927734, Throughput: 69.756041 im/s
Epoch 24, Train Loss: 10340.9931641, Time: 71.5639s, Throughput: 69.755837 im/s
Epoch 25, Iter 39, Loss: 10635.7597656, Throughput: 69.975796 im/s
Epoch 25, Train Loss: 10323.4414062, Time: 71.3391s, Throughput: 69.975606 im/s
Epoch 26, Iter 39, Loss: 10136.6074219, Throughput: 70.159786 im/s
Epoch 26, Train Loss: 10279.9414062, Time: 71.1521s, Throughput: 70.159581 im/s
Epoch 27, Iter 39, Loss: 10619.1210938, Throughput: 70.355963 im/s
Epoch 27, Train Loss: 10159.9980469, Time: 70.9537s, Throughput: 70.355743 im/s
Epoch 28, Iter 39, Loss: 10132.7783203, Throughput: 69.809403 im/s
Epoch 28, Train Loss: 10147.0664062, Time: 71.5092s, Throughput: 69.809211 im/s
Epoch 29, Iter 39, Loss: 10007.3378906, Throughput: 69.890188 im/s
Epoch 29, Train Loss: 10106.4736328, Time: 71.4265s, Throughput: 69.889981 im/s
Epoch 30, Iter 39, Loss: 10497.1904297, Throughput: 69.567296 im/s
Epoch 30, Train Loss: 10027.5244141, Time: 71.7580s, Throughput: 69.567130 im/s
Epoch 31, Iter 39, Loss: 10424.7890625, Throughput: 70.174814 im/s
Epoch 31, Train Loss: 10022.3906250, Time: 71.1368s, Throughput: 70.174616 im/s
Epoch 32, Iter 39, Loss: 9882.9121094, Throughput: 69.879522 im/s
Epoch 32, Train Loss: 9922.4306641, Time: 71.4374s, Throughput: 69.879333 im/s
Epoch 33, Iter 39, Loss: 9984.2197266, Throughput: 69.898995 im/s
Epoch 33, Train Loss: 9872.2197266, Time: 71.4175s, Throughput: 69.898821 im/s
Epoch 34, Iter 39, Loss: 9512.2236328, Throughput: 69.747149 im/s
Epoch 34, Train Loss: 9880.5351562, Time: 71.5730s, Throughput: 69.746966 im/s
Epoch 35, Iter 39, Loss: 9685.8984375, Throughput: 69.441293 im/s
Epoch 35, Train Loss: 9773.3935547, Time: 71.8883s, Throughput: 69.441110 im/s
Epoch 36, Iter 39, Loss: 9843.2304688, Throughput: 69.998108 im/s
Epoch 36, Train Loss: 9732.1083984, Time: 71.3164s, Throughput: 69.997938 im/s
Epoch 37, Iter 39, Loss: 9806.4257812, Throughput: 69.940864 im/s
Epoch 37, Train Loss: 9774.1025391, Time: 71.3748s, Throughput: 69.940673 im/s
Epoch 38, Iter 39, Loss: 9395.5527344, Throughput: 69.530292 im/s
Epoch 38, Train Loss: 9660.3330078, Time: 71.7962s, Throughput: 69.530110 im/s
Epoch 39, Iter 39, Loss: 9113.3398438, Throughput: 69.479401 im/s
Epoch 39, Train Loss: 9646.8730469, Time: 71.8488s, Throughput: 69.479233 im/s
Epoch 40, Iter 39, Loss: 10219.8320312, Throughput: 69.979399 im/s
Epoch 40, Train Loss: 9661.8701172, Time: 71.3355s, Throughput: 69.979208 im/s
Epoch 41, Iter 39, Loss: 9689.9218750, Throughput: 69.365387 im/s
Epoch 41, Train Loss: 9596.1748047, Time: 71.9669s, Throughput: 69.365203 im/s
Epoch 42, Iter 39, Loss: 9044.1542969, Throughput: 70.311626 im/s
Epoch 42, Train Loss: 9566.1035156, Time: 70.9984s, Throughput: 70.311434 im/s
Epoch 43, Iter 39, Loss: 9780.5039062, Throughput: 69.941419 im/s
Epoch 43, Train Loss: 9580.5126953, Time: 71.3742s, Throughput: 69.941218 im/s
Epoch 44, Iter 39, Loss: 9624.0097656, Throughput: 69.860732 im/s
Epoch 44, Train Loss: 9485.1220703, Time: 71.4566s, Throughput: 69.860546 im/s
Epoch 45, Iter 39, Loss: 9495.5761719, Throughput: 70.395974 im/s
Epoch 45, Train Loss: 9496.0253906, Time: 70.9134s, Throughput: 70.395771 im/s
Epoch 46, Iter 39, Loss: 9700.1972656, Throughput: 70.418514 im/s
Epoch 46, Train Loss: 9479.0449219, Time: 70.8906s, Throughput: 70.418333 im/s
Epoch 47, Iter 39, Loss: 9682.1826172, Throughput: 70.200815 im/s
Epoch 47, Train Loss: 9444.2392578, Time: 71.1104s, Throughput: 70.200658 im/s
Epoch 48, Iter 39, Loss: 9512.1660156, Throughput: 69.847681 im/s
Epoch 48, Train Loss: 9372.6777344, Time: 71.4700s, Throughput: 69.847508 im/s
Epoch 49, Iter 39, Loss: 9309.6445312, Throughput: 70.615257 im/s
Epoch 49, Train Loss: 9373.5751953, Time: 70.6931s, Throughput: 70.615048 im/s
Epoch 50, Iter 39, Loss: 9241.4404297, Throughput: 70.194145 im/s
Epoch 50, Train Loss: 9305.9453125, Time: 71.1172s, Throughput: 70.193975 im/s

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
