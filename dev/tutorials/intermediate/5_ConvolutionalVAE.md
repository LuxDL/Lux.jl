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
Epoch 1, Iter 39, Loss: 23709.7363281, Throughput: 4.784546 im/s
Epoch 1, Train Loss: 39612.0546875, Time: 1043.6660s, Throughput: 4.783139 im/s
Epoch 2, Iter 39, Loss: 17915.8554688, Throughput: 72.694600 im/s
Epoch 2, Train Loss: 20221.9160156, Time: 68.6710s, Throughput: 72.694431 im/s
Epoch 3, Iter 39, Loss: 15977.2890625, Throughput: 73.156202 im/s
Epoch 3, Train Loss: 16558.5039062, Time: 68.2377s, Throughput: 73.156022 im/s
Epoch 4, Iter 39, Loss: 14957.2451172, Throughput: 73.299175 im/s
Epoch 4, Train Loss: 15081.5156250, Time: 68.1046s, Throughput: 73.299002 im/s
Epoch 5, Iter 39, Loss: 14716.1982422, Throughput: 73.253993 im/s
Epoch 5, Train Loss: 14144.2275391, Time: 68.1466s, Throughput: 73.253814 im/s
Epoch 6, Iter 39, Loss: 13682.9921875, Throughput: 72.784752 im/s
Epoch 6, Train Loss: 13422.0468750, Time: 68.5860s, Throughput: 72.784561 im/s
Epoch 7, Iter 39, Loss: 13254.1308594, Throughput: 73.010247 im/s
Epoch 7, Train Loss: 12926.5566406, Time: 68.3741s, Throughput: 73.010074 im/s
Epoch 8, Iter 39, Loss: 12135.5410156, Throughput: 73.290202 im/s
Epoch 8, Train Loss: 12712.0839844, Time: 68.1129s, Throughput: 73.290029 im/s
Epoch 9, Iter 39, Loss: 11748.8076172, Throughput: 73.324228 im/s
Epoch 9, Train Loss: 12273.1484375, Time: 68.0813s, Throughput: 73.324054 im/s
Epoch 10, Iter 39, Loss: 12347.8359375, Throughput: 72.959865 im/s
Epoch 10, Train Loss: 12006.8642578, Time: 68.4213s, Throughput: 72.959702 im/s
Epoch 11, Iter 39, Loss: 12061.1171875, Throughput: 72.755392 im/s
Epoch 11, Train Loss: 11914.9843750, Time: 68.6136s, Throughput: 72.755224 im/s
Epoch 12, Iter 39, Loss: 11044.7832031, Throughput: 73.321919 im/s
Epoch 12, Train Loss: 11757.7968750, Time: 68.0835s, Throughput: 73.321749 im/s
Epoch 13, Iter 39, Loss: 11601.7451172, Throughput: 73.178894 im/s
Epoch 13, Train Loss: 11543.7744141, Time: 68.2165s, Throughput: 73.178728 im/s
Epoch 14, Iter 39, Loss: 11414.5185547, Throughput: 73.162515 im/s
Epoch 14, Train Loss: 11364.8916016, Time: 68.2318s, Throughput: 73.162341 im/s
Epoch 15, Iter 39, Loss: 11854.3183594, Throughput: 72.732060 im/s
Epoch 15, Train Loss: 11315.4755859, Time: 68.6357s, Throughput: 72.731865 im/s
Epoch 16, Iter 39, Loss: 10780.8847656, Throughput: 72.819364 im/s
Epoch 16, Train Loss: 11122.2490234, Time: 68.5533s, Throughput: 72.819199 im/s
Epoch 17, Iter 39, Loss: 11141.4804688, Throughput: 72.884859 im/s
Epoch 17, Train Loss: 11098.9814453, Time: 68.4918s, Throughput: 72.884678 im/s
Epoch 18, Iter 39, Loss: 11228.5107422, Throughput: 73.187646 im/s
Epoch 18, Train Loss: 10955.6142578, Time: 68.2084s, Throughput: 73.187469 im/s
Epoch 19, Iter 39, Loss: 10776.5146484, Throughput: 72.834019 im/s
Epoch 19, Train Loss: 10839.1357422, Time: 68.5396s, Throughput: 72.833846 im/s
Epoch 20, Iter 39, Loss: 11239.9755859, Throughput: 73.278053 im/s
Epoch 20, Train Loss: 10718.1621094, Time: 68.1243s, Throughput: 73.277843 im/s
Epoch 21, Iter 39, Loss: 10491.0517578, Throughput: 72.986327 im/s
Epoch 21, Train Loss: 10681.5498047, Time: 68.3965s, Throughput: 72.986180 im/s
Epoch 22, Iter 39, Loss: 10995.9570312, Throughput: 73.135147 im/s
Epoch 22, Train Loss: 10568.6777344, Time: 68.2574s, Throughput: 73.134969 im/s
Epoch 23, Iter 39, Loss: 10259.7753906, Throughput: 72.818659 im/s
Epoch 23, Train Loss: 10513.3105469, Time: 68.5540s, Throughput: 72.818487 im/s
Epoch 24, Iter 39, Loss: 10564.4990234, Throughput: 72.435619 im/s
Epoch 24, Train Loss: 10460.9570312, Time: 68.9166s, Throughput: 72.435422 im/s
Epoch 25, Iter 39, Loss: 10099.8417969, Throughput: 72.762994 im/s
Epoch 25, Train Loss: 10417.5673828, Time: 68.6065s, Throughput: 72.762815 im/s
Epoch 26, Iter 39, Loss: 10421.3701172, Throughput: 71.920607 im/s
Epoch 26, Train Loss: 10291.8417969, Time: 69.4100s, Throughput: 71.920439 im/s
Epoch 27, Iter 39, Loss: 10332.1542969, Throughput: 72.470533 im/s
Epoch 27, Train Loss: 10170.1171875, Time: 68.8833s, Throughput: 72.470354 im/s
Epoch 28, Iter 39, Loss: 10414.2109375, Throughput: 72.731638 im/s
Epoch 28, Train Loss: 10211.5253906, Time: 68.6360s, Throughput: 72.731469 im/s
Epoch 29, Iter 39, Loss: 10229.8828125, Throughput: 74.036006 im/s
Epoch 29, Train Loss: 10152.9482422, Time: 67.4268s, Throughput: 74.035829 im/s
Epoch 30, Iter 39, Loss: 10022.5488281, Throughput: 73.097046 im/s
Epoch 30, Train Loss: 10097.3505859, Time: 68.2929s, Throughput: 73.096862 im/s
Epoch 31, Iter 39, Loss: 9969.2412109, Throughput: 72.781774 im/s
Epoch 31, Train Loss: 10056.9335938, Time: 68.5888s, Throughput: 72.781610 im/s
Epoch 32, Iter 39, Loss: 10402.6640625, Throughput: 72.072708 im/s
Epoch 32, Train Loss: 10062.1181641, Time: 69.2636s, Throughput: 72.072508 im/s
Epoch 33, Iter 39, Loss: 9348.0927734, Throughput: 71.845979 im/s
Epoch 33, Train Loss: 9980.1611328, Time: 69.4821s, Throughput: 71.845825 im/s
Epoch 34, Iter 39, Loss: 9740.7460938, Throughput: 71.732123 im/s
Epoch 34, Train Loss: 9976.4326172, Time: 69.5924s, Throughput: 71.731937 im/s
Epoch 35, Iter 39, Loss: 9620.7666016, Throughput: 72.315286 im/s
Epoch 35, Train Loss: 9798.8115234, Time: 69.0312s, Throughput: 72.315108 im/s
Epoch 36, Iter 39, Loss: 9560.7773438, Throughput: 71.943058 im/s
Epoch 36, Train Loss: 9873.7714844, Time: 69.3884s, Throughput: 71.942894 im/s
Epoch 37, Iter 39, Loss: 9822.9902344, Throughput: 72.016072 im/s
Epoch 37, Train Loss: 9857.9287109, Time: 69.3180s, Throughput: 72.015885 im/s
Epoch 38, Iter 39, Loss: 9548.0537109, Throughput: 71.546794 im/s
Epoch 38, Train Loss: 9770.3623047, Time: 69.7727s, Throughput: 71.546624 im/s
Epoch 39, Iter 39, Loss: 9796.1806641, Throughput: 71.392936 im/s
Epoch 39, Train Loss: 9713.8828125, Time: 69.9231s, Throughput: 71.392767 im/s
Epoch 40, Iter 39, Loss: 9592.5605469, Throughput: 71.806915 im/s
Epoch 40, Train Loss: 9722.4042969, Time: 69.5199s, Throughput: 71.806730 im/s
Epoch 41, Iter 39, Loss: 9478.2275391, Throughput: 72.201927 im/s
Epoch 41, Train Loss: 9667.8232422, Time: 69.1396s, Throughput: 72.201768 im/s
Epoch 42, Iter 39, Loss: 9511.6093750, Throughput: 71.040978 im/s
Epoch 42, Train Loss: 9615.7431641, Time: 70.2695s, Throughput: 71.040803 im/s
Epoch 43, Iter 39, Loss: 9742.1542969, Throughput: 71.666278 im/s
Epoch 43, Train Loss: 9601.1406250, Time: 69.6563s, Throughput: 71.666124 im/s
Epoch 44, Iter 39, Loss: 9178.7187500, Throughput: 71.784041 im/s
Epoch 44, Train Loss: 9542.4140625, Time: 69.5421s, Throughput: 71.783869 im/s
Epoch 45, Iter 39, Loss: 9261.2695312, Throughput: 71.665436 im/s
Epoch 45, Train Loss: 9483.0966797, Time: 69.6572s, Throughput: 71.665259 im/s
Epoch 46, Iter 39, Loss: 9509.0468750, Throughput: 72.450102 im/s
Epoch 46, Train Loss: 9492.9472656, Time: 68.9028s, Throughput: 72.449932 im/s
Epoch 47, Iter 39, Loss: 9463.7167969, Throughput: 71.688638 im/s
Epoch 47, Train Loss: 9455.3935547, Time: 69.6346s, Throughput: 71.688471 im/s
Epoch 48, Iter 39, Loss: 9721.0019531, Throughput: 71.675526 im/s
Epoch 48, Train Loss: 9439.2080078, Time: 69.6474s, Throughput: 71.675340 im/s
Epoch 49, Iter 39, Loss: 9172.4648438, Throughput: 71.591071 im/s
Epoch 49, Train Loss: 9463.2919922, Time: 69.7295s, Throughput: 71.590903 im/s
Epoch 50, Iter 39, Loss: 9742.7382812, Throughput: 72.821963 im/s
Epoch 50, Train Loss: 9415.4208984, Time: 68.5509s, Throughput: 72.821821 im/s

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
Julia Version 1.12.5
Commit 5fe89b8ddc1 (2026-02-09 16:05 UTC)
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
