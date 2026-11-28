---
url: /v1/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24246.8339844, Throughput: 23.000274 im/s
Epoch 1, Train Loss: 40259.4218750, Time: 217.2336s, Throughput: 22.979867 im/s
Epoch 2, Iter 39, Loss: 18688.1035156, Throughput: 99.713324 im/s
Epoch 2, Train Loss: 20524.4589844, Time: 50.0637s, Throughput: 99.712940 im/s
Epoch 3, Iter 39, Loss: 15942.9746094, Throughput: 99.587776 im/s
Epoch 3, Train Loss: 16997.4824219, Time: 50.1268s, Throughput: 99.587369 im/s
Epoch 4, Iter 39, Loss: 14515.8652344, Throughput: 99.512018 im/s
Epoch 4, Train Loss: 15407.2705078, Time: 50.1650s, Throughput: 99.511544 im/s
Epoch 5, Iter 39, Loss: 14496.1435547, Throughput: 98.950324 im/s
Epoch 5, Train Loss: 14513.0292969, Time: 50.4498s, Throughput: 98.949942 im/s
Epoch 6, Iter 39, Loss: 13413.6220703, Throughput: 99.136364 im/s
Epoch 6, Train Loss: 13857.9326172, Time: 50.3551s, Throughput: 99.135926 im/s
Epoch 7, Iter 39, Loss: 13612.9736328, Throughput: 99.283944 im/s
Epoch 7, Train Loss: 13324.7841797, Time: 50.2802s, Throughput: 99.283528 im/s
Epoch 8, Iter 39, Loss: 13049.5781250, Throughput: 99.179191 im/s
Epoch 8, Train Loss: 12976.7705078, Time: 50.3333s, Throughput: 99.178840 im/s
Epoch 9, Iter 39, Loss: 12661.1064453, Throughput: 99.447383 im/s
Epoch 9, Train Loss: 12693.6660156, Time: 50.1976s, Throughput: 99.446995 im/s
Epoch 10, Iter 39, Loss: 13109.4248047, Throughput: 99.560907 im/s
Epoch 10, Train Loss: 12422.4501953, Time: 50.1403s, Throughput: 99.560538 im/s
Epoch 11, Iter 39, Loss: 11320.7851562, Throughput: 99.545379 im/s
Epoch 11, Train Loss: 12244.5351562, Time: 50.1482s, Throughput: 99.545024 im/s
Epoch 12, Iter 39, Loss: 11917.0292969, Throughput: 99.386108 im/s
Epoch 12, Train Loss: 11949.1367188, Time: 50.2286s, Throughput: 99.385641 im/s
Epoch 13, Iter 39, Loss: 12062.7851562, Throughput: 99.377240 im/s
Epoch 13, Train Loss: 11906.8623047, Time: 50.2331s, Throughput: 99.376781 im/s
Epoch 14, Iter 39, Loss: 12219.7421875, Throughput: 99.465122 im/s
Epoch 14, Train Loss: 11804.7119141, Time: 50.1887s, Throughput: 99.464690 im/s
Epoch 15, Iter 39, Loss: 12210.2988281, Throughput: 99.500399 im/s
Epoch 15, Train Loss: 11569.2119141, Time: 50.1708s, Throughput: 99.500010 im/s
Epoch 16, Iter 39, Loss: 11594.8134766, Throughput: 99.972668 im/s
Epoch 16, Train Loss: 11416.6806641, Time: 49.9338s, Throughput: 99.972311 im/s
Epoch 17, Iter 39, Loss: 11980.2919922, Throughput: 99.974846 im/s
Epoch 17, Train Loss: 11295.6025391, Time: 49.9327s, Throughput: 99.974496 im/s
Epoch 18, Iter 39, Loss: 11418.1943359, Throughput: 100.158389 im/s
Epoch 18, Train Loss: 11217.8271484, Time: 49.8413s, Throughput: 100.157988 im/s
Epoch 19, Iter 39, Loss: 11019.4658203, Throughput: 100.118608 im/s
Epoch 19, Train Loss: 11189.8867188, Time: 49.8610s, Throughput: 100.118239 im/s
Epoch 20, Iter 39, Loss: 11123.6894531, Throughput: 100.200640 im/s
Epoch 20, Train Loss: 11021.7460938, Time: 49.8202s, Throughput: 100.200244 im/s
Epoch 21, Iter 39, Loss: 11615.0429688, Throughput: 99.772073 im/s
Epoch 21, Train Loss: 10950.2031250, Time: 50.0342s, Throughput: 99.771660 im/s
Epoch 22, Iter 39, Loss: 10255.5126953, Throughput: 99.466688 im/s
Epoch 22, Train Loss: 10973.0664062, Time: 50.1879s, Throughput: 99.466283 im/s
Epoch 23, Iter 39, Loss: 10704.8955078, Throughput: 99.372729 im/s
Epoch 23, Train Loss: 10747.0517578, Time: 50.2353s, Throughput: 99.372314 im/s
Epoch 24, Iter 39, Loss: 11674.4873047, Throughput: 99.668574 im/s
Epoch 24, Train Loss: 10764.5205078, Time: 50.0862s, Throughput: 99.668192 im/s
Epoch 25, Iter 39, Loss: 11133.3642578, Throughput: 100.418660 im/s
Epoch 25, Train Loss: 10661.8662109, Time: 49.7121s, Throughput: 100.418286 im/s
Epoch 26, Iter 39, Loss: 11310.8310547, Throughput: 100.674255 im/s
Epoch 26, Train Loss: 10546.8886719, Time: 49.5858s, Throughput: 100.673884 im/s
Epoch 27, Iter 39, Loss: 11033.6259766, Throughput: 100.829539 im/s
Epoch 27, Train Loss: 10528.7900391, Time: 49.5095s, Throughput: 100.829134 im/s
Epoch 28, Iter 39, Loss: 11316.9472656, Throughput: 100.804469 im/s
Epoch 28, Train Loss: 10488.1035156, Time: 49.5218s, Throughput: 100.804052 im/s
Epoch 29, Iter 39, Loss: 11042.3750000, Throughput: 100.764341 im/s
Epoch 29, Train Loss: 10441.3183594, Time: 49.5415s, Throughput: 100.763941 im/s
Epoch 30, Iter 39, Loss: 10316.5009766, Throughput: 100.694609 im/s
Epoch 30, Train Loss: 10367.0527344, Time: 49.5758s, Throughput: 100.694203 im/s
Epoch 31, Iter 39, Loss: 9924.6074219, Throughput: 100.917955 im/s
Epoch 31, Train Loss: 10353.7480469, Time: 49.4662s, Throughput: 100.917488 im/s
Epoch 32, Iter 39, Loss: 10412.8535156, Throughput: 100.966336 im/s
Epoch 32, Train Loss: 10338.6181641, Time: 49.4424s, Throughput: 100.965954 im/s
Epoch 33, Iter 39, Loss: 10364.6064453, Throughput: 100.804567 im/s
Epoch 33, Train Loss: 10192.5332031, Time: 49.5218s, Throughput: 100.804184 im/s
Epoch 34, Iter 39, Loss: 10056.6113281, Throughput: 100.886402 im/s
Epoch 34, Train Loss: 10238.1396484, Time: 49.4816s, Throughput: 100.886002 im/s
Epoch 35, Iter 39, Loss: 11007.7626953, Throughput: 101.111571 im/s
Epoch 35, Train Loss: 10183.7050781, Time: 49.3714s, Throughput: 101.111221 im/s
Epoch 36, Iter 39, Loss: 9956.6347656, Throughput: 100.800567 im/s
Epoch 36, Train Loss: 10079.3730469, Time: 49.5237s, Throughput: 100.800178 im/s
Epoch 37, Iter 39, Loss: 10438.8183594, Throughput: 101.197108 im/s
Epoch 37, Train Loss: 10111.3027344, Time: 49.3297s, Throughput: 101.196716 im/s
Epoch 38, Iter 39, Loss: 10047.5976562, Throughput: 100.905458 im/s
Epoch 38, Train Loss: 10075.9267578, Time: 49.4722s, Throughput: 100.905077 im/s
Epoch 39, Iter 39, Loss: 10611.9931641, Throughput: 100.858208 im/s
Epoch 39, Train Loss: 9994.9130859, Time: 49.4954s, Throughput: 100.857825 im/s
Epoch 40, Iter 39, Loss: 10099.2128906, Throughput: 100.926865 im/s
Epoch 40, Train Loss: 9998.3652344, Time: 49.4617s, Throughput: 100.926514 im/s
Epoch 41, Iter 39, Loss: 10387.9589844, Throughput: 100.903282 im/s
Epoch 41, Train Loss: 9941.6533203, Time: 49.4733s, Throughput: 100.902932 im/s
Epoch 42, Iter 39, Loss: 10032.5292969, Throughput: 101.023325 im/s
Epoch 42, Train Loss: 9832.7812500, Time: 49.4145s, Throughput: 101.022967 im/s
Epoch 43, Iter 39, Loss: 9809.0068359, Throughput: 100.741491 im/s
Epoch 43, Train Loss: 9773.6894531, Time: 49.5528s, Throughput: 100.741101 im/s
Epoch 44, Iter 39, Loss: 9717.1230469, Throughput: 100.660642 im/s
Epoch 44, Train Loss: 9834.3886719, Time: 49.5926s, Throughput: 100.660252 im/s
Epoch 45, Iter 39, Loss: 10257.8105469, Throughput: 100.554729 im/s
Epoch 45, Train Loss: 9838.8466797, Time: 49.6448s, Throughput: 100.554352 im/s
Epoch 46, Iter 39, Loss: 9678.6289062, Throughput: 100.483246 im/s
Epoch 46, Train Loss: 9827.0888672, Time: 49.6801s, Throughput: 100.482823 im/s
Epoch 47, Iter 39, Loss: 10466.1572266, Throughput: 100.198657 im/s
Epoch 47, Train Loss: 9696.6621094, Time: 49.8212s, Throughput: 100.198279 im/s
Epoch 48, Iter 39, Loss: 9369.2773438, Throughput: 100.358770 im/s
Epoch 48, Train Loss: 9672.5292969, Time: 49.7418s, Throughput: 100.358343 im/s
Epoch 49, Iter 39, Loss: 9417.0742188, Throughput: 99.592240 im/s
Epoch 49, Train Loss: 9675.2646484, Time: 50.1246s, Throughput: 99.591839 im/s
Epoch 50, Iter 39, Loss: 10186.4111328, Throughput: 100.002007 im/s
Epoch 50, Train Loss: 9640.7167969, Time: 49.9192s, Throughput: 100.001609 im/s

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
Julia Version 1.13.4
Commit c1ac32ab733 (2026-11-13 17:30 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × Intel(R) Xeon(R) 6975P-C
  WORD_SIZE: 64
  LLVM: libLLVM-20.1.8 (ORCJIT, graniterapids)
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
