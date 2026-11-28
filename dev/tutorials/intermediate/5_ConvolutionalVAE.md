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
Epoch 1, Iter 39, Loss: 24910.3359375, Throughput: 19.210261 im/s
Epoch 1, Train Loss: 40456.6523438, Time: 260.1212s, Throughput: 19.191052 im/s
Epoch 2, Iter 39, Loss: 19196.7089844, Throughput: 148.139215 im/s
Epoch 2, Train Loss: 20794.9140625, Time: 33.6982s, Throughput: 148.138332 im/s
Epoch 3, Iter 39, Loss: 17263.7382812, Throughput: 148.863044 im/s
Epoch 3, Train Loss: 17111.4082031, Time: 33.5344s, Throughput: 148.862170 im/s
Epoch 4, Iter 39, Loss: 15036.9589844, Throughput: 148.849090 im/s
Epoch 4, Train Loss: 15495.0107422, Time: 33.5375s, Throughput: 148.848362 im/s
Epoch 5, Iter 39, Loss: 13910.8476562, Throughput: 149.157498 im/s
Epoch 5, Train Loss: 14561.1845703, Time: 33.4681s, Throughput: 149.156806 im/s
Epoch 6, Iter 39, Loss: 13800.8027344, Throughput: 148.853963 im/s
Epoch 6, Train Loss: 13846.7050781, Time: 33.5364s, Throughput: 148.853257 im/s
Epoch 7, Iter 39, Loss: 12889.6835938, Throughput: 148.436607 im/s
Epoch 7, Train Loss: 13360.3369141, Time: 33.6307s, Throughput: 148.435870 im/s
Epoch 8, Iter 39, Loss: 12567.4833984, Throughput: 148.437649 im/s
Epoch 8, Train Loss: 13026.4394531, Time: 33.6304s, Throughput: 148.436930 im/s
Epoch 9, Iter 39, Loss: 12350.4306641, Throughput: 148.933496 im/s
Epoch 9, Train Loss: 12661.1523438, Time: 33.5185s, Throughput: 148.932755 im/s
Epoch 10, Iter 39, Loss: 12180.0507812, Throughput: 148.230833 im/s
Epoch 10, Train Loss: 12348.0468750, Time: 33.6774s, Throughput: 148.230102 im/s
Epoch 11, Iter 39, Loss: 12417.6044922, Throughput: 148.192562 im/s
Epoch 11, Train Loss: 12220.6455078, Time: 33.6860s, Throughput: 148.191946 im/s
Epoch 12, Iter 39, Loss: 11427.8271484, Throughput: 148.536580 im/s
Epoch 12, Train Loss: 12010.1611328, Time: 33.6080s, Throughput: 148.535864 im/s
Epoch 13, Iter 39, Loss: 12179.2236328, Throughput: 148.689190 im/s
Epoch 13, Train Loss: 11886.3291016, Time: 33.5736s, Throughput: 148.688460 im/s
Epoch 14, Iter 39, Loss: 11062.2656250, Throughput: 149.247738 im/s
Epoch 14, Train Loss: 11685.5546875, Time: 33.4479s, Throughput: 149.246957 im/s
Epoch 15, Iter 39, Loss: 11326.1767578, Throughput: 148.960498 im/s
Epoch 15, Train Loss: 11509.7080078, Time: 33.5124s, Throughput: 148.959721 im/s
Epoch 16, Iter 39, Loss: 11155.5341797, Throughput: 149.270570 im/s
Epoch 16, Train Loss: 11402.4892578, Time: 33.4428s, Throughput: 149.269825 im/s
Epoch 17, Iter 39, Loss: 11170.9355469, Throughput: 148.621766 im/s
Epoch 17, Train Loss: 11314.6777344, Time: 33.5888s, Throughput: 148.621075 im/s
Epoch 18, Iter 39, Loss: 11117.1142578, Throughput: 149.017421 im/s
Epoch 18, Train Loss: 11231.3593750, Time: 33.4996s, Throughput: 149.016766 im/s
Epoch 19, Iter 39, Loss: 10595.1289062, Throughput: 148.573016 im/s
Epoch 19, Train Loss: 11093.0693359, Time: 33.5998s, Throughput: 148.572278 im/s
Epoch 20, Iter 39, Loss: 11095.4296875, Throughput: 148.408342 im/s
Epoch 20, Train Loss: 11074.6552734, Time: 33.6371s, Throughput: 148.407558 im/s
Epoch 21, Iter 39, Loss: 10830.2363281, Throughput: 148.360805 im/s
Epoch 21, Train Loss: 10938.7656250, Time: 33.6479s, Throughput: 148.359866 im/s
Epoch 22, Iter 39, Loss: 10336.9765625, Throughput: 149.116120 im/s
Epoch 22, Train Loss: 10888.1591797, Time: 33.4774s, Throughput: 149.115372 im/s
Epoch 23, Iter 39, Loss: 9302.5087891, Throughput: 148.632553 im/s
Epoch 23, Train Loss: 10756.5166016, Time: 33.5864s, Throughput: 148.631797 im/s
Epoch 24, Iter 39, Loss: 11266.1542969, Throughput: 148.752300 im/s
Epoch 24, Train Loss: 10624.2685547, Time: 33.5593s, Throughput: 148.751626 im/s
Epoch 25, Iter 39, Loss: 10397.2080078, Throughput: 148.263314 im/s
Epoch 25, Train Loss: 10682.4042969, Time: 33.6700s, Throughput: 148.262663 im/s
Epoch 26, Iter 39, Loss: 10197.4296875, Throughput: 148.621699 im/s
Epoch 26, Train Loss: 10600.5566406, Time: 33.5888s, Throughput: 148.621022 im/s
Epoch 27, Iter 39, Loss: 10344.3769531, Throughput: 149.100799 im/s
Epoch 27, Train Loss: 10525.1748047, Time: 33.4809s, Throughput: 149.100072 im/s
Epoch 28, Iter 39, Loss: 10354.2373047, Throughput: 148.722218 im/s
Epoch 28, Train Loss: 10424.9472656, Time: 33.5661s, Throughput: 148.721455 im/s
Epoch 29, Iter 39, Loss: 10058.0957031, Throughput: 148.661891 im/s
Epoch 29, Train Loss: 10342.2939453, Time: 33.5797s, Throughput: 148.661067 im/s
Epoch 30, Iter 39, Loss: 9902.7734375, Throughput: 149.202252 im/s
Epoch 30, Train Loss: 10386.9931641, Time: 33.4581s, Throughput: 149.201459 im/s
Epoch 31, Iter 39, Loss: 10805.1835938, Throughput: 148.573511 im/s
Epoch 31, Train Loss: 10341.9218750, Time: 33.5997s, Throughput: 148.572751 im/s
Epoch 32, Iter 39, Loss: 10416.6464844, Throughput: 149.114726 im/s
Epoch 32, Train Loss: 10373.3486328, Time: 33.4777s, Throughput: 149.114004 im/s
Epoch 33, Iter 39, Loss: 10075.8808594, Throughput: 148.753542 im/s
Epoch 33, Train Loss: 10182.7128906, Time: 33.5590s, Throughput: 148.752890 im/s
Epoch 34, Iter 39, Loss: 10232.0000000, Throughput: 149.103934 im/s
Epoch 34, Train Loss: 10181.6445312, Time: 33.4801s, Throughput: 149.103297 im/s
Epoch 35, Iter 39, Loss: 10256.7968750, Throughput: 149.000127 im/s
Epoch 35, Train Loss: 10126.2744141, Time: 33.5035s, Throughput: 148.999345 im/s
Epoch 36, Iter 39, Loss: 9967.9833984, Throughput: 148.922287 im/s
Epoch 36, Train Loss: 10170.8691406, Time: 33.5210s, Throughput: 148.921554 im/s
Epoch 37, Iter 39, Loss: 10389.6464844, Throughput: 148.868382 im/s
Epoch 37, Train Loss: 10033.7646484, Time: 33.5332s, Throughput: 148.867583 im/s
Epoch 38, Iter 39, Loss: 9831.4501953, Throughput: 148.742287 im/s
Epoch 38, Train Loss: 9963.8994141, Time: 33.5616s, Throughput: 148.741490 im/s
Epoch 39, Iter 39, Loss: 9822.0039062, Throughput: 148.728896 im/s
Epoch 39, Train Loss: 9975.6767578, Time: 33.5646s, Throughput: 148.728178 im/s
Epoch 40, Iter 39, Loss: 10295.3769531, Throughput: 148.332904 im/s
Epoch 40, Train Loss: 9975.7080078, Time: 33.6542s, Throughput: 148.332102 im/s
Epoch 41, Iter 39, Loss: 10163.5625000, Throughput: 149.072939 im/s
Epoch 41, Train Loss: 9900.7431641, Time: 33.4871s, Throughput: 149.072227 im/s
Epoch 42, Iter 39, Loss: 9620.4453125, Throughput: 149.028075 im/s
Epoch 42, Train Loss: 9816.9794922, Time: 33.4972s, Throughput: 149.027408 im/s
Epoch 43, Iter 39, Loss: 9184.0019531, Throughput: 148.612275 im/s
Epoch 43, Train Loss: 9883.7294922, Time: 33.5909s, Throughput: 148.611545 im/s
Epoch 44, Iter 39, Loss: 10168.5361328, Throughput: 148.374117 im/s
Epoch 44, Train Loss: 9815.0107422, Time: 33.6448s, Throughput: 148.373451 im/s
Epoch 45, Iter 39, Loss: 9611.9091797, Throughput: 148.931844 im/s
Epoch 45, Train Loss: 9845.9414062, Time: 33.5189s, Throughput: 148.931016 im/s
Epoch 46, Iter 39, Loss: 9991.4199219, Throughput: 149.251733 im/s
Epoch 46, Train Loss: 9713.0234375, Time: 33.4470s, Throughput: 149.250872 im/s
Epoch 47, Iter 39, Loss: 10010.2275391, Throughput: 148.444001 im/s
Epoch 47, Train Loss: 9753.4248047, Time: 33.6290s, Throughput: 148.443303 im/s
Epoch 48, Iter 39, Loss: 9486.3818359, Throughput: 149.007034 im/s
Epoch 48, Train Loss: 9718.4560547, Time: 33.5020s, Throughput: 149.006153 im/s
Epoch 49, Iter 39, Loss: 9279.2773438, Throughput: 148.996637 im/s
Epoch 49, Train Loss: 9649.6630859, Time: 33.5043s, Throughput: 148.995992 im/s
Epoch 50, Iter 39, Loss: 10167.5390625, Throughput: 149.185629 im/s
Epoch 50, Train Loss: 9708.8925781, Time: 33.4618s, Throughput: 149.184956 im/s

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
  CPU: 4 × AMD EPYC 7763 64-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-20.1.8 (ORCJIT, znver3)
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
