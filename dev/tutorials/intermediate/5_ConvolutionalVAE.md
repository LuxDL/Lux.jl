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
Epoch 1, Iter 39, Loss: 24825.8398438, Throughput: 5.226426 im/s
Epoch 1, Train Loss: 39826.4648438, Time: 955.4366s, Throughput: 5.224836 im/s
Epoch 2, Iter 39, Loss: 17413.2265625, Throughput: 72.907964 im/s
Epoch 2, Train Loss: 20704.6425781, Time: 68.4701s, Throughput: 72.907781 im/s
Epoch 3, Iter 39, Loss: 16102.8984375, Throughput: 73.274171 im/s
Epoch 3, Train Loss: 16949.2519531, Time: 68.1279s, Throughput: 73.273941 im/s
Epoch 4, Iter 39, Loss: 14272.3730469, Throughput: 73.198829 im/s
Epoch 4, Train Loss: 15267.3349609, Time: 68.1980s, Throughput: 73.198645 im/s
Epoch 5, Iter 39, Loss: 14923.2207031, Throughput: 73.386914 im/s
Epoch 5, Train Loss: 14214.3769531, Time: 68.0232s, Throughput: 73.386715 im/s
Epoch 6, Iter 39, Loss: 12776.9384766, Throughput: 73.953359 im/s
Epoch 6, Train Loss: 13624.0371094, Time: 67.5022s, Throughput: 73.953178 im/s
Epoch 7, Iter 39, Loss: 12552.5644531, Throughput: 73.409241 im/s
Epoch 7, Train Loss: 12913.1279297, Time: 68.0025s, Throughput: 73.409074 im/s
Epoch 8, Iter 39, Loss: 13908.7392578, Throughput: 73.464363 im/s
Epoch 8, Train Loss: 12675.9599609, Time: 67.9515s, Throughput: 73.464181 im/s
Epoch 9, Iter 39, Loss: 12082.8183594, Throughput: 73.458290 im/s
Epoch 9, Train Loss: 12368.3779297, Time: 67.9571s, Throughput: 73.458114 im/s
Epoch 10, Iter 39, Loss: 11576.1308594, Throughput: 73.142254 im/s
Epoch 10, Train Loss: 12083.7607422, Time: 68.2507s, Throughput: 73.142082 im/s
Epoch 11, Iter 39, Loss: 11254.5546875, Throughput: 72.838464 im/s
Epoch 11, Train Loss: 11853.3857422, Time: 68.5354s, Throughput: 72.838286 im/s
Epoch 12, Iter 39, Loss: 12146.5371094, Throughput: 73.265194 im/s
Epoch 12, Train Loss: 11670.3427734, Time: 68.1362s, Throughput: 73.265030 im/s
Epoch 13, Iter 39, Loss: 11636.7480469, Throughput: 73.509712 im/s
Epoch 13, Train Loss: 11519.4169922, Time: 67.9096s, Throughput: 73.509536 im/s
Epoch 14, Iter 39, Loss: 11618.5214844, Throughput: 73.403236 im/s
Epoch 14, Train Loss: 11267.5019531, Time: 68.0081s, Throughput: 73.403067 im/s
Epoch 15, Iter 39, Loss: 11618.4970703, Throughput: 73.126756 im/s
Epoch 15, Train Loss: 11183.3457031, Time: 68.2652s, Throughput: 73.126581 im/s
Epoch 16, Iter 39, Loss: 11888.4423828, Throughput: 73.626205 im/s
Epoch 16, Train Loss: 11110.8818359, Time: 67.8021s, Throughput: 73.626041 im/s
Epoch 17, Iter 39, Loss: 11280.0195312, Throughput: 73.769646 im/s
Epoch 17, Train Loss: 11051.0830078, Time: 67.6703s, Throughput: 73.769462 im/s
Epoch 18, Iter 39, Loss: 11085.2421875, Throughput: 72.972700 im/s
Epoch 18, Train Loss: 10876.1582031, Time: 68.4093s, Throughput: 72.972542 im/s
Epoch 19, Iter 39, Loss: 11267.8369141, Throughput: 73.546391 im/s
Epoch 19, Train Loss: 10722.9541016, Time: 67.8757s, Throughput: 73.546211 im/s
Epoch 20, Iter 39, Loss: 10426.3613281, Throughput: 73.835734 im/s
Epoch 20, Train Loss: 10629.9121094, Time: 67.6097s, Throughput: 73.835539 im/s
Epoch 21, Iter 39, Loss: 11076.0634766, Throughput: 73.347677 im/s
Epoch 21, Train Loss: 10603.1054688, Time: 68.0596s, Throughput: 73.347489 im/s
Epoch 22, Iter 39, Loss: 10476.7187500, Throughput: 73.654521 im/s
Epoch 22, Train Loss: 10558.6689453, Time: 67.7761s, Throughput: 73.654289 im/s
Epoch 23, Iter 39, Loss: 10082.6259766, Throughput: 73.554927 im/s
Epoch 23, Train Loss: 10434.9833984, Time: 67.8678s, Throughput: 73.554765 im/s
Epoch 24, Iter 39, Loss: 10713.1796875, Throughput: 73.453808 im/s
Epoch 24, Train Loss: 10378.1718750, Time: 67.9613s, Throughput: 73.453618 im/s
Epoch 25, Iter 39, Loss: 10313.5830078, Throughput: 73.314453 im/s
Epoch 25, Train Loss: 10309.0380859, Time: 68.0904s, Throughput: 73.314276 im/s
Epoch 26, Iter 39, Loss: 10406.4648438, Throughput: 73.669984 im/s
Epoch 26, Train Loss: 10331.5351562, Time: 67.7618s, Throughput: 73.669807 im/s
Epoch 27, Iter 39, Loss: 9516.3076172, Throughput: 73.528398 im/s
Epoch 27, Train Loss: 10184.6269531, Time: 67.8923s, Throughput: 73.528240 im/s
Epoch 28, Iter 39, Loss: 10314.6601562, Throughput: 73.216540 im/s
Epoch 28, Train Loss: 10113.0673828, Time: 68.1815s, Throughput: 73.216370 im/s
Epoch 29, Iter 39, Loss: 9716.3564453, Throughput: 73.427393 im/s
Epoch 29, Train Loss: 10071.6230469, Time: 67.9857s, Throughput: 73.427218 im/s
Epoch 30, Iter 39, Loss: 9657.2314453, Throughput: 73.264003 im/s
Epoch 30, Train Loss: 9969.6455078, Time: 68.1373s, Throughput: 73.263816 im/s
Epoch 31, Iter 39, Loss: 9788.8652344, Throughput: 73.954448 im/s
Epoch 31, Train Loss: 9930.8867188, Time: 67.5012s, Throughput: 73.954286 im/s
Epoch 32, Iter 39, Loss: 9900.6474609, Throughput: 72.932421 im/s
Epoch 32, Train Loss: 9992.7851562, Time: 68.4471s, Throughput: 72.932260 im/s
Epoch 33, Iter 39, Loss: 9443.6064453, Throughput: 73.337531 im/s
Epoch 33, Train Loss: 9897.4394531, Time: 68.0690s, Throughput: 73.337366 im/s
Epoch 34, Iter 39, Loss: 9646.4335938, Throughput: 73.626090 im/s
Epoch 34, Train Loss: 9867.7832031, Time: 67.8022s, Throughput: 73.625911 im/s
Epoch 35, Iter 39, Loss: 10180.8251953, Throughput: 73.311781 im/s
Epoch 35, Train Loss: 9764.8056641, Time: 68.0929s, Throughput: 73.311614 im/s
Epoch 36, Iter 39, Loss: 10458.6816406, Throughput: 73.325736 im/s
Epoch 36, Train Loss: 9762.3554688, Time: 68.0799s, Throughput: 73.325570 im/s
Epoch 37, Iter 39, Loss: 8936.0605469, Throughput: 72.603283 im/s
Epoch 37, Train Loss: 9742.5322266, Time: 68.7574s, Throughput: 72.603106 im/s
Epoch 38, Iter 39, Loss: 9448.7929688, Throughput: 71.895324 im/s
Epoch 38, Train Loss: 9697.6523438, Time: 69.4345s, Throughput: 71.895136 im/s
Epoch 39, Iter 39, Loss: 9696.8447266, Throughput: 71.827644 im/s
Epoch 39, Train Loss: 9787.6337891, Time: 69.4999s, Throughput: 71.827492 im/s
Epoch 40, Iter 39, Loss: 9805.3994141, Throughput: 72.537316 im/s
Epoch 40, Train Loss: 9684.3125000, Time: 68.8199s, Throughput: 72.537137 im/s
Epoch 41, Iter 39, Loss: 9626.4111328, Throughput: 73.066183 im/s
Epoch 41, Train Loss: 9678.0869141, Time: 68.3218s, Throughput: 73.065997 im/s
Epoch 42, Iter 39, Loss: 9675.1992188, Throughput: 73.225399 im/s
Epoch 42, Train Loss: 9577.1962891, Time: 68.1732s, Throughput: 73.225204 im/s
Epoch 43, Iter 39, Loss: 9981.0722656, Throughput: 73.241902 im/s
Epoch 43, Train Loss: 9517.4267578, Time: 68.1579s, Throughput: 73.241715 im/s
Epoch 44, Iter 39, Loss: 10130.3701172, Throughput: 72.702277 im/s
Epoch 44, Train Loss: 9523.8730469, Time: 68.6638s, Throughput: 72.702109 im/s
Epoch 45, Iter 39, Loss: 10209.7812500, Throughput: 73.324221 im/s
Epoch 45, Train Loss: 9499.2167969, Time: 68.0813s, Throughput: 73.324070 im/s
Epoch 46, Iter 39, Loss: 9482.0976562, Throughput: 73.450912 im/s
Epoch 46, Train Loss: 9388.9384766, Time: 67.9639s, Throughput: 73.450720 im/s
Epoch 47, Iter 39, Loss: 9392.2480469, Throughput: 73.202440 im/s
Epoch 47, Train Loss: 9366.5029297, Time: 68.1946s, Throughput: 73.202282 im/s
Epoch 48, Iter 39, Loss: 9588.3505859, Throughput: 73.168340 im/s
Epoch 48, Train Loss: 9371.4033203, Time: 68.2264s, Throughput: 73.168179 im/s
Epoch 49, Iter 39, Loss: 9415.5195312, Throughput: 73.241920 im/s
Epoch 49, Train Loss: 9420.9863281, Time: 68.1579s, Throughput: 73.241738 im/s
Epoch 50, Iter 39, Loss: 8805.2597656, Throughput: 72.559536 im/s
Epoch 50, Train Loss: 9393.9306641, Time: 68.7988s, Throughput: 72.559388 im/s

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
