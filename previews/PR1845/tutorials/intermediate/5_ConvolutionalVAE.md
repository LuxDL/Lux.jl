---
url: /previews/PR1845/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23788.3183594, Throughput: 24.713052 im/s
Epoch 1, Train Loss: 39688.6367188, Time: 202.1558s, Throughput: 24.693825 im/s
Epoch 2, Iter 39, Loss: 17072.3867188, Throughput: 130.320235 im/s
Epoch 2, Train Loss: 20020.5488281, Time: 38.3058s, Throughput: 130.319694 im/s
Epoch 3, Iter 39, Loss: 15036.5253906, Throughput: 127.520083 im/s
Epoch 3, Train Loss: 16501.5976562, Time: 39.1469s, Throughput: 127.519516 im/s
Epoch 4, Iter 39, Loss: 14192.1093750, Throughput: 129.536422 im/s
Epoch 4, Train Loss: 14939.2304688, Time: 38.5376s, Throughput: 129.535918 im/s
Epoch 5, Iter 39, Loss: 13568.1796875, Throughput: 129.453588 im/s
Epoch 5, Train Loss: 14062.5048828, Time: 38.5622s, Throughput: 129.453051 im/s
Epoch 6, Iter 39, Loss: 13290.4384766, Throughput: 127.871598 im/s
Epoch 6, Train Loss: 13351.6923828, Time: 39.0393s, Throughput: 127.871047 im/s
Epoch 7, Iter 39, Loss: 12073.6806641, Throughput: 123.888379 im/s
Epoch 7, Train Loss: 12911.1484375, Time: 40.2945s, Throughput: 123.887885 im/s
Epoch 8, Iter 39, Loss: 12104.6484375, Throughput: 123.144306 im/s
Epoch 8, Train Loss: 12544.5888672, Time: 40.5380s, Throughput: 123.143854 im/s
Epoch 9, Iter 39, Loss: 11805.6914062, Throughput: 122.944636 im/s
Epoch 9, Train Loss: 12183.7333984, Time: 40.6038s, Throughput: 122.944125 im/s
Epoch 10, Iter 39, Loss: 11709.5263672, Throughput: 121.586721 im/s
Epoch 10, Train Loss: 12008.7500000, Time: 41.0573s, Throughput: 121.586236 im/s
Epoch 11, Iter 39, Loss: 11931.4238281, Throughput: 121.542394 im/s
Epoch 11, Train Loss: 11808.2353516, Time: 41.0722s, Throughput: 121.541953 im/s
Epoch 12, Iter 39, Loss: 11216.9248047, Throughput: 121.109397 im/s
Epoch 12, Train Loss: 11614.6523438, Time: 41.2191s, Throughput: 121.108898 im/s
Epoch 13, Iter 39, Loss: 10403.5498047, Throughput: 120.712759 im/s
Epoch 13, Train Loss: 11372.5146484, Time: 41.3545s, Throughput: 120.712342 im/s
Epoch 14, Iter 39, Loss: 10658.3066406, Throughput: 121.262627 im/s
Epoch 14, Train Loss: 11227.2187500, Time: 41.1670s, Throughput: 121.262137 im/s
Epoch 15, Iter 39, Loss: 10925.7910156, Throughput: 120.946769 im/s
Epoch 15, Train Loss: 11107.2558594, Time: 41.2745s, Throughput: 120.946277 im/s
Epoch 16, Iter 39, Loss: 10822.7177734, Throughput: 121.960441 im/s
Epoch 16, Train Loss: 11017.8505859, Time: 40.9315s, Throughput: 121.959955 im/s
Epoch 17, Iter 39, Loss: 10375.8476562, Throughput: 124.889165 im/s
Epoch 17, Train Loss: 10904.8544922, Time: 39.9716s, Throughput: 124.888699 im/s
Epoch 18, Iter 39, Loss: 10150.3310547, Throughput: 124.058404 im/s
Epoch 18, Train Loss: 10691.8486328, Time: 40.2393s, Throughput: 124.057870 im/s
Epoch 19, Iter 39, Loss: 9903.1484375, Throughput: 121.825473 im/s
Epoch 19, Train Loss: 10697.0146484, Time: 40.9768s, Throughput: 121.825030 im/s
Epoch 20, Iter 39, Loss: 10682.7421875, Throughput: 122.961126 im/s
Epoch 20, Train Loss: 10527.5429688, Time: 40.5983s, Throughput: 122.960669 im/s
Epoch 21, Iter 39, Loss: 10731.7949219, Throughput: 122.430763 im/s
Epoch 21, Train Loss: 10525.3359375, Time: 40.7742s, Throughput: 122.430244 im/s
Epoch 22, Iter 39, Loss: 10355.2412109, Throughput: 121.847329 im/s
Epoch 22, Train Loss: 10496.2519531, Time: 40.9695s, Throughput: 121.846856 im/s
Epoch 23, Iter 39, Loss: 10196.7470703, Throughput: 121.772692 im/s
Epoch 23, Train Loss: 10474.5742188, Time: 40.9946s, Throughput: 121.772136 im/s
Epoch 24, Iter 39, Loss: 10348.8759766, Throughput: 121.865390 im/s
Epoch 24, Train Loss: 10340.8857422, Time: 40.9634s, Throughput: 121.864914 im/s
Epoch 25, Iter 39, Loss: 11588.8056641, Throughput: 121.715065 im/s
Epoch 25, Train Loss: 10234.2568359, Time: 41.0140s, Throughput: 121.714599 im/s
Epoch 26, Iter 39, Loss: 10232.8535156, Throughput: 121.560039 im/s
Epoch 26, Train Loss: 10194.5605469, Time: 41.0663s, Throughput: 121.559572 im/s
Epoch 27, Iter 39, Loss: 9870.5488281, Throughput: 121.832912 im/s
Epoch 27, Train Loss: 10183.8066406, Time: 40.9743s, Throughput: 121.832401 im/s
Epoch 28, Iter 39, Loss: 9950.3935547, Throughput: 121.656633 im/s
Epoch 28, Train Loss: 10067.0371094, Time: 41.0337s, Throughput: 121.656185 im/s
Epoch 29, Iter 39, Loss: 10060.6699219, Throughput: 121.357214 im/s
Epoch 29, Train Loss: 9961.9443359, Time: 41.1349s, Throughput: 121.356704 im/s
Epoch 30, Iter 39, Loss: 9579.8652344, Throughput: 121.590462 im/s
Epoch 30, Train Loss: 9945.0449219, Time: 41.0560s, Throughput: 121.590018 im/s
Epoch 31, Iter 39, Loss: 9715.5839844, Throughput: 121.596960 im/s
Epoch 31, Train Loss: 9951.3671875, Time: 41.0538s, Throughput: 121.596450 im/s
Epoch 32, Iter 39, Loss: 10326.5859375, Throughput: 121.343553 im/s
Epoch 32, Train Loss: 9858.3906250, Time: 41.1396s, Throughput: 121.343082 im/s
Epoch 33, Iter 39, Loss: 9844.0302734, Throughput: 121.918552 im/s
Epoch 33, Train Loss: 9800.8730469, Time: 40.9455s, Throughput: 121.918067 im/s
Epoch 34, Iter 39, Loss: 10885.5224609, Throughput: 122.225569 im/s
Epoch 34, Train Loss: 9805.2226562, Time: 40.8427s, Throughput: 122.225117 im/s
Epoch 35, Iter 39, Loss: 9663.7011719, Throughput: 121.790529 im/s
Epoch 35, Train Loss: 9799.0390625, Time: 40.9886s, Throughput: 121.789988 im/s
Epoch 36, Iter 39, Loss: 9966.4687500, Throughput: 121.826243 im/s
Epoch 36, Train Loss: 9787.1210938, Time: 40.9766s, Throughput: 121.825749 im/s
Epoch 37, Iter 39, Loss: 9658.3505859, Throughput: 121.031952 im/s
Epoch 37, Train Loss: 9817.6679688, Time: 41.2455s, Throughput: 121.031438 im/s
Epoch 38, Iter 39, Loss: 9190.0439453, Throughput: 122.380266 im/s
Epoch 38, Train Loss: 9658.0605469, Time: 40.7911s, Throughput: 122.379765 im/s
Epoch 39, Iter 39, Loss: 9250.8623047, Throughput: 122.225031 im/s
Epoch 39, Train Loss: 9560.5322266, Time: 40.8429s, Throughput: 122.224567 im/s
Epoch 40, Iter 39, Loss: 9255.2832031, Throughput: 122.133317 im/s
Epoch 40, Train Loss: 9559.1894531, Time: 40.8735s, Throughput: 122.132839 im/s
Epoch 41, Iter 39, Loss: 9460.3964844, Throughput: 122.703465 im/s
Epoch 41, Train Loss: 9565.3144531, Time: 40.6836s, Throughput: 122.702988 im/s
Epoch 42, Iter 39, Loss: 9370.1269531, Throughput: 123.630658 im/s
Epoch 42, Train Loss: 9556.2109375, Time: 40.3785s, Throughput: 123.630149 im/s
Epoch 43, Iter 39, Loss: 9772.5078125, Throughput: 123.554597 im/s
Epoch 43, Train Loss: 9506.8007812, Time: 40.4033s, Throughput: 123.554142 im/s
Epoch 44, Iter 39, Loss: 9718.4267578, Throughput: 128.326826 im/s
Epoch 44, Train Loss: 9437.6005859, Time: 38.9008s, Throughput: 128.326337 im/s
Epoch 45, Iter 39, Loss: 9651.4189453, Throughput: 129.131650 im/s
Epoch 45, Train Loss: 9452.8769531, Time: 38.6584s, Throughput: 129.131149 im/s
Epoch 46, Iter 39, Loss: 9000.6523438, Throughput: 130.703881 im/s
Epoch 46, Train Loss: 9453.2734375, Time: 38.1934s, Throughput: 130.703238 im/s
Epoch 47, Iter 39, Loss: 9536.1757812, Throughput: 130.656371 im/s
Epoch 47, Train Loss: 9364.1982422, Time: 38.2072s, Throughput: 130.655862 im/s
Epoch 48, Iter 39, Loss: 9892.9355469, Throughput: 129.659282 im/s
Epoch 48, Train Loss: 9338.2207031, Time: 38.5011s, Throughput: 129.658565 im/s
Epoch 49, Iter 39, Loss: 9363.6425781, Throughput: 130.475802 im/s
Epoch 49, Train Loss: 9362.6132812, Time: 38.2601s, Throughput: 130.475293 im/s
Epoch 50, Iter 39, Loss: 9130.2851562, Throughput: 130.401814 im/s
Epoch 50, Train Loss: 9268.4882812, Time: 38.2818s, Throughput: 130.401255 im/s

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
  CPU: 4 × AMD EPYC 9V45 96-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver5)
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
