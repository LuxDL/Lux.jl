---
url: /previews/PR1832/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23349.8808594, Throughput: 16.495822 im/s
Epoch 1, Train Loss: 39684.9414062, Time: 302.8815s, Throughput: 16.481691 im/s
Epoch 2, Iter 39, Loss: 17581.2421875, Throughput: 173.181120 im/s
Epoch 2, Train Loss: 20059.4355469, Time: 28.8255s, Throughput: 173.180164 im/s
Epoch 3, Iter 39, Loss: 16183.6933594, Throughput: 173.193455 im/s
Epoch 3, Train Loss: 16530.7519531, Time: 28.8234s, Throughput: 173.192543 im/s
Epoch 4, Iter 39, Loss: 14228.1367188, Throughput: 173.511991 im/s
Epoch 4, Train Loss: 15052.0449219, Time: 28.7705s, Throughput: 173.510996 im/s
Epoch 5, Iter 39, Loss: 13949.9169922, Throughput: 173.060606 im/s
Epoch 5, Train Loss: 14092.6250000, Time: 28.8455s, Throughput: 173.059634 im/s
Epoch 6, Iter 39, Loss: 13411.4218750, Throughput: 173.371409 im/s
Epoch 6, Train Loss: 13617.7021484, Time: 28.7938s, Throughput: 173.370439 im/s
Epoch 7, Iter 39, Loss: 12975.6289062, Throughput: 173.001956 im/s
Epoch 7, Train Loss: 12987.3906250, Time: 28.8553s, Throughput: 173.001082 im/s
Epoch 8, Iter 39, Loss: 12008.6386719, Throughput: 172.930627 im/s
Epoch 8, Train Loss: 12728.5654297, Time: 28.8672s, Throughput: 172.929722 im/s
Epoch 9, Iter 39, Loss: 12593.3154297, Throughput: 172.463422 im/s
Epoch 9, Train Loss: 12242.7343750, Time: 28.9455s, Throughput: 172.462290 im/s
Epoch 10, Iter 39, Loss: 12947.8281250, Throughput: 173.362100 im/s
Epoch 10, Train Loss: 12027.2197266, Time: 28.7954s, Throughput: 173.361179 im/s
Epoch 11, Iter 39, Loss: 11274.9433594, Throughput: 172.575172 im/s
Epoch 11, Train Loss: 11908.2080078, Time: 28.9267s, Throughput: 172.574217 im/s
Epoch 12, Iter 39, Loss: 12018.0937500, Throughput: 172.723692 im/s
Epoch 12, Train Loss: 11700.2783203, Time: 28.9018s, Throughput: 172.722812 im/s
Epoch 13, Iter 39, Loss: 12802.8222656, Throughput: 172.442488 im/s
Epoch 13, Train Loss: 11421.5068359, Time: 28.9489s, Throughput: 172.441541 im/s
Epoch 14, Iter 39, Loss: 11555.8554688, Throughput: 173.106355 im/s
Epoch 14, Train Loss: 11341.8798828, Time: 28.8379s, Throughput: 173.105430 im/s
Epoch 15, Iter 39, Loss: 11052.7968750, Throughput: 172.353867 im/s
Epoch 15, Train Loss: 11131.1777344, Time: 28.9639s, Throughput: 172.352772 im/s
Epoch 16, Iter 39, Loss: 10596.0097656, Throughput: 173.443500 im/s
Epoch 16, Train Loss: 11053.9277344, Time: 28.7818s, Throughput: 173.442662 im/s
Epoch 17, Iter 39, Loss: 10688.7080078, Throughput: 172.425323 im/s
Epoch 17, Train Loss: 10941.3193359, Time: 28.9518s, Throughput: 172.424447 im/s
Epoch 18, Iter 39, Loss: 10738.1787109, Throughput: 173.902187 im/s
Epoch 18, Train Loss: 10950.6816406, Time: 28.7060s, Throughput: 173.901175 im/s
Epoch 19, Iter 39, Loss: 10727.8193359, Throughput: 173.418043 im/s
Epoch 19, Train Loss: 10676.8593750, Time: 28.7861s, Throughput: 173.417151 im/s
Epoch 20, Iter 39, Loss: 10934.0996094, Throughput: 173.040311 im/s
Epoch 20, Train Loss: 10668.0205078, Time: 28.8489s, Throughput: 173.039412 im/s
Epoch 21, Iter 39, Loss: 10123.9570312, Throughput: 173.405711 im/s
Epoch 21, Train Loss: 10509.9658203, Time: 28.7882s, Throughput: 173.404663 im/s
Epoch 22, Iter 39, Loss: 10133.0634766, Throughput: 172.937228 im/s
Epoch 22, Train Loss: 10576.9970703, Time: 28.8661s, Throughput: 172.936276 im/s
Epoch 23, Iter 39, Loss: 9948.9599609, Throughput: 172.382458 im/s
Epoch 23, Train Loss: 10410.6835938, Time: 28.9590s, Throughput: 172.381488 im/s
Epoch 24, Iter 39, Loss: 10999.0976562, Throughput: 171.161502 im/s
Epoch 24, Train Loss: 10308.9697266, Time: 29.1656s, Throughput: 171.160516 im/s
Epoch 25, Iter 39, Loss: 10218.6582031, Throughput: 172.930130 im/s
Epoch 25, Train Loss: 10222.9335938, Time: 28.8673s, Throughput: 172.929195 im/s
Epoch 26, Iter 39, Loss: 10581.6894531, Throughput: 173.275804 im/s
Epoch 26, Train Loss: 10171.7607422, Time: 28.8097s, Throughput: 173.274920 im/s
Epoch 27, Iter 39, Loss: 10799.0244141, Throughput: 173.505664 im/s
Epoch 27, Train Loss: 10218.5234375, Time: 28.7715s, Throughput: 173.504796 im/s
Epoch 28, Iter 39, Loss: 10141.2001953, Throughput: 173.020443 im/s
Epoch 28, Train Loss: 10113.0830078, Time: 28.8522s, Throughput: 173.019478 im/s
Epoch 29, Iter 39, Loss: 10561.2646484, Throughput: 173.410271 im/s
Epoch 29, Train Loss: 10100.9082031, Time: 28.7874s, Throughput: 173.409368 im/s
Epoch 30, Iter 39, Loss: 10020.9335938, Throughput: 172.409417 im/s
Epoch 30, Train Loss: 9985.2275391, Time: 28.9545s, Throughput: 172.408517 im/s
Epoch 31, Iter 39, Loss: 9436.9062500, Throughput: 173.609982 im/s
Epoch 31, Train Loss: 9877.6269531, Time: 28.7543s, Throughput: 173.609052 im/s
Epoch 32, Iter 39, Loss: 10412.2246094, Throughput: 173.104949 im/s
Epoch 32, Train Loss: 9923.4482422, Time: 28.8382s, Throughput: 173.104013 im/s
Epoch 33, Iter 39, Loss: 10068.6992188, Throughput: 173.887733 im/s
Epoch 33, Train Loss: 9850.6621094, Time: 28.7083s, Throughput: 173.886813 im/s
Epoch 34, Iter 39, Loss: 9964.0361328, Throughput: 173.429697 im/s
Epoch 34, Train Loss: 9765.9501953, Time: 28.7842s, Throughput: 173.428714 im/s
Epoch 35, Iter 39, Loss: 9459.3398438, Throughput: 173.360752 im/s
Epoch 35, Train Loss: 9776.6894531, Time: 28.7956s, Throughput: 173.359842 im/s
Epoch 36, Iter 39, Loss: 9594.2060547, Throughput: 173.438046 im/s
Epoch 36, Train Loss: 9671.0332031, Time: 28.7828s, Throughput: 173.437052 im/s
Epoch 37, Iter 39, Loss: 9668.5546875, Throughput: 173.439391 im/s
Epoch 37, Train Loss: 9764.1123047, Time: 28.7826s, Throughput: 173.438384 im/s
Epoch 38, Iter 39, Loss: 9715.4003906, Throughput: 172.155443 im/s
Epoch 38, Train Loss: 9666.9365234, Time: 28.9972s, Throughput: 172.154440 im/s
Epoch 39, Iter 39, Loss: 10230.2011719, Throughput: 173.520982 im/s
Epoch 39, Train Loss: 9625.6855469, Time: 28.7690s, Throughput: 173.519885 im/s
Epoch 40, Iter 39, Loss: 10271.9863281, Throughput: 172.787809 im/s
Epoch 40, Train Loss: 9528.1718750, Time: 28.8911s, Throughput: 172.786894 im/s
Epoch 41, Iter 39, Loss: 9920.9365234, Throughput: 173.213408 im/s
Epoch 41, Train Loss: 9506.3457031, Time: 28.8201s, Throughput: 173.212507 im/s
Epoch 42, Iter 39, Loss: 9731.9492188, Throughput: 173.444278 im/s
Epoch 42, Train Loss: 9580.8955078, Time: 28.7818s, Throughput: 173.443187 im/s
Epoch 43, Iter 39, Loss: 9204.1767578, Throughput: 174.133914 im/s
Epoch 43, Train Loss: 9479.9716797, Time: 28.6678s, Throughput: 174.132881 im/s
Epoch 44, Iter 39, Loss: 9930.4785156, Throughput: 173.445085 im/s
Epoch 44, Train Loss: 9463.5976562, Time: 28.7816s, Throughput: 173.444187 im/s
Epoch 45, Iter 39, Loss: 9651.7626953, Throughput: 172.885876 im/s
Epoch 45, Train Loss: 9480.6464844, Time: 28.8747s, Throughput: 172.884978 im/s
Epoch 46, Iter 39, Loss: 9037.3310547, Throughput: 173.231313 im/s
Epoch 46, Train Loss: 9416.8144531, Time: 28.8171s, Throughput: 173.230406 im/s
Epoch 47, Iter 39, Loss: 9042.7890625, Throughput: 170.730388 im/s
Epoch 47, Train Loss: 9323.1914062, Time: 29.2392s, Throughput: 170.729442 im/s
Epoch 48, Iter 39, Loss: 9384.7080078, Throughput: 172.725884 im/s
Epoch 48, Train Loss: 9350.6113281, Time: 28.9015s, Throughput: 172.724861 im/s
Epoch 49, Iter 39, Loss: 9241.6035156, Throughput: 173.124580 im/s
Epoch 49, Train Loss: 9362.9423828, Time: 28.8349s, Throughput: 173.123542 im/s
Epoch 50, Iter 39, Loss: 9041.9824219, Throughput: 173.247569 im/s
Epoch 50, Train Loss: 9315.7812500, Time: 28.8144s, Throughput: 173.246680 im/s

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
