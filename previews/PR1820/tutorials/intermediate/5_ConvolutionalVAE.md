---
url: /previews/PR1820/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24481.7382812, Throughput: 15.284101 im/s
Epoch 1, Train Loss: 39520.9570312, Time: 326.8814s, Throughput: 15.271596 im/s
Epoch 2, Iter 39, Loss: 17752.1718750, Throughput: 97.736719 im/s
Epoch 2, Train Loss: 20015.5058594, Time: 51.0762s, Throughput: 97.736369 im/s
Epoch 3, Iter 39, Loss: 15134.4560547, Throughput: 97.867084 im/s
Epoch 3, Train Loss: 16439.9511719, Time: 51.0081s, Throughput: 97.866758 im/s
Epoch 4, Iter 39, Loss: 15449.5214844, Throughput: 98.057291 im/s
Epoch 4, Train Loss: 14939.6230469, Time: 50.9092s, Throughput: 98.056958 im/s
Epoch 5, Iter 39, Loss: 13644.2783203, Throughput: 97.603067 im/s
Epoch 5, Train Loss: 13921.8349609, Time: 51.1461s, Throughput: 97.602778 im/s
Epoch 6, Iter 39, Loss: 13098.6416016, Throughput: 97.854438 im/s
Epoch 6, Train Loss: 13294.9121094, Time: 51.0147s, Throughput: 97.854092 im/s
Epoch 7, Iter 39, Loss: 13092.9433594, Throughput: 98.207483 im/s
Epoch 7, Train Loss: 12887.4003906, Time: 50.8313s, Throughput: 98.207180 im/s
Epoch 8, Iter 39, Loss: 12231.7470703, Throughput: 98.047126 im/s
Epoch 8, Train Loss: 12547.9931641, Time: 50.9144s, Throughput: 98.046839 im/s
Epoch 9, Iter 39, Loss: 11947.9033203, Throughput: 98.098123 im/s
Epoch 9, Train Loss: 12109.3144531, Time: 50.8880s, Throughput: 98.097818 im/s
Epoch 10, Iter 39, Loss: 11975.8056641, Throughput: 97.898558 im/s
Epoch 10, Train Loss: 11895.2744141, Time: 50.9917s, Throughput: 97.898220 im/s
Epoch 11, Iter 39, Loss: 11588.0517578, Throughput: 97.633793 im/s
Epoch 11, Train Loss: 11661.0976562, Time: 51.1300s, Throughput: 97.633493 im/s
Epoch 12, Iter 39, Loss: 11205.2832031, Throughput: 98.046259 im/s
Epoch 12, Train Loss: 11564.8994141, Time: 50.9149s, Throughput: 98.045903 im/s
Epoch 13, Iter 39, Loss: 11035.6464844, Throughput: 97.958482 im/s
Epoch 13, Train Loss: 11366.8740234, Time: 50.9605s, Throughput: 97.958206 im/s
Epoch 14, Iter 39, Loss: 11582.1640625, Throughput: 97.990981 im/s
Epoch 14, Train Loss: 11152.2744141, Time: 50.9436s, Throughput: 97.990688 im/s
Epoch 15, Iter 39, Loss: 10629.9287109, Throughput: 98.111532 im/s
Epoch 15, Train Loss: 11122.4970703, Time: 50.8810s, Throughput: 98.111254 im/s
Epoch 16, Iter 39, Loss: 10271.3945312, Throughput: 98.357475 im/s
Epoch 16, Train Loss: 10968.6181641, Time: 50.7538s, Throughput: 98.357169 im/s
Epoch 17, Iter 39, Loss: 10571.8183594, Throughput: 98.246849 im/s
Epoch 17, Train Loss: 10900.9355469, Time: 50.8110s, Throughput: 98.246516 im/s
Epoch 18, Iter 39, Loss: 11145.9218750, Throughput: 98.549596 im/s
Epoch 18, Train Loss: 10751.1894531, Time: 50.6548s, Throughput: 98.549317 im/s
Epoch 19, Iter 39, Loss: 10988.6992188, Throughput: 98.383454 im/s
Epoch 19, Train Loss: 10731.9550781, Time: 50.7404s, Throughput: 98.383125 im/s
Epoch 20, Iter 39, Loss: 10717.8212891, Throughput: 98.445162 im/s
Epoch 20, Train Loss: 10645.7695312, Time: 50.7086s, Throughput: 98.444875 im/s
Epoch 21, Iter 39, Loss: 10560.1523438, Throughput: 98.606576 im/s
Epoch 21, Train Loss: 10579.1992188, Time: 50.6256s, Throughput: 98.606287 im/s
Epoch 22, Iter 39, Loss: 10567.7324219, Throughput: 98.602799 im/s
Epoch 22, Train Loss: 10461.5322266, Time: 50.6275s, Throughput: 98.602499 im/s
Epoch 23, Iter 39, Loss: 10764.8916016, Throughput: 98.541493 im/s
Epoch 23, Train Loss: 10363.6435547, Time: 50.6590s, Throughput: 98.541162 im/s
Epoch 24, Iter 39, Loss: 10481.4658203, Throughput: 98.891622 im/s
Epoch 24, Train Loss: 10345.4570312, Time: 50.4797s, Throughput: 98.891240 im/s
Epoch 25, Iter 39, Loss: 10390.3261719, Throughput: 98.545777 im/s
Epoch 25, Train Loss: 10168.5458984, Time: 50.6568s, Throughput: 98.545426 im/s
Epoch 26, Iter 39, Loss: 9271.7382812, Throughput: 98.766425 im/s
Epoch 26, Train Loss: 10227.8330078, Time: 50.5437s, Throughput: 98.766105 im/s
Epoch 27, Iter 39, Loss: 9939.1064453, Throughput: 98.700616 im/s
Epoch 27, Train Loss: 10123.9453125, Time: 50.5773s, Throughput: 98.700329 im/s
Epoch 28, Iter 39, Loss: 10021.6972656, Throughput: 97.953707 im/s
Epoch 28, Train Loss: 10094.1269531, Time: 50.9630s, Throughput: 97.953379 im/s
Epoch 29, Iter 39, Loss: 10593.4277344, Throughput: 98.941030 im/s
Epoch 29, Train Loss: 10099.8867188, Time: 50.4544s, Throughput: 98.940734 im/s
Epoch 30, Iter 39, Loss: 10213.2421875, Throughput: 98.810180 im/s
Epoch 30, Train Loss: 9993.7011719, Time: 50.5213s, Throughput: 98.809811 im/s
Epoch 31, Iter 39, Loss: 9993.5917969, Throughput: 98.916186 im/s
Epoch 31, Train Loss: 9895.9707031, Time: 50.4671s, Throughput: 98.915880 im/s
Epoch 32, Iter 39, Loss: 10022.4443359, Throughput: 98.906542 im/s
Epoch 32, Train Loss: 9953.1972656, Time: 50.4721s, Throughput: 98.906203 im/s
Epoch 33, Iter 39, Loss: 9366.9453125, Throughput: 99.004697 im/s
Epoch 33, Train Loss: 9958.5908203, Time: 50.4220s, Throughput: 99.004409 im/s
Epoch 34, Iter 39, Loss: 10221.5615234, Throughput: 98.834727 im/s
Epoch 34, Train Loss: 9826.1015625, Time: 50.5087s, Throughput: 98.834425 im/s
Epoch 35, Iter 39, Loss: 9784.0937500, Throughput: 98.982881 im/s
Epoch 35, Train Loss: 9836.5781250, Time: 50.4331s, Throughput: 98.982518 im/s
Epoch 36, Iter 39, Loss: 10130.7373047, Throughput: 98.990941 im/s
Epoch 36, Train Loss: 9686.7050781, Time: 50.4290s, Throughput: 98.990621 im/s
Epoch 37, Iter 39, Loss: 10374.1855469, Throughput: 98.905421 im/s
Epoch 37, Train Loss: 9673.9824219, Time: 50.4726s, Throughput: 98.905131 im/s
Epoch 38, Iter 39, Loss: 9425.1289062, Throughput: 99.069896 im/s
Epoch 38, Train Loss: 9674.4531250, Time: 50.3888s, Throughput: 99.069579 im/s
Epoch 39, Iter 39, Loss: 10072.0107422, Throughput: 98.848414 im/s
Epoch 39, Train Loss: 9636.8310547, Time: 50.5017s, Throughput: 98.848103 im/s
Epoch 40, Iter 39, Loss: 10534.4062500, Throughput: 99.211889 im/s
Epoch 40, Train Loss: 9671.3105469, Time: 50.3167s, Throughput: 99.211582 im/s
Epoch 41, Iter 39, Loss: 9836.2871094, Throughput: 98.966586 im/s
Epoch 41, Train Loss: 9644.8876953, Time: 50.4414s, Throughput: 98.966258 im/s
Epoch 42, Iter 39, Loss: 10041.7871094, Throughput: 99.128986 im/s
Epoch 42, Train Loss: 9614.2187500, Time: 50.3588s, Throughput: 99.128702 im/s
Epoch 43, Iter 39, Loss: 9267.8115234, Throughput: 99.091253 im/s
Epoch 43, Train Loss: 9472.9960938, Time: 50.3780s, Throughput: 99.090946 im/s
Epoch 44, Iter 39, Loss: 9599.9140625, Throughput: 98.977844 im/s
Epoch 44, Train Loss: 9467.2636719, Time: 50.4357s, Throughput: 98.977545 im/s
Epoch 45, Iter 39, Loss: 9180.3398438, Throughput: 98.793759 im/s
Epoch 45, Train Loss: 9409.0937500, Time: 50.5297s, Throughput: 98.793361 im/s
Epoch 46, Iter 39, Loss: 9174.3613281, Throughput: 98.803756 im/s
Epoch 46, Train Loss: 9382.0292969, Time: 50.5246s, Throughput: 98.803427 im/s
Epoch 47, Iter 39, Loss: 9177.2070312, Throughput: 98.642371 im/s
Epoch 47, Train Loss: 9319.6337891, Time: 50.6072s, Throughput: 98.642058 im/s
Epoch 48, Iter 39, Loss: 9371.7783203, Throughput: 98.892501 im/s
Epoch 48, Train Loss: 9316.7490234, Time: 50.4792s, Throughput: 98.892166 im/s
Epoch 49, Iter 39, Loss: 9899.1933594, Throughput: 99.011662 im/s
Epoch 49, Train Loss: 9400.5625000, Time: 50.4185s, Throughput: 99.011278 im/s
Epoch 50, Iter 39, Loss: 9456.5292969, Throughput: 98.929726 im/s
Epoch 50, Train Loss: 9330.8535156, Time: 50.4602s, Throughput: 98.929428 im/s

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
  CPU: 4 × Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, icelake-server)
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
