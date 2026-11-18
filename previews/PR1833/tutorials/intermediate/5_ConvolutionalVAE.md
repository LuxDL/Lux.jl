---
url: /previews/PR1833/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23968.4433594, Throughput: 16.650300 im/s
Epoch 1, Train Loss: 39506.5546875, Time: 300.0744s, Throughput: 16.635872 im/s
Epoch 2, Iter 39, Loss: 18427.6796875, Throughput: 174.902771 im/s
Epoch 2, Train Loss: 20205.4921875, Time: 28.5417s, Throughput: 174.901863 im/s
Epoch 3, Iter 39, Loss: 15159.6962891, Throughput: 174.383778 im/s
Epoch 3, Train Loss: 16594.6015625, Time: 28.6267s, Throughput: 174.382810 im/s
Epoch 4, Iter 39, Loss: 14572.1582031, Throughput: 175.071403 im/s
Epoch 4, Train Loss: 14925.5693359, Time: 28.5142s, Throughput: 175.070592 im/s
Epoch 5, Iter 39, Loss: 13588.5341797, Throughput: 173.974775 im/s
Epoch 5, Train Loss: 13979.4531250, Time: 28.6940s, Throughput: 173.973854 im/s
Epoch 6, Iter 39, Loss: 13196.3681641, Throughput: 175.132982 im/s
Epoch 6, Train Loss: 13308.6513672, Time: 28.5042s, Throughput: 175.132117 im/s
Epoch 7, Iter 39, Loss: 13008.6660156, Throughput: 174.200702 im/s
Epoch 7, Train Loss: 12829.3554688, Time: 28.6567s, Throughput: 174.199808 im/s
Epoch 8, Iter 39, Loss: 12626.8359375, Throughput: 174.965182 im/s
Epoch 8, Train Loss: 12437.6953125, Time: 28.5315s, Throughput: 174.964287 im/s
Epoch 9, Iter 39, Loss: 11681.3457031, Throughput: 174.527135 im/s
Epoch 9, Train Loss: 12104.2451172, Time: 28.6032s, Throughput: 174.526183 im/s
Epoch 10, Iter 39, Loss: 12229.2021484, Throughput: 175.148635 im/s
Epoch 10, Train Loss: 11961.6406250, Time: 28.5017s, Throughput: 175.147565 im/s
Epoch 11, Iter 39, Loss: 11965.9775391, Throughput: 174.499633 im/s
Epoch 11, Train Loss: 11745.2392578, Time: 28.6077s, Throughput: 174.498735 im/s
Epoch 12, Iter 39, Loss: 12172.1718750, Throughput: 174.774060 im/s
Epoch 12, Train Loss: 11427.9892578, Time: 28.5628s, Throughput: 174.772995 im/s
Epoch 13, Iter 39, Loss: 11568.7968750, Throughput: 174.214812 im/s
Epoch 13, Train Loss: 11368.3525391, Time: 28.6544s, Throughput: 174.213809 im/s
Epoch 14, Iter 39, Loss: 11350.6347656, Throughput: 175.091839 im/s
Epoch 14, Train Loss: 11109.9394531, Time: 28.5109s, Throughput: 175.090721 im/s
Epoch 15, Iter 39, Loss: 11206.2128906, Throughput: 174.424802 im/s
Epoch 15, Train Loss: 11015.8515625, Time: 28.6199s, Throughput: 174.424040 im/s
Epoch 16, Iter 39, Loss: 10905.6582031, Throughput: 174.539926 im/s
Epoch 16, Train Loss: 10941.3818359, Time: 28.6011s, Throughput: 174.538889 im/s
Epoch 17, Iter 39, Loss: 11829.4580078, Throughput: 174.843257 im/s
Epoch 17, Train Loss: 10873.8251953, Time: 28.5514s, Throughput: 174.842314 im/s
Epoch 18, Iter 39, Loss: 10431.6894531, Throughput: 175.352623 im/s
Epoch 18, Train Loss: 10712.6044922, Time: 28.4685s, Throughput: 175.351539 im/s
Epoch 19, Iter 39, Loss: 10433.3046875, Throughput: 174.357515 im/s
Epoch 19, Train Loss: 10715.5097656, Time: 28.6310s, Throughput: 174.356497 im/s
Epoch 20, Iter 39, Loss: 10379.6621094, Throughput: 175.062158 im/s
Epoch 20, Train Loss: 10599.3134766, Time: 28.5157s, Throughput: 175.061212 im/s
Epoch 21, Iter 39, Loss: 10947.8906250, Throughput: 173.153363 im/s
Epoch 21, Train Loss: 10470.5947266, Time: 28.8301s, Throughput: 173.152449 im/s
Epoch 22, Iter 39, Loss: 10074.9023438, Throughput: 174.446257 im/s
Epoch 22, Train Loss: 10405.8886719, Time: 28.6164s, Throughput: 174.445362 im/s
Epoch 23, Iter 39, Loss: 10427.4218750, Throughput: 174.599587 im/s
Epoch 23, Train Loss: 10370.1250000, Time: 28.5913s, Throughput: 174.598578 im/s
Epoch 24, Iter 39, Loss: 10341.5117188, Throughput: 174.488727 im/s
Epoch 24, Train Loss: 10297.5458984, Time: 28.6095s, Throughput: 174.487507 im/s
Epoch 25, Iter 39, Loss: 9901.4824219, Throughput: 175.010426 im/s
Epoch 25, Train Loss: 10221.9248047, Time: 28.5242s, Throughput: 175.009469 im/s
Epoch 26, Iter 39, Loss: 10942.9921875, Throughput: 174.057661 im/s
Epoch 26, Train Loss: 10133.8271484, Time: 28.6803s, Throughput: 174.056768 im/s
Epoch 27, Iter 39, Loss: 10815.4042969, Throughput: 174.982397 im/s
Epoch 27, Train Loss: 10100.0859375, Time: 28.5287s, Throughput: 174.981433 im/s
Epoch 28, Iter 39, Loss: 9929.2675781, Throughput: 174.539041 im/s
Epoch 28, Train Loss: 10024.7968750, Time: 28.6012s, Throughput: 174.538174 im/s
Epoch 29, Iter 39, Loss: 10409.9804688, Throughput: 175.311375 im/s
Epoch 29, Train Loss: 10088.6582031, Time: 28.4752s, Throughput: 175.310242 im/s
Epoch 30, Iter 39, Loss: 9743.1250000, Throughput: 173.889138 im/s
Epoch 30, Train Loss: 9863.3251953, Time: 28.7081s, Throughput: 173.888218 im/s
Epoch 31, Iter 39, Loss: 9570.9062500, Throughput: 174.589034 im/s
Epoch 31, Train Loss: 9860.4482422, Time: 28.5930s, Throughput: 174.587911 im/s
Epoch 32, Iter 39, Loss: 9574.8125000, Throughput: 174.069181 im/s
Epoch 32, Train Loss: 9810.2578125, Time: 28.6784s, Throughput: 174.068307 im/s
Epoch 33, Iter 39, Loss: 9012.6796875, Throughput: 174.495168 im/s
Epoch 33, Train Loss: 9806.6816406, Time: 28.6084s, Throughput: 174.494405 im/s
Epoch 34, Iter 39, Loss: 9763.0761719, Throughput: 174.247552 im/s
Epoch 34, Train Loss: 9724.1689453, Time: 28.6491s, Throughput: 174.246566 im/s
Epoch 35, Iter 39, Loss: 10151.6533203, Throughput: 174.806449 im/s
Epoch 35, Train Loss: 9703.0605469, Time: 28.5575s, Throughput: 174.805493 im/s
Epoch 36, Iter 39, Loss: 9933.1152344, Throughput: 174.163598 im/s
Epoch 36, Train Loss: 9676.4511719, Time: 28.6629s, Throughput: 174.162687 im/s
Epoch 37, Iter 39, Loss: 9324.4238281, Throughput: 174.574815 im/s
Epoch 37, Train Loss: 9621.7119141, Time: 28.5954s, Throughput: 174.573815 im/s
Epoch 38, Iter 39, Loss: 9368.1503906, Throughput: 174.142601 im/s
Epoch 38, Train Loss: 9620.2705078, Time: 28.6663s, Throughput: 174.141676 im/s
Epoch 39, Iter 39, Loss: 9431.2226562, Throughput: 174.421030 im/s
Epoch 39, Train Loss: 9580.2294922, Time: 28.6206s, Throughput: 174.420121 im/s
Epoch 40, Iter 39, Loss: 9567.5878906, Throughput: 173.749283 im/s
Epoch 40, Train Loss: 9620.0644531, Time: 28.7312s, Throughput: 173.748279 im/s
Epoch 41, Iter 39, Loss: 10339.6630859, Throughput: 173.105105 im/s
Epoch 41, Train Loss: 9535.1679688, Time: 28.8381s, Throughput: 173.104109 im/s
Epoch 42, Iter 39, Loss: 8904.5839844, Throughput: 173.761105 im/s
Epoch 42, Train Loss: 9430.7167969, Time: 28.7293s, Throughput: 173.759768 im/s
Epoch 43, Iter 39, Loss: 9315.8964844, Throughput: 175.089590 im/s
Epoch 43, Train Loss: 9414.4570312, Time: 28.5113s, Throughput: 175.088663 im/s
Epoch 44, Iter 39, Loss: 9813.0898438, Throughput: 174.679402 im/s
Epoch 44, Train Loss: 9476.8056641, Time: 28.5783s, Throughput: 174.678174 im/s
Epoch 45, Iter 39, Loss: 10002.7763672, Throughput: 174.302796 im/s
Epoch 45, Train Loss: 9383.6269531, Time: 28.6400s, Throughput: 174.301883 im/s
Epoch 46, Iter 39, Loss: 9267.9365234, Throughput: 175.174054 im/s
Epoch 46, Train Loss: 9294.4042969, Time: 28.4975s, Throughput: 175.173096 im/s
Epoch 47, Iter 39, Loss: 9276.5039062, Throughput: 174.296880 im/s
Epoch 47, Train Loss: 9347.5253906, Time: 28.6409s, Throughput: 174.295943 im/s
Epoch 48, Iter 39, Loss: 9384.6503906, Throughput: 174.397600 im/s
Epoch 48, Train Loss: 9333.4824219, Time: 28.6244s, Throughput: 174.396400 im/s
Epoch 49, Iter 39, Loss: 9293.3417969, Throughput: 174.199121 im/s
Epoch 49, Train Loss: 9260.2724609, Time: 28.6570s, Throughput: 174.198190 im/s
Epoch 50, Iter 39, Loss: 9703.9882812, Throughput: 175.425773 im/s
Epoch 50, Train Loss: 9233.1142578, Time: 28.4566s, Throughput: 175.424775 im/s

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
