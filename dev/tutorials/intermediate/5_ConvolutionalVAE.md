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
Epoch 1, Iter 39, Loss: 24096.6464844, Throughput: 11.772989 im/s
Epoch 1, Train Loss: 39563.7929688, Time: 424.3200s, Throughput: 11.764707 im/s
Epoch 2, Iter 39, Loss: 17849.1757812, Throughput: 93.480732 im/s
Epoch 2, Train Loss: 20062.4218750, Time: 53.4016s, Throughput: 93.480417 im/s
Epoch 3, Iter 39, Loss: 15436.8613281, Throughput: 92.499053 im/s
Epoch 3, Train Loss: 16543.7265625, Time: 53.9683s, Throughput: 92.498717 im/s
Epoch 4, Iter 39, Loss: 14981.2265625, Throughput: 92.258111 im/s
Epoch 4, Train Loss: 15002.7968750, Time: 54.1093s, Throughput: 92.257787 im/s
Epoch 5, Iter 39, Loss: 14264.9472656, Throughput: 92.837134 im/s
Epoch 5, Train Loss: 13977.5751953, Time: 53.7718s, Throughput: 92.836819 im/s
Epoch 6, Iter 39, Loss: 12952.0195312, Throughput: 92.915322 im/s
Epoch 6, Train Loss: 13331.1875000, Time: 53.7265s, Throughput: 92.914961 im/s
Epoch 7, Iter 39, Loss: 12415.9931641, Throughput: 93.033351 im/s
Epoch 7, Train Loss: 12859.9052734, Time: 53.6584s, Throughput: 93.032988 im/s
Epoch 8, Iter 39, Loss: 11956.2246094, Throughput: 93.176357 im/s
Epoch 8, Train Loss: 12426.7890625, Time: 53.5760s, Throughput: 93.176040 im/s
Epoch 9, Iter 39, Loss: 12050.7050781, Throughput: 93.464936 im/s
Epoch 9, Train Loss: 12159.0458984, Time: 53.4106s, Throughput: 93.464579 im/s
Epoch 10, Iter 39, Loss: 12668.9716797, Throughput: 93.131251 im/s
Epoch 10, Train Loss: 12053.6689453, Time: 53.6020s, Throughput: 93.130925 im/s
Epoch 11, Iter 39, Loss: 11353.0283203, Throughput: 93.102258 im/s
Epoch 11, Train Loss: 11735.3261719, Time: 53.6186s, Throughput: 93.101941 im/s
Epoch 12, Iter 39, Loss: 12297.8671875, Throughput: 93.436014 im/s
Epoch 12, Train Loss: 11609.3105469, Time: 53.4271s, Throughput: 93.435676 im/s
Epoch 13, Iter 39, Loss: 12212.6562500, Throughput: 92.997243 im/s
Epoch 13, Train Loss: 11424.1757812, Time: 53.6792s, Throughput: 92.996944 im/s
Epoch 14, Iter 39, Loss: 11718.1601562, Throughput: 93.379254 im/s
Epoch 14, Train Loss: 11277.3857422, Time: 53.4596s, Throughput: 93.378922 im/s
Epoch 15, Iter 39, Loss: 10822.5263672, Throughput: 93.113899 im/s
Epoch 15, Train Loss: 11201.0156250, Time: 53.6119s, Throughput: 93.113593 im/s
Epoch 16, Iter 39, Loss: 11084.4775391, Throughput: 91.905326 im/s
Epoch 16, Train Loss: 11074.6494141, Time: 54.3170s, Throughput: 91.904974 im/s
Epoch 17, Iter 39, Loss: 10878.3818359, Throughput: 91.618860 im/s
Epoch 17, Train Loss: 10928.5439453, Time: 54.4870s, Throughput: 91.618225 im/s
Epoch 18, Iter 39, Loss: 10655.6914062, Throughput: 92.576687 im/s
Epoch 18, Train Loss: 10929.1826172, Time: 53.9231s, Throughput: 92.576352 im/s
Epoch 19, Iter 39, Loss: 10615.9082031, Throughput: 93.683095 im/s
Epoch 19, Train Loss: 10765.4619141, Time: 53.2862s, Throughput: 93.682733 im/s
Epoch 20, Iter 39, Loss: 10472.5869141, Throughput: 93.269893 im/s
Epoch 20, Train Loss: 10696.2138672, Time: 53.5223s, Throughput: 93.269607 im/s
Epoch 21, Iter 39, Loss: 11046.2470703, Throughput: 92.612282 im/s
Epoch 21, Train Loss: 10722.1972656, Time: 53.9034s, Throughput: 92.610095 im/s
Epoch 22, Iter 39, Loss: 10533.2792969, Throughput: 92.803289 im/s
Epoch 22, Train Loss: 10546.6318359, Time: 53.7914s, Throughput: 92.802989 im/s
Epoch 23, Iter 39, Loss: 10596.2187500, Throughput: 93.011318 im/s
Epoch 23, Train Loss: 10465.4931641, Time: 53.6711s, Throughput: 93.010950 im/s
Epoch 24, Iter 39, Loss: 10409.6015625, Throughput: 92.763984 im/s
Epoch 24, Train Loss: 10374.1933594, Time: 53.8142s, Throughput: 92.763679 im/s
Epoch 25, Iter 39, Loss: 10915.7871094, Throughput: 93.161247 im/s
Epoch 25, Train Loss: 10310.9707031, Time: 53.5847s, Throughput: 93.160978 im/s
Epoch 26, Iter 39, Loss: 10414.3574219, Throughput: 93.635857 im/s
Epoch 26, Train Loss: 10320.1552734, Time: 53.3131s, Throughput: 93.635542 im/s
Epoch 27, Iter 39, Loss: 9771.1582031, Throughput: 92.719876 im/s
Epoch 27, Train Loss: 10204.7558594, Time: 53.8398s, Throughput: 92.719540 im/s
Epoch 28, Iter 39, Loss: 10548.2724609, Throughput: 93.098153 im/s
Epoch 28, Train Loss: 10173.8222656, Time: 53.6210s, Throughput: 93.097845 im/s
Epoch 29, Iter 39, Loss: 9918.7089844, Throughput: 92.879174 im/s
Epoch 29, Train Loss: 10187.7236328, Time: 53.7474s, Throughput: 92.878846 im/s
Epoch 30, Iter 39, Loss: 9480.1767578, Throughput: 92.535964 im/s
Epoch 30, Train Loss: 10060.2441406, Time: 53.9468s, Throughput: 92.535661 im/s
Epoch 31, Iter 39, Loss: 9963.9492188, Throughput: 92.703518 im/s
Epoch 31, Train Loss: 10091.0937500, Time: 53.8493s, Throughput: 92.703165 im/s
Epoch 32, Iter 39, Loss: 10007.0742188, Throughput: 91.066966 im/s
Epoch 32, Train Loss: 9990.1289062, Time: 54.8170s, Throughput: 91.066658 im/s
Epoch 33, Iter 39, Loss: 9902.7656250, Throughput: 92.377999 im/s
Epoch 33, Train Loss: 9914.0332031, Time: 54.0391s, Throughput: 92.377532 im/s
Epoch 34, Iter 39, Loss: 9745.6015625, Throughput: 92.426880 im/s
Epoch 34, Train Loss: 9928.6386719, Time: 54.0104s, Throughput: 92.426608 im/s
Epoch 35, Iter 39, Loss: 10084.4042969, Throughput: 92.009261 im/s
Epoch 35, Train Loss: 9917.5849609, Time: 54.2556s, Throughput: 92.008942 im/s
Epoch 36, Iter 39, Loss: 10085.3652344, Throughput: 92.068374 im/s
Epoch 36, Train Loss: 9893.0576172, Time: 54.2208s, Throughput: 92.068031 im/s
Epoch 37, Iter 39, Loss: 9949.9902344, Throughput: 92.509806 im/s
Epoch 37, Train Loss: 9843.1132812, Time: 53.9620s, Throughput: 92.509467 im/s
Epoch 38, Iter 39, Loss: 9780.0908203, Throughput: 92.422385 im/s
Epoch 38, Train Loss: 9825.3818359, Time: 54.0131s, Throughput: 92.422000 im/s
Epoch 39, Iter 39, Loss: 9850.6914062, Throughput: 92.889642 im/s
Epoch 39, Train Loss: 9747.0781250, Time: 53.7414s, Throughput: 92.889357 im/s
Epoch 40, Iter 39, Loss: 9469.6494141, Throughput: 92.081757 im/s
Epoch 40, Train Loss: 9709.0693359, Time: 54.2129s, Throughput: 92.081424 im/s
Epoch 41, Iter 39, Loss: 9799.9248047, Throughput: 92.590312 im/s
Epoch 41, Train Loss: 9675.8769531, Time: 53.9151s, Throughput: 92.589948 im/s
Epoch 42, Iter 39, Loss: 9373.8193359, Throughput: 92.767701 im/s
Epoch 42, Train Loss: 9689.1406250, Time: 53.8120s, Throughput: 92.767407 im/s
Epoch 43, Iter 39, Loss: 9679.8535156, Throughput: 92.935068 im/s
Epoch 43, Train Loss: 9621.4912109, Time: 53.7151s, Throughput: 92.934741 im/s
Epoch 44, Iter 39, Loss: 9526.9472656, Throughput: 93.422945 im/s
Epoch 44, Train Loss: 9606.1435547, Time: 53.4346s, Throughput: 93.422623 im/s
Epoch 45, Iter 39, Loss: 9014.1562500, Throughput: 92.609820 im/s
Epoch 45, Train Loss: 9530.8730469, Time: 53.9038s, Throughput: 92.609466 im/s
Epoch 46, Iter 39, Loss: 9784.6875000, Throughput: 93.158803 im/s
Epoch 46, Train Loss: 9530.3232422, Time: 53.5861s, Throughput: 93.158486 im/s
Epoch 47, Iter 39, Loss: 9791.7021484, Throughput: 92.419886 im/s
Epoch 47, Train Loss: 9568.5039062, Time: 54.0145s, Throughput: 92.419561 im/s
Epoch 48, Iter 39, Loss: 9331.5966797, Throughput: 91.980647 im/s
Epoch 48, Train Loss: 9531.9873047, Time: 54.2725s, Throughput: 91.980334 im/s
Epoch 49, Iter 39, Loss: 10074.0957031, Throughput: 91.995295 im/s
Epoch 49, Train Loss: 9441.3681641, Time: 54.2639s, Throughput: 91.994905 im/s
Epoch 50, Iter 39, Loss: 10005.2695312, Throughput: 91.825578 im/s
Epoch 50, Train Loss: 9395.7919922, Time: 54.3641s, Throughput: 91.825241 im/s

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
