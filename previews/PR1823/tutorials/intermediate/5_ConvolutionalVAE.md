---
url: /previews/PR1823/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23358.9628906, Throughput: 20.580526 im/s
Epoch 1, Train Loss: 39691.0117188, Time: 242.7524s, Throughput: 20.564162 im/s
Epoch 2, Iter 39, Loss: 17965.5527344, Throughput: 122.059594 im/s
Epoch 2, Train Loss: 20140.2441406, Time: 40.8982s, Throughput: 122.059108 im/s
Epoch 3, Iter 39, Loss: 15730.1015625, Throughput: 122.303675 im/s
Epoch 3, Train Loss: 16482.8535156, Time: 40.8166s, Throughput: 122.303160 im/s
Epoch 4, Iter 39, Loss: 14052.9707031, Throughput: 121.697757 im/s
Epoch 4, Train Loss: 14993.9101562, Time: 41.0198s, Throughput: 121.697232 im/s
Epoch 5, Iter 39, Loss: 13754.2617188, Throughput: 122.273383 im/s
Epoch 5, Train Loss: 14086.1699219, Time: 40.8267s, Throughput: 122.272826 im/s
Epoch 6, Iter 39, Loss: 14272.3154297, Throughput: 123.019395 im/s
Epoch 6, Train Loss: 13409.8173828, Time: 40.5791s, Throughput: 123.018880 im/s
Epoch 7, Iter 39, Loss: 11878.1396484, Throughput: 122.915054 im/s
Epoch 7, Train Loss: 12872.8955078, Time: 40.6136s, Throughput: 122.914482 im/s
Epoch 8, Iter 39, Loss: 12461.9218750, Throughput: 122.669423 im/s
Epoch 8, Train Loss: 12483.5253906, Time: 40.6949s, Throughput: 122.668964 im/s
Epoch 9, Iter 39, Loss: 12077.6523438, Throughput: 121.553397 im/s
Epoch 9, Train Loss: 12157.5800781, Time: 41.0686s, Throughput: 121.552865 im/s
Epoch 10, Iter 39, Loss: 12431.6328125, Throughput: 121.320244 im/s
Epoch 10, Train Loss: 11880.3701172, Time: 41.1475s, Throughput: 121.319660 im/s
Epoch 11, Iter 39, Loss: 12165.6855469, Throughput: 121.497398 im/s
Epoch 11, Train Loss: 11734.5976562, Time: 41.0875s, Throughput: 121.496915 im/s
Epoch 12, Iter 39, Loss: 11138.8671875, Throughput: 121.660348 im/s
Epoch 12, Train Loss: 11572.3417969, Time: 41.0325s, Throughput: 121.659627 im/s
Epoch 13, Iter 39, Loss: 11746.3730469, Throughput: 121.159826 im/s
Epoch 13, Train Loss: 11351.6943359, Time: 41.2020s, Throughput: 121.159299 im/s
Epoch 14, Iter 39, Loss: 11381.5390625, Throughput: 121.990569 im/s
Epoch 14, Train Loss: 11219.3681641, Time: 40.9214s, Throughput: 121.989946 im/s
Epoch 15, Iter 39, Loss: 10803.4511719, Throughput: 121.457404 im/s
Epoch 15, Train Loss: 11159.5146484, Time: 41.1010s, Throughput: 121.456881 im/s
Epoch 16, Iter 39, Loss: 10590.0566406, Throughput: 122.278609 im/s
Epoch 16, Train Loss: 10973.2675781, Time: 40.8250s, Throughput: 122.278040 im/s
Epoch 17, Iter 39, Loss: 11696.1093750, Throughput: 121.501176 im/s
Epoch 17, Train Loss: 10937.8818359, Time: 41.0862s, Throughput: 121.500552 im/s
Epoch 18, Iter 39, Loss: 11386.7929688, Throughput: 122.067751 im/s
Epoch 18, Train Loss: 10830.6230469, Time: 40.8955s, Throughput: 122.067181 im/s
Epoch 19, Iter 39, Loss: 10731.9707031, Throughput: 121.795179 im/s
Epoch 19, Train Loss: 10729.4970703, Time: 40.9870s, Throughput: 121.794659 im/s
Epoch 20, Iter 39, Loss: 10269.1503906, Throughput: 122.502926 im/s
Epoch 20, Train Loss: 10576.1718750, Time: 40.7502s, Throughput: 122.502319 im/s
Epoch 21, Iter 39, Loss: 11074.8876953, Throughput: 122.778815 im/s
Epoch 21, Train Loss: 10533.0576172, Time: 40.6587s, Throughput: 122.778283 im/s
Epoch 22, Iter 39, Loss: 11048.8056641, Throughput: 122.599712 im/s
Epoch 22, Train Loss: 10453.6982422, Time: 40.7180s, Throughput: 122.599287 im/s
Epoch 23, Iter 39, Loss: 10663.7421875, Throughput: 122.586613 im/s
Epoch 23, Train Loss: 10360.6054688, Time: 40.7224s, Throughput: 122.586026 im/s
Epoch 24, Iter 39, Loss: 9392.1425781, Throughput: 122.103892 im/s
Epoch 24, Train Loss: 10317.7294922, Time: 40.8834s, Throughput: 122.103214 im/s
Epoch 25, Iter 39, Loss: 10450.4570312, Throughput: 122.995336 im/s
Epoch 25, Train Loss: 10290.4892578, Time: 40.5871s, Throughput: 122.994781 im/s
Epoch 26, Iter 39, Loss: 9711.2050781, Throughput: 122.176937 im/s
Epoch 26, Train Loss: 10115.0703125, Time: 40.8590s, Throughput: 122.176315 im/s
Epoch 27, Iter 39, Loss: 10974.1757812, Throughput: 122.162814 im/s
Epoch 27, Train Loss: 10137.4248047, Time: 40.8637s, Throughput: 122.162209 im/s
Epoch 28, Iter 39, Loss: 10005.4609375, Throughput: 122.846078 im/s
Epoch 28, Train Loss: 10073.1289062, Time: 40.6364s, Throughput: 122.845573 im/s
Epoch 29, Iter 39, Loss: 9381.3789062, Throughput: 123.052983 im/s
Epoch 29, Train Loss: 10021.4345703, Time: 40.5681s, Throughput: 123.052494 im/s
Epoch 30, Iter 39, Loss: 9353.3574219, Throughput: 123.144519 im/s
Epoch 30, Train Loss: 9991.8593750, Time: 40.5379s, Throughput: 123.144048 im/s
Epoch 31, Iter 39, Loss: 9647.3457031, Throughput: 123.281556 im/s
Epoch 31, Train Loss: 9928.8750000, Time: 40.4928s, Throughput: 123.281057 im/s
Epoch 32, Iter 39, Loss: 10418.1386719, Throughput: 122.596454 im/s
Epoch 32, Train Loss: 9815.9853516, Time: 40.7191s, Throughput: 122.595988 im/s
Epoch 33, Iter 39, Loss: 9811.6093750, Throughput: 122.320268 im/s
Epoch 33, Train Loss: 9795.9931641, Time: 40.8111s, Throughput: 122.319744 im/s
Epoch 34, Iter 39, Loss: 10017.6650391, Throughput: 122.108688 im/s
Epoch 34, Train Loss: 9766.3789062, Time: 40.8818s, Throughput: 122.108087 im/s
Epoch 35, Iter 39, Loss: 10143.6171875, Throughput: 122.507661 im/s
Epoch 35, Train Loss: 9739.3046875, Time: 40.7491s, Throughput: 122.505909 im/s
Epoch 36, Iter 39, Loss: 10043.5976562, Throughput: 121.844961 im/s
Epoch 36, Train Loss: 9753.5996094, Time: 40.9703s, Throughput: 121.844434 im/s
Epoch 37, Iter 39, Loss: 10219.9287109, Throughput: 122.987326 im/s
Epoch 37, Train Loss: 9718.8662109, Time: 40.5897s, Throughput: 122.986784 im/s
Epoch 38, Iter 39, Loss: 9684.8515625, Throughput: 122.698165 im/s
Epoch 38, Train Loss: 9566.6582031, Time: 40.6854s, Throughput: 122.697604 im/s
Epoch 39, Iter 39, Loss: 9888.2128906, Throughput: 122.281005 im/s
Epoch 39, Train Loss: 9608.0644531, Time: 40.8242s, Throughput: 122.280391 im/s
Epoch 40, Iter 39, Loss: 9474.7832031, Throughput: 122.979237 im/s
Epoch 40, Train Loss: 9550.8330078, Time: 40.5924s, Throughput: 122.978637 im/s
Epoch 41, Iter 39, Loss: 9604.3632812, Throughput: 122.394129 im/s
Epoch 41, Train Loss: 9534.7705078, Time: 40.7864s, Throughput: 122.393622 im/s
Epoch 42, Iter 39, Loss: 9661.0908203, Throughput: 122.428841 im/s
Epoch 42, Train Loss: 9473.0107422, Time: 40.7749s, Throughput: 122.428315 im/s
Epoch 43, Iter 39, Loss: 9461.9882812, Throughput: 123.252083 im/s
Epoch 43, Train Loss: 9495.7451172, Time: 40.5025s, Throughput: 123.251502 im/s
Epoch 44, Iter 39, Loss: 9380.8125000, Throughput: 123.273516 im/s
Epoch 44, Train Loss: 9434.0273438, Time: 40.4955s, Throughput: 123.273054 im/s
Epoch 45, Iter 39, Loss: 9187.1455078, Throughput: 122.871362 im/s
Epoch 45, Train Loss: 9474.4843750, Time: 40.6280s, Throughput: 122.870781 im/s
Epoch 46, Iter 39, Loss: 9962.5976562, Throughput: 123.277023 im/s
Epoch 46, Train Loss: 9430.4335938, Time: 40.4943s, Throughput: 123.276531 im/s
Epoch 47, Iter 39, Loss: 9501.3359375, Throughput: 122.811380 im/s
Epoch 47, Train Loss: 9293.8759766, Time: 40.6479s, Throughput: 122.810872 im/s
Epoch 48, Iter 39, Loss: 10165.6904297, Throughput: 123.145467 im/s
Epoch 48, Train Loss: 9341.4179688, Time: 40.5377s, Throughput: 123.144628 im/s
Epoch 49, Iter 39, Loss: 9084.9013672, Throughput: 123.161025 im/s
Epoch 49, Train Loss: 9371.6611328, Time: 40.5325s, Throughput: 123.160475 im/s
Epoch 50, Iter 39, Loss: 9786.5214844, Throughput: 122.360649 im/s
Epoch 50, Train Loss: 9298.8330078, Time: 40.7976s, Throughput: 122.360169 im/s

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
  CPU: 4 × Intel(R) Xeon(R) 6975P-C
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, graniterapids)
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
