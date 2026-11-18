---
url: /previews/PR1831/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23601.4335938, Throughput: 16.600130 im/s
Epoch 1, Train Loss: 39739.5898438, Time: 300.9795s, Throughput: 16.585846 im/s
Epoch 2, Iter 39, Loss: 17372.5937500, Throughput: 174.833091 im/s
Epoch 2, Train Loss: 20080.6445312, Time: 28.5531s, Throughput: 174.832045 im/s
Epoch 3, Iter 39, Loss: 14959.7099609, Throughput: 174.397005 im/s
Epoch 3, Train Loss: 16579.5742188, Time: 28.6245s, Throughput: 174.395859 im/s
Epoch 4, Iter 39, Loss: 13965.3750000, Throughput: 173.777038 im/s
Epoch 4, Train Loss: 15131.2431641, Time: 28.7266s, Throughput: 173.775966 im/s
Epoch 5, Iter 39, Loss: 14395.3027344, Throughput: 174.409403 im/s
Epoch 5, Train Loss: 14269.1621094, Time: 28.6225s, Throughput: 174.408471 im/s
Epoch 6, Iter 39, Loss: 13516.0673828, Throughput: 174.106980 im/s
Epoch 6, Train Loss: 13578.9521484, Time: 28.6722s, Throughput: 174.106062 im/s
Epoch 7, Iter 39, Loss: 13295.0009766, Throughput: 175.021169 im/s
Epoch 7, Train Loss: 13221.5234375, Time: 28.5224s, Throughput: 175.020255 im/s
Epoch 8, Iter 39, Loss: 12969.7158203, Throughput: 174.481579 im/s
Epoch 8, Train Loss: 12730.7763672, Time: 28.6106s, Throughput: 174.480678 im/s
Epoch 9, Iter 39, Loss: 12505.2626953, Throughput: 174.988543 im/s
Epoch 9, Train Loss: 12352.6923828, Time: 28.5277s, Throughput: 174.987641 im/s
Epoch 10, Iter 39, Loss: 12010.2236328, Throughput: 174.381620 im/s
Epoch 10, Train Loss: 12096.4638672, Time: 28.6270s, Throughput: 174.380756 im/s
Epoch 11, Iter 39, Loss: 12022.9423828, Throughput: 174.650635 im/s
Epoch 11, Train Loss: 11918.8681641, Time: 28.5829s, Throughput: 174.649713 im/s
Epoch 12, Iter 39, Loss: 11769.9140625, Throughput: 174.303685 im/s
Epoch 12, Train Loss: 11764.3457031, Time: 28.6398s, Throughput: 174.302765 im/s
Epoch 13, Iter 39, Loss: 11283.1074219, Throughput: 174.145182 im/s
Epoch 13, Train Loss: 11561.5839844, Time: 28.6659s, Throughput: 174.144173 im/s
Epoch 14, Iter 39, Loss: 12358.1562500, Throughput: 172.313934 im/s
Epoch 14, Train Loss: 11432.4580078, Time: 28.9706s, Throughput: 172.312912 im/s
Epoch 15, Iter 39, Loss: 10796.7978516, Throughput: 174.857642 im/s
Epoch 15, Train Loss: 11326.6728516, Time: 28.5491s, Throughput: 174.856607 im/s
Epoch 16, Iter 39, Loss: 11585.8261719, Throughput: 174.821018 im/s
Epoch 16, Train Loss: 11197.2158203, Time: 28.5551s, Throughput: 174.820051 im/s
Epoch 17, Iter 39, Loss: 11363.2187500, Throughput: 174.448776 im/s
Epoch 17, Train Loss: 10953.4169922, Time: 28.6160s, Throughput: 174.447856 im/s
Epoch 18, Iter 39, Loss: 10610.7929688, Throughput: 174.407721 im/s
Epoch 18, Train Loss: 10933.5224609, Time: 28.6228s, Throughput: 174.406613 im/s
Epoch 19, Iter 39, Loss: 10381.5712891, Throughput: 174.481892 im/s
Epoch 19, Train Loss: 10858.7919922, Time: 28.6106s, Throughput: 174.480886 im/s
Epoch 20, Iter 39, Loss: 10707.3554688, Throughput: 174.016560 im/s
Epoch 20, Train Loss: 10757.4130859, Time: 28.6871s, Throughput: 174.015584 im/s
Epoch 21, Iter 39, Loss: 10036.1630859, Throughput: 175.018543 im/s
Epoch 21, Train Loss: 10726.6015625, Time: 28.5229s, Throughput: 175.017556 im/s
Epoch 22, Iter 39, Loss: 10287.6582031, Throughput: 174.182486 im/s
Epoch 22, Train Loss: 10632.4453125, Time: 28.6598s, Throughput: 174.181537 im/s
Epoch 23, Iter 39, Loss: 11382.2226562, Throughput: 174.358093 im/s
Epoch 23, Train Loss: 10534.6572266, Time: 28.6309s, Throughput: 174.357065 im/s
Epoch 24, Iter 39, Loss: 10478.1826172, Throughput: 174.569857 im/s
Epoch 24, Train Loss: 10427.0742188, Time: 28.5962s, Throughput: 174.568850 im/s
Epoch 25, Iter 39, Loss: 10148.9140625, Throughput: 174.759119 im/s
Epoch 25, Train Loss: 10355.8417969, Time: 28.5652s, Throughput: 174.758098 im/s
Epoch 26, Iter 39, Loss: 10816.1718750, Throughput: 174.447806 im/s
Epoch 26, Train Loss: 10256.5996094, Time: 28.6162s, Throughput: 174.446673 im/s
Epoch 27, Iter 39, Loss: 10121.0507812, Throughput: 174.283688 im/s
Epoch 27, Train Loss: 10223.0937500, Time: 28.6431s, Throughput: 174.282708 im/s
Epoch 28, Iter 39, Loss: 10534.7578125, Throughput: 174.693969 im/s
Epoch 28, Train Loss: 10196.9453125, Time: 28.5759s, Throughput: 174.692911 im/s
Epoch 29, Iter 39, Loss: 10427.6074219, Throughput: 174.108109 im/s
Epoch 29, Train Loss: 10198.4804688, Time: 28.6720s, Throughput: 174.107077 im/s
Epoch 30, Iter 39, Loss: 10365.9658203, Throughput: 174.624604 im/s
Epoch 30, Train Loss: 10124.7724609, Time: 28.5872s, Throughput: 174.623725 im/s
Epoch 31, Iter 39, Loss: 9388.7333984, Throughput: 173.992856 im/s
Epoch 31, Train Loss: 10046.6669922, Time: 28.6910s, Throughput: 173.992026 im/s
Epoch 32, Iter 39, Loss: 10320.7802734, Throughput: 174.461535 im/s
Epoch 32, Train Loss: 10066.4755859, Time: 28.6139s, Throughput: 174.460615 im/s
Epoch 33, Iter 39, Loss: 10170.9628906, Throughput: 174.857380 im/s
Epoch 33, Train Loss: 10032.3056641, Time: 28.5491s, Throughput: 174.856448 im/s
Epoch 34, Iter 39, Loss: 9715.7070312, Throughput: 174.462261 im/s
Epoch 34, Train Loss: 9980.7343750, Time: 28.6138s, Throughput: 174.461359 im/s
Epoch 35, Iter 39, Loss: 10870.8271484, Throughput: 171.870092 im/s
Epoch 35, Train Loss: 9968.6494141, Time: 29.0454s, Throughput: 171.869133 im/s
Epoch 36, Iter 39, Loss: 9565.6923828, Throughput: 174.323540 im/s
Epoch 36, Train Loss: 9918.2548828, Time: 28.6366s, Throughput: 174.322559 im/s
Epoch 37, Iter 39, Loss: 9884.0332031, Throughput: 174.007662 im/s
Epoch 37, Train Loss: 9876.9511719, Time: 28.6886s, Throughput: 174.006684 im/s
Epoch 38, Iter 39, Loss: 10097.7880859, Throughput: 174.548554 im/s
Epoch 38, Train Loss: 9876.4589844, Time: 28.5996s, Throughput: 174.547664 im/s
Epoch 39, Iter 39, Loss: 10616.3164062, Throughput: 174.069952 im/s
Epoch 39, Train Loss: 9754.7324219, Time: 28.6783s, Throughput: 174.069035 im/s
Epoch 40, Iter 39, Loss: 9128.3574219, Throughput: 174.599635 im/s
Epoch 40, Train Loss: 9781.4902344, Time: 28.5913s, Throughput: 174.598615 im/s
Epoch 41, Iter 39, Loss: 9162.2207031, Throughput: 173.939101 im/s
Epoch 41, Train Loss: 9637.9238281, Time: 28.6999s, Throughput: 173.938155 im/s
Epoch 42, Iter 39, Loss: 9385.7958984, Throughput: 174.309119 im/s
Epoch 42, Train Loss: 9612.8759766, Time: 28.6389s, Throughput: 174.308079 im/s
Epoch 43, Iter 39, Loss: 9462.0234375, Throughput: 173.827971 im/s
Epoch 43, Train Loss: 9608.8203125, Time: 28.7182s, Throughput: 173.827020 im/s
Epoch 44, Iter 39, Loss: 9918.1103516, Throughput: 174.145875 im/s
Epoch 44, Train Loss: 9603.8476562, Time: 28.6658s, Throughput: 174.144770 im/s
Epoch 45, Iter 39, Loss: 9255.2705078, Throughput: 174.046834 im/s
Epoch 45, Train Loss: 9555.9843750, Time: 28.6821s, Throughput: 174.045924 im/s
Epoch 46, Iter 39, Loss: 9596.5009766, Throughput: 174.666946 im/s
Epoch 46, Train Loss: 9614.5927734, Time: 28.5803s, Throughput: 174.665949 im/s
Epoch 47, Iter 39, Loss: 9325.3564453, Throughput: 174.448745 im/s
Epoch 47, Train Loss: 9490.6757812, Time: 28.6160s, Throughput: 174.447837 im/s
Epoch 48, Iter 39, Loss: 9266.0283203, Throughput: 173.870841 im/s
Epoch 48, Train Loss: 9490.3027344, Time: 28.7112s, Throughput: 173.869594 im/s
Epoch 49, Iter 39, Loss: 9365.4570312, Throughput: 174.305918 im/s
Epoch 49, Train Loss: 9405.1494141, Time: 28.6395s, Throughput: 174.304914 im/s
Epoch 50, Iter 39, Loss: 9246.7929688, Throughput: 173.810497 im/s
Epoch 50, Train Loss: 9379.5380859, Time: 28.7211s, Throughput: 173.809559 im/s

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
