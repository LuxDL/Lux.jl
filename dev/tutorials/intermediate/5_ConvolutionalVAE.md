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
Epoch 1, Iter 39, Loss: 23347.9882812, Throughput: 19.358998 im/s
Epoch 1, Train Loss: 39526.6054688, Time: 258.0270s, Throughput: 19.346814 im/s
Epoch 2, Iter 39, Loss: 17265.3652344, Throughput: 85.385009 im/s
Epoch 2, Train Loss: 19865.1289062, Time: 58.4648s, Throughput: 85.384739 im/s
Epoch 3, Iter 39, Loss: 15449.1660156, Throughput: 85.329464 im/s
Epoch 3, Train Loss: 16352.3876953, Time: 58.5029s, Throughput: 85.329156 im/s
Epoch 4, Iter 39, Loss: 14843.9267578, Throughput: 87.020277 im/s
Epoch 4, Train Loss: 14934.9755859, Time: 57.3661s, Throughput: 87.019998 im/s
Epoch 5, Iter 39, Loss: 13542.4980469, Throughput: 86.190246 im/s
Epoch 5, Train Loss: 13948.0556641, Time: 57.9186s, Throughput: 86.189957 im/s
Epoch 6, Iter 39, Loss: 12989.8505859, Throughput: 86.764124 im/s
Epoch 6, Train Loss: 13271.0644531, Time: 57.5355s, Throughput: 86.763796 im/s
Epoch 7, Iter 39, Loss: 11935.0566406, Throughput: 85.233705 im/s
Epoch 7, Train Loss: 12911.4384766, Time: 58.5686s, Throughput: 85.233418 im/s
Epoch 8, Iter 39, Loss: 13335.5917969, Throughput: 84.357998 im/s
Epoch 8, Train Loss: 12416.5390625, Time: 59.1765s, Throughput: 84.357748 im/s
Epoch 9, Iter 39, Loss: 11876.3750000, Throughput: 84.374825 im/s
Epoch 9, Train Loss: 12196.8408203, Time: 59.1648s, Throughput: 84.374541 im/s
Epoch 10, Iter 39, Loss: 11997.3886719, Throughput: 84.650552 im/s
Epoch 10, Train Loss: 11893.8105469, Time: 58.9721s, Throughput: 84.650265 im/s
Epoch 11, Iter 39, Loss: 11049.6259766, Throughput: 85.712343 im/s
Epoch 11, Train Loss: 11734.7851562, Time: 58.2415s, Throughput: 85.712069 im/s
Epoch 12, Iter 39, Loss: 10998.7636719, Throughput: 87.058147 im/s
Epoch 12, Train Loss: 11587.4218750, Time: 57.3412s, Throughput: 87.057853 im/s
Epoch 13, Iter 39, Loss: 11876.9619141, Throughput: 89.430932 im/s
Epoch 13, Train Loss: 11380.1679688, Time: 55.8198s, Throughput: 89.430621 im/s
Epoch 14, Iter 39, Loss: 10757.9218750, Throughput: 87.207916 im/s
Epoch 14, Train Loss: 11229.6582031, Time: 57.2427s, Throughput: 87.207637 im/s
Epoch 15, Iter 39, Loss: 11482.0371094, Throughput: 87.399547 im/s
Epoch 15, Train Loss: 11068.8056641, Time: 57.1172s, Throughput: 87.399242 im/s
Epoch 16, Iter 39, Loss: 10531.9960938, Throughput: 86.135966 im/s
Epoch 16, Train Loss: 10941.6142578, Time: 57.9551s, Throughput: 86.135707 im/s
Epoch 17, Iter 39, Loss: 11065.9091797, Throughput: 85.612791 im/s
Epoch 17, Train Loss: 10819.7509766, Time: 58.3092s, Throughput: 85.612534 im/s
Epoch 18, Iter 39, Loss: 10874.4531250, Throughput: 84.767901 im/s
Epoch 18, Train Loss: 10780.4023438, Time: 58.8904s, Throughput: 84.767568 im/s
Epoch 19, Iter 39, Loss: 10350.5478516, Throughput: 85.572200 im/s
Epoch 19, Train Loss: 10638.9130859, Time: 58.3369s, Throughput: 85.571930 im/s
Epoch 20, Iter 39, Loss: 10649.6884766, Throughput: 87.651380 im/s
Epoch 20, Train Loss: 10559.9775391, Time: 56.9531s, Throughput: 87.651098 im/s
Epoch 21, Iter 39, Loss: 10832.2470703, Throughput: 86.506736 im/s
Epoch 21, Train Loss: 10411.2304688, Time: 57.7067s, Throughput: 86.506435 im/s
Epoch 22, Iter 39, Loss: 10001.6308594, Throughput: 84.480839 im/s
Epoch 22, Train Loss: 10343.7451172, Time: 59.0905s, Throughput: 84.480579 im/s
Epoch 23, Iter 39, Loss: 10990.5410156, Throughput: 85.651105 im/s
Epoch 23, Train Loss: 10375.0810547, Time: 58.2831s, Throughput: 85.650853 im/s
Epoch 24, Iter 39, Loss: 10886.8837891, Throughput: 84.861013 im/s
Epoch 24, Train Loss: 10289.5664062, Time: 58.8258s, Throughput: 84.860748 im/s
Epoch 25, Iter 39, Loss: 9690.1132812, Throughput: 83.330441 im/s
Epoch 25, Train Loss: 10149.9169922, Time: 59.9063s, Throughput: 83.330200 im/s
Epoch 26, Iter 39, Loss: 10263.4218750, Throughput: 83.900463 im/s
Epoch 26, Train Loss: 10114.6748047, Time: 59.4992s, Throughput: 83.900230 im/s
Epoch 27, Iter 39, Loss: 10399.1308594, Throughput: 85.139618 im/s
Epoch 27, Train Loss: 10077.9013672, Time: 58.6333s, Throughput: 85.139327 im/s
Epoch 28, Iter 39, Loss: 10514.5917969, Throughput: 84.440801 im/s
Epoch 28, Train Loss: 9955.6699219, Time: 59.1185s, Throughput: 84.440537 im/s
Epoch 29, Iter 39, Loss: 10138.7968750, Throughput: 83.616671 im/s
Epoch 29, Train Loss: 9975.9794922, Time: 59.7012s, Throughput: 83.616438 im/s
Epoch 30, Iter 39, Loss: 10170.3710938, Throughput: 84.861799 im/s
Epoch 30, Train Loss: 9979.2705078, Time: 58.8252s, Throughput: 84.861523 im/s
Epoch 31, Iter 39, Loss: 10608.8203125, Throughput: 84.812388 im/s
Epoch 31, Train Loss: 9923.3886719, Time: 58.8595s, Throughput: 84.812124 im/s
Epoch 32, Iter 39, Loss: 9569.0634766, Throughput: 82.524208 im/s
Epoch 32, Train Loss: 9872.1572266, Time: 60.4915s, Throughput: 82.523976 im/s
Epoch 33, Iter 39, Loss: 9796.2617188, Throughput: 84.340843 im/s
Epoch 33, Train Loss: 9756.5400391, Time: 59.1886s, Throughput: 84.340567 im/s
Epoch 34, Iter 39, Loss: 9187.7753906, Throughput: 81.326019 im/s
Epoch 34, Train Loss: 9693.2792969, Time: 61.3828s, Throughput: 81.325778 im/s
Epoch 35, Iter 39, Loss: 9430.6816406, Throughput: 83.763723 im/s
Epoch 35, Train Loss: 9675.7382812, Time: 59.5964s, Throughput: 83.763478 im/s
Epoch 36, Iter 39, Loss: 9511.7402344, Throughput: 83.062290 im/s
Epoch 36, Train Loss: 9673.3593750, Time: 60.0996s, Throughput: 83.062053 im/s
Epoch 37, Iter 39, Loss: 9689.2646484, Throughput: 85.246621 im/s
Epoch 37, Train Loss: 9640.6191406, Time: 58.5597s, Throughput: 85.246383 im/s
Epoch 38, Iter 39, Loss: 9350.9160156, Throughput: 84.158054 im/s
Epoch 38, Train Loss: 9616.3769531, Time: 59.3171s, Throughput: 84.157814 im/s
Epoch 39, Iter 39, Loss: 10132.4052734, Throughput: 84.806530 im/s
Epoch 39, Train Loss: 9634.6816406, Time: 58.8636s, Throughput: 84.806282 im/s
Epoch 40, Iter 39, Loss: 9303.7109375, Throughput: 81.082751 im/s
Epoch 40, Train Loss: 9560.8515625, Time: 61.5669s, Throughput: 81.082512 im/s
Epoch 41, Iter 39, Loss: 9124.9062500, Throughput: 80.644300 im/s
Epoch 41, Train Loss: 9544.6298828, Time: 61.9016s, Throughput: 80.644080 im/s
Epoch 42, Iter 39, Loss: 9673.5214844, Throughput: 81.643818 im/s
Epoch 42, Train Loss: 9464.6406250, Time: 61.1440s, Throughput: 81.643316 im/s
Epoch 43, Iter 39, Loss: 9680.9589844, Throughput: 82.080034 im/s
Epoch 43, Train Loss: 9436.1992188, Time: 60.8189s, Throughput: 82.079810 im/s
Epoch 44, Iter 39, Loss: 9551.5810547, Throughput: 82.591967 im/s
Epoch 44, Train Loss: 9520.9306641, Time: 60.4419s, Throughput: 82.591735 im/s
Epoch 45, Iter 39, Loss: 9125.5312500, Throughput: 81.474696 im/s
Epoch 45, Train Loss: 9410.8916016, Time: 61.2707s, Throughput: 81.474445 im/s
Epoch 46, Iter 39, Loss: 9251.0820312, Throughput: 81.544204 im/s
Epoch 46, Train Loss: 9378.6162109, Time: 61.2185s, Throughput: 81.543962 im/s
Epoch 47, Iter 39, Loss: 9431.2207031, Throughput: 81.033108 im/s
Epoch 47, Train Loss: 9377.7705078, Time: 61.6046s, Throughput: 81.032873 im/s
Epoch 48, Iter 39, Loss: 9095.7119141, Throughput: 80.922455 im/s
Epoch 48, Train Loss: 9342.2529297, Time: 61.6889s, Throughput: 80.922210 im/s
Epoch 49, Iter 39, Loss: 9645.0761719, Throughput: 80.553835 im/s
Epoch 49, Train Loss: 9306.7089844, Time: 61.9712s, Throughput: 80.553568 im/s
Epoch 50, Iter 39, Loss: 9472.5800781, Throughput: 80.924273 im/s
Epoch 50, Train Loss: 9264.4843750, Time: 61.6875s, Throughput: 80.924048 im/s

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
