---
url: /previews/PR1754/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23981.0917969, Throughput: 20.284218 im/s
Epoch 1, Train Loss: 39857.3515625, Time: 246.3138s, Throughput: 20.266829 im/s
Epoch 2, Iter 39, Loss: 17920.7382812, Throughput: 204.034238 im/s
Epoch 2, Train Loss: 20236.0234375, Time: 24.4667s, Throughput: 204.032829 im/s
Epoch 3, Iter 39, Loss: 16027.8515625, Throughput: 204.125568 im/s
Epoch 3, Train Loss: 16619.3437500, Time: 24.4557s, Throughput: 204.124358 im/s
Epoch 4, Iter 39, Loss: 14652.9863281, Throughput: 204.559959 im/s
Epoch 4, Train Loss: 14967.9375000, Time: 24.4038s, Throughput: 204.558710 im/s
Epoch 5, Iter 39, Loss: 13452.7646484, Throughput: 203.867987 im/s
Epoch 5, Train Loss: 14087.5253906, Time: 24.4866s, Throughput: 203.866720 im/s
Epoch 6, Iter 39, Loss: 13660.7792969, Throughput: 203.832799 im/s
Epoch 6, Train Loss: 13404.9531250, Time: 24.4908s, Throughput: 203.831417 im/s
Epoch 7, Iter 39, Loss: 12108.8007812, Throughput: 203.806537 im/s
Epoch 7, Train Loss: 12877.7207031, Time: 24.4940s, Throughput: 203.805389 im/s
Epoch 8, Iter 39, Loss: 11809.8281250, Throughput: 204.490642 im/s
Epoch 8, Train Loss: 12482.8457031, Time: 24.4120s, Throughput: 204.489394 im/s
Epoch 9, Iter 39, Loss: 11946.6796875, Throughput: 203.974954 im/s
Epoch 9, Train Loss: 12214.1582031, Time: 24.4738s, Throughput: 203.973612 im/s
Epoch 10, Iter 39, Loss: 12305.3154297, Throughput: 204.193327 im/s
Epoch 10, Train Loss: 12068.4082031, Time: 24.4476s, Throughput: 204.192066 im/s
Epoch 11, Iter 39, Loss: 11301.2578125, Throughput: 204.796665 im/s
Epoch 11, Train Loss: 11642.4414062, Time: 24.3756s, Throughput: 204.795371 im/s
Epoch 12, Iter 39, Loss: 12013.2041016, Throughput: 204.313269 im/s
Epoch 12, Train Loss: 11536.0634766, Time: 24.4332s, Throughput: 204.311981 im/s
Epoch 13, Iter 39, Loss: 11614.6992188, Throughput: 203.938447 im/s
Epoch 13, Train Loss: 11340.1259766, Time: 24.4781s, Throughput: 203.937172 im/s
Epoch 14, Iter 39, Loss: 11082.6201172, Throughput: 204.019411 im/s
Epoch 14, Train Loss: 11247.2343750, Time: 24.4685s, Throughput: 204.017819 im/s
Epoch 15, Iter 39, Loss: 10225.0175781, Throughput: 204.004264 im/s
Epoch 15, Train Loss: 11096.5263672, Time: 24.4702s, Throughput: 204.003011 im/s
Epoch 16, Iter 39, Loss: 10173.1093750, Throughput: 203.517337 im/s
Epoch 16, Train Loss: 11022.4902344, Time: 24.5288s, Throughput: 203.515958 im/s
Epoch 17, Iter 39, Loss: 11191.8720703, Throughput: 199.631713 im/s
Epoch 17, Train Loss: 10936.0605469, Time: 25.0062s, Throughput: 199.630388 im/s
Epoch 18, Iter 39, Loss: 10888.2431641, Throughput: 203.492960 im/s
Epoch 18, Train Loss: 10759.4814453, Time: 24.5317s, Throughput: 203.491560 im/s
Epoch 19, Iter 39, Loss: 10649.3212891, Throughput: 203.884948 im/s
Epoch 19, Train Loss: 10713.4042969, Time: 24.4845s, Throughput: 203.883731 im/s
Epoch 20, Iter 39, Loss: 11576.0771484, Throughput: 203.546007 im/s
Epoch 20, Train Loss: 10562.3691406, Time: 24.5253s, Throughput: 203.544762 im/s
Epoch 21, Iter 39, Loss: 10962.7929688, Throughput: 203.901386 im/s
Epoch 21, Train Loss: 10504.5341797, Time: 24.4826s, Throughput: 203.900205 im/s
Epoch 22, Iter 39, Loss: 11094.3300781, Throughput: 203.757763 im/s
Epoch 22, Train Loss: 10437.4169922, Time: 24.4998s, Throughput: 203.756490 im/s
Epoch 23, Iter 39, Loss: 10523.9218750, Throughput: 203.803191 im/s
Epoch 23, Train Loss: 10431.6396484, Time: 24.4944s, Throughput: 203.801893 im/s
Epoch 24, Iter 39, Loss: 10669.7675781, Throughput: 203.827665 im/s
Epoch 24, Train Loss: 10290.5791016, Time: 24.4914s, Throughput: 203.826258 im/s
Epoch 25, Iter 39, Loss: 10115.6142578, Throughput: 203.766197 im/s
Epoch 25, Train Loss: 10256.2050781, Time: 24.4988s, Throughput: 203.764923 im/s
Epoch 26, Iter 39, Loss: 10035.9287109, Throughput: 203.516739 im/s
Epoch 26, Train Loss: 10296.2177734, Time: 24.5288s, Throughput: 203.515503 im/s
Epoch 27, Iter 39, Loss: 10469.4570312, Throughput: 203.915378 im/s
Epoch 27, Train Loss: 10243.6806641, Time: 24.4809s, Throughput: 203.914129 im/s
Epoch 28, Iter 39, Loss: 10808.6367188, Throughput: 203.652129 im/s
Epoch 28, Train Loss: 10050.7167969, Time: 24.5125s, Throughput: 203.650784 im/s
Epoch 29, Iter 39, Loss: 10388.3710938, Throughput: 203.945096 im/s
Epoch 29, Train Loss: 10041.8310547, Time: 24.4773s, Throughput: 203.943854 im/s
Epoch 30, Iter 39, Loss: 10477.7812500, Throughput: 203.485886 im/s
Epoch 30, Train Loss: 9927.0957031, Time: 24.5326s, Throughput: 203.484526 im/s
Epoch 31, Iter 39, Loss: 10173.3476562, Throughput: 202.844752 im/s
Epoch 31, Train Loss: 9889.8505859, Time: 24.6101s, Throughput: 202.843434 im/s
Epoch 32, Iter 39, Loss: 10421.8164062, Throughput: 203.170635 im/s
Epoch 32, Train Loss: 9907.0224609, Time: 24.5706s, Throughput: 203.169261 im/s
Epoch 33, Iter 39, Loss: 10216.6787109, Throughput: 203.452655 im/s
Epoch 33, Train Loss: 9873.6025391, Time: 24.5366s, Throughput: 203.451302 im/s
Epoch 34, Iter 39, Loss: 9795.2324219, Throughput: 203.196393 im/s
Epoch 34, Train Loss: 9816.1689453, Time: 24.5675s, Throughput: 203.195169 im/s
Epoch 35, Iter 39, Loss: 9957.8095703, Throughput: 203.331361 im/s
Epoch 35, Train Loss: 9887.8554688, Time: 24.5512s, Throughput: 203.330101 im/s
Epoch 36, Iter 39, Loss: 10227.5800781, Throughput: 203.926776 im/s
Epoch 36, Train Loss: 9778.5341797, Time: 24.4795s, Throughput: 203.925501 im/s
Epoch 37, Iter 39, Loss: 9560.4111328, Throughput: 201.504429 im/s
Epoch 37, Train Loss: 9619.8916016, Time: 24.7738s, Throughput: 201.503118 im/s
Epoch 38, Iter 39, Loss: 10429.3144531, Throughput: 203.765970 im/s
Epoch 38, Train Loss: 9674.9550781, Time: 24.4988s, Throughput: 203.764691 im/s
Epoch 39, Iter 39, Loss: 9773.2246094, Throughput: 203.658226 im/s
Epoch 39, Train Loss: 9625.9755859, Time: 24.5118s, Throughput: 203.656830 im/s
Epoch 40, Iter 39, Loss: 9696.4882812, Throughput: 204.383828 im/s
Epoch 40, Train Loss: 9594.6054688, Time: 24.4248s, Throughput: 204.382497 im/s
Epoch 41, Iter 39, Loss: 10262.5742188, Throughput: 204.067567 im/s
Epoch 41, Train Loss: 9511.0722656, Time: 24.4626s, Throughput: 204.066274 im/s
Epoch 42, Iter 39, Loss: 9967.8505859, Throughput: 203.424989 im/s
Epoch 42, Train Loss: 9594.1572266, Time: 24.5399s, Throughput: 203.423762 im/s
Epoch 43, Iter 39, Loss: 9271.8593750, Throughput: 204.279850 im/s
Epoch 43, Train Loss: 9573.2324219, Time: 24.4372s, Throughput: 204.278612 im/s
Epoch 44, Iter 39, Loss: 9704.5488281, Throughput: 203.950694 im/s
Epoch 44, Train Loss: 9481.1992188, Time: 24.4767s, Throughput: 203.949270 im/s
Epoch 45, Iter 39, Loss: 10068.9404297, Throughput: 204.262925 im/s
Epoch 45, Train Loss: 9440.1503906, Time: 24.4392s, Throughput: 204.261655 im/s
Epoch 46, Iter 39, Loss: 8839.6767578, Throughput: 203.317058 im/s
Epoch 46, Train Loss: 9415.4033203, Time: 24.5529s, Throughput: 203.315775 im/s
Epoch 47, Iter 39, Loss: 10424.5761719, Throughput: 204.390799 im/s
Epoch 47, Train Loss: 9416.3320312, Time: 24.4240s, Throughput: 204.389275 im/s
Epoch 48, Iter 39, Loss: 9287.3242188, Throughput: 203.934824 im/s
Epoch 48, Train Loss: 9373.4931641, Time: 24.4786s, Throughput: 203.933533 im/s
Epoch 49, Iter 39, Loss: 9421.8339844, Throughput: 203.703812 im/s
Epoch 49, Train Loss: 9346.5683594, Time: 24.5063s, Throughput: 203.702464 im/s
Epoch 50, Iter 39, Loss: 9153.0595703, Throughput: 204.224602 im/s
Epoch 50, Train Loss: 9319.4667969, Time: 24.4438s, Throughput: 204.223255 im/s

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
  CPU: 4 × AMD EPYC 9V74 80-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver4)
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
