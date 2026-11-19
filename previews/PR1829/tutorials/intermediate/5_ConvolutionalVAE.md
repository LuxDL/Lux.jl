---
url: /previews/PR1829/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24076.0585938, Throughput: 15.902953 im/s
Epoch 1, Train Loss: 39548.4843750, Time: 314.1723s, Throughput: 15.889372 im/s
Epoch 2, Iter 39, Loss: 16691.0703125, Throughput: 172.047707 im/s
Epoch 2, Train Loss: 20114.2343750, Time: 29.0154s, Throughput: 172.046645 im/s
Epoch 3, Iter 39, Loss: 15854.7695312, Throughput: 172.796703 im/s
Epoch 3, Train Loss: 16479.8320312, Time: 28.8896s, Throughput: 172.795758 im/s
Epoch 4, Iter 39, Loss: 14080.3642578, Throughput: 172.370644 im/s
Epoch 4, Train Loss: 15038.9394531, Time: 28.9610s, Throughput: 172.369632 im/s
Epoch 5, Iter 39, Loss: 13828.3183594, Throughput: 173.104252 im/s
Epoch 5, Train Loss: 13989.1044922, Time: 28.8383s, Throughput: 173.103311 im/s
Epoch 6, Iter 39, Loss: 12925.4863281, Throughput: 172.645419 im/s
Epoch 6, Train Loss: 13219.6923828, Time: 28.9150s, Throughput: 172.644248 im/s
Epoch 7, Iter 39, Loss: 13405.1542969, Throughput: 173.665232 im/s
Epoch 7, Train Loss: 12719.1611328, Time: 28.7451s, Throughput: 173.664254 im/s
Epoch 8, Iter 39, Loss: 11982.5458984, Throughput: 173.667759 im/s
Epoch 8, Train Loss: 12429.0351562, Time: 28.7447s, Throughput: 173.666671 im/s
Epoch 9, Iter 39, Loss: 12557.9814453, Throughput: 173.530586 im/s
Epoch 9, Train Loss: 12091.6943359, Time: 28.7674s, Throughput: 173.529663 im/s
Epoch 10, Iter 39, Loss: 12032.6425781, Throughput: 172.909799 im/s
Epoch 10, Train Loss: 11842.4189453, Time: 28.8707s, Throughput: 172.908937 im/s
Epoch 11, Iter 39, Loss: 11233.8808594, Throughput: 172.680027 im/s
Epoch 11, Train Loss: 11881.8203125, Time: 28.9093s, Throughput: 172.678068 im/s
Epoch 12, Iter 39, Loss: 11767.1806641, Throughput: 173.064369 im/s
Epoch 12, Train Loss: 11480.3769531, Time: 28.8449s, Throughput: 173.063366 im/s
Epoch 13, Iter 39, Loss: 10965.8017578, Throughput: 173.035795 im/s
Epoch 13, Train Loss: 11359.2998047, Time: 28.8497s, Throughput: 173.034896 im/s
Epoch 14, Iter 39, Loss: 11361.6689453, Throughput: 172.802171 im/s
Epoch 14, Train Loss: 11267.4931641, Time: 28.8887s, Throughput: 172.801237 im/s
Epoch 15, Iter 39, Loss: 11840.8662109, Throughput: 172.553666 im/s
Epoch 15, Train Loss: 11086.5234375, Time: 28.9303s, Throughput: 172.552766 im/s
Epoch 16, Iter 39, Loss: 11108.9365234, Throughput: 171.833307 im/s
Epoch 16, Train Loss: 11022.2568359, Time: 29.0516s, Throughput: 171.832325 im/s
Epoch 17, Iter 39, Loss: 11076.4492188, Throughput: 172.857322 im/s
Epoch 17, Train Loss: 10826.5615234, Time: 28.8795s, Throughput: 172.856340 im/s
Epoch 18, Iter 39, Loss: 10288.5136719, Throughput: 169.878512 im/s
Epoch 18, Train Loss: 10797.3017578, Time: 29.3859s, Throughput: 169.877546 im/s
Epoch 19, Iter 39, Loss: 10829.6015625, Throughput: 171.793148 im/s
Epoch 19, Train Loss: 10646.8417969, Time: 29.0584s, Throughput: 171.792130 im/s
Epoch 20, Iter 39, Loss: 11023.6376953, Throughput: 172.272649 im/s
Epoch 20, Train Loss: 10687.5175781, Time: 28.9775s, Throughput: 172.271602 im/s
Epoch 21, Iter 39, Loss: 10095.2744141, Throughput: 172.394906 im/s
Epoch 21, Train Loss: 10527.9931641, Time: 28.9570s, Throughput: 172.393787 im/s
Epoch 22, Iter 39, Loss: 11408.2861328, Throughput: 170.962089 im/s
Epoch 22, Train Loss: 10448.0849609, Time: 29.1996s, Throughput: 170.961153 im/s
Epoch 23, Iter 39, Loss: 10491.8730469, Throughput: 172.048021 im/s
Epoch 23, Train Loss: 10381.9531250, Time: 29.0153s, Throughput: 172.046936 im/s
Epoch 24, Iter 39, Loss: 9887.3007812, Throughput: 171.535699 im/s
Epoch 24, Train Loss: 10269.2919922, Time: 29.1020s, Throughput: 171.534497 im/s
Epoch 25, Iter 39, Loss: 11027.6318359, Throughput: 172.428377 im/s
Epoch 25, Train Loss: 10305.6542969, Time: 28.9513s, Throughput: 172.427532 im/s
Epoch 26, Iter 39, Loss: 10163.6777344, Throughput: 171.008426 im/s
Epoch 26, Train Loss: 10131.0634766, Time: 29.1917s, Throughput: 171.007430 im/s
Epoch 27, Iter 39, Loss: 9590.3720703, Throughput: 172.175649 im/s
Epoch 27, Train Loss: 10125.3925781, Time: 28.9938s, Throughput: 172.174663 im/s
Epoch 28, Iter 39, Loss: 10130.7792969, Throughput: 172.546110 im/s
Epoch 28, Train Loss: 10118.1240234, Time: 28.9316s, Throughput: 172.544833 im/s
Epoch 29, Iter 39, Loss: 10186.9042969, Throughput: 172.184130 im/s
Epoch 29, Train Loss: 9964.9580078, Time: 28.9924s, Throughput: 172.183048 im/s
Epoch 30, Iter 39, Loss: 9643.0410156, Throughput: 173.311579 im/s
Epoch 30, Train Loss: 9969.3544922, Time: 28.8038s, Throughput: 173.310736 im/s
Epoch 31, Iter 39, Loss: 10081.8691406, Throughput: 173.845599 im/s
Epoch 31, Train Loss: 9996.8525391, Time: 28.7153s, Throughput: 173.844720 im/s
Epoch 32, Iter 39, Loss: 10309.7558594, Throughput: 173.815017 im/s
Epoch 32, Train Loss: 9813.7402344, Time: 28.7203s, Throughput: 173.814129 im/s
Epoch 33, Iter 39, Loss: 9901.4648438, Throughput: 173.285830 im/s
Epoch 33, Train Loss: 9810.7802734, Time: 28.8080s, Throughput: 173.284976 im/s
Epoch 34, Iter 39, Loss: 10107.9677734, Throughput: 173.537734 im/s
Epoch 34, Train Loss: 9781.9140625, Time: 28.7662s, Throughput: 173.536884 im/s
Epoch 35, Iter 39, Loss: 10401.5917969, Throughput: 172.588107 im/s
Epoch 35, Train Loss: 9789.4794922, Time: 28.9245s, Throughput: 172.587206 im/s
Epoch 36, Iter 39, Loss: 10096.9746094, Throughput: 172.922150 im/s
Epoch 36, Train Loss: 9761.7148438, Time: 28.8686s, Throughput: 172.921295 im/s
Epoch 37, Iter 39, Loss: 10288.2373047, Throughput: 173.108785 im/s
Epoch 37, Train Loss: 9705.8417969, Time: 28.8375s, Throughput: 173.107807 im/s
Epoch 38, Iter 39, Loss: 10162.2167969, Throughput: 173.144203 im/s
Epoch 38, Train Loss: 9606.4326172, Time: 28.8316s, Throughput: 173.143218 im/s
Epoch 39, Iter 39, Loss: 9504.4726562, Throughput: 172.186547 im/s
Epoch 39, Train Loss: 9607.6220703, Time: 28.9920s, Throughput: 172.185554 im/s
Epoch 40, Iter 39, Loss: 9152.0625000, Throughput: 173.111186 im/s
Epoch 40, Train Loss: 9518.9287109, Time: 28.8371s, Throughput: 173.110243 im/s
Epoch 41, Iter 39, Loss: 9542.3378906, Throughput: 173.016605 im/s
Epoch 41, Train Loss: 9553.4091797, Time: 28.8529s, Throughput: 173.015693 im/s
Epoch 42, Iter 39, Loss: 9555.3320312, Throughput: 173.440391 im/s
Epoch 42, Train Loss: 9501.6220703, Time: 28.7824s, Throughput: 173.439517 im/s
Epoch 43, Iter 39, Loss: 9566.7080078, Throughput: 172.908189 im/s
Epoch 43, Train Loss: 9492.0742188, Time: 28.8709s, Throughput: 172.907375 im/s
Epoch 44, Iter 39, Loss: 9910.8242188, Throughput: 173.361764 im/s
Epoch 44, Train Loss: 9477.9804688, Time: 28.7955s, Throughput: 173.360609 im/s
Epoch 45, Iter 39, Loss: 9965.0136719, Throughput: 172.765331 im/s
Epoch 45, Train Loss: 9454.9365234, Time: 28.8949s, Throughput: 172.764249 im/s
Epoch 46, Iter 39, Loss: 9559.8457031, Throughput: 171.469861 im/s
Epoch 46, Train Loss: 9425.6992188, Time: 29.1132s, Throughput: 171.468841 im/s
Epoch 47, Iter 39, Loss: 9505.5488281, Throughput: 173.025859 im/s
Epoch 47, Train Loss: 9359.9355469, Time: 28.8513s, Throughput: 173.024899 im/s
Epoch 48, Iter 39, Loss: 9101.6132812, Throughput: 173.726027 im/s
Epoch 48, Train Loss: 9353.9443359, Time: 28.7351s, Throughput: 173.724770 im/s
Epoch 49, Iter 39, Loss: 9033.0312500, Throughput: 173.359427 im/s
Epoch 49, Train Loss: 9278.9765625, Time: 28.7958s, Throughput: 173.358596 im/s
Epoch 50, Iter 39, Loss: 9301.8046875, Throughput: 173.917842 im/s
Epoch 50, Train Loss: 9300.5673828, Time: 28.7034s, Throughput: 173.916661 im/s

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
