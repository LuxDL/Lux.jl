---
url: /previews/PR1855/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24283.3710938, Throughput: 19.008329 im/s
Epoch 1, Train Loss: 39658.3750000, Time: 262.8445s, Throughput: 18.992221 im/s
Epoch 2, Iter 39, Loss: 18476.4511719, Throughput: 204.982892 im/s
Epoch 2, Train Loss: 20268.2753906, Time: 24.3534s, Throughput: 204.981696 im/s
Epoch 3, Iter 39, Loss: 16302.3349609, Throughput: 205.232051 im/s
Epoch 3, Train Loss: 16724.7480469, Time: 24.3238s, Throughput: 205.230685 im/s
Epoch 4, Iter 39, Loss: 13685.6523438, Throughput: 204.365495 im/s
Epoch 4, Train Loss: 14983.5820312, Time: 24.4270s, Throughput: 204.364324 im/s
Epoch 5, Iter 39, Loss: 13580.0712891, Throughput: 204.373997 im/s
Epoch 5, Train Loss: 13975.6601562, Time: 24.4260s, Throughput: 204.372732 im/s
Epoch 6, Iter 39, Loss: 12460.0273438, Throughput: 203.963936 im/s
Epoch 6, Train Loss: 13368.7958984, Time: 24.4751s, Throughput: 203.962712 im/s
Epoch 7, Iter 39, Loss: 12310.0117188, Throughput: 204.514897 im/s
Epoch 7, Train Loss: 12785.2236328, Time: 24.4091s, Throughput: 204.513606 im/s
Epoch 8, Iter 39, Loss: 11995.9843750, Throughput: 204.859234 im/s
Epoch 8, Train Loss: 12427.6943359, Time: 24.3681s, Throughput: 204.858007 im/s
Epoch 9, Iter 39, Loss: 12417.1230469, Throughput: 203.988713 im/s
Epoch 9, Train Loss: 12228.3095703, Time: 24.4721s, Throughput: 203.987447 im/s
Epoch 10, Iter 39, Loss: 10597.7558594, Throughput: 203.613013 im/s
Epoch 10, Train Loss: 11879.2089844, Time: 24.5172s, Throughput: 203.611825 im/s
Epoch 11, Iter 39, Loss: 11143.6386719, Throughput: 202.574431 im/s
Epoch 11, Train Loss: 11687.2587891, Time: 24.6430s, Throughput: 202.573141 im/s
Epoch 12, Iter 39, Loss: 11180.1640625, Throughput: 203.108801 im/s
Epoch 12, Train Loss: 11405.0302734, Time: 24.5781s, Throughput: 203.107314 im/s
Epoch 13, Iter 39, Loss: 11614.5390625, Throughput: 203.730415 im/s
Epoch 13, Train Loss: 11334.5400391, Time: 24.5031s, Throughput: 203.729200 im/s
Epoch 14, Iter 39, Loss: 10419.3886719, Throughput: 204.757461 im/s
Epoch 14, Train Loss: 11180.7060547, Time: 24.3802s, Throughput: 204.756293 im/s
Epoch 15, Iter 39, Loss: 10935.2568359, Throughput: 202.936851 im/s
Epoch 15, Train Loss: 11037.7880859, Time: 24.5989s, Throughput: 202.935639 im/s
Epoch 16, Iter 39, Loss: 10942.7851562, Throughput: 203.936922 im/s
Epoch 16, Train Loss: 10936.3486328, Time: 24.4783s, Throughput: 203.935690 im/s
Epoch 17, Iter 39, Loss: 11137.0087891, Throughput: 204.350287 im/s
Epoch 17, Train Loss: 10779.6826172, Time: 24.4288s, Throughput: 204.348948 im/s
Epoch 18, Iter 39, Loss: 10770.1035156, Throughput: 203.926941 im/s
Epoch 18, Train Loss: 10759.0556641, Time: 24.4795s, Throughput: 203.925584 im/s
Epoch 19, Iter 39, Loss: 10567.6093750, Throughput: 203.119562 im/s
Epoch 19, Train Loss: 10699.0302734, Time: 24.5768s, Throughput: 203.118237 im/s
Epoch 20, Iter 39, Loss: 10388.4160156, Throughput: 202.627098 im/s
Epoch 20, Train Loss: 10503.0693359, Time: 24.6366s, Throughput: 202.625625 im/s
Epoch 21, Iter 39, Loss: 11252.3876953, Throughput: 198.711757 im/s
Epoch 21, Train Loss: 10492.2705078, Time: 25.1220s, Throughput: 198.710437 im/s
Epoch 22, Iter 39, Loss: 10729.2031250, Throughput: 202.312811 im/s
Epoch 22, Train Loss: 10355.4267578, Time: 24.6748s, Throughput: 202.311589 im/s
Epoch 23, Iter 39, Loss: 10913.2441406, Throughput: 202.374700 im/s
Epoch 23, Train Loss: 10318.3037109, Time: 24.6673s, Throughput: 202.373288 im/s
Epoch 24, Iter 39, Loss: 9752.7421875, Throughput: 203.570028 im/s
Epoch 24, Train Loss: 10279.0166016, Time: 24.5224s, Throughput: 203.568658 im/s
Epoch 25, Iter 39, Loss: 9742.5927734, Throughput: 202.420417 im/s
Epoch 25, Train Loss: 10143.0781250, Time: 24.6617s, Throughput: 202.419038 im/s
Epoch 26, Iter 39, Loss: 10156.6181641, Throughput: 203.371684 im/s
Epoch 26, Train Loss: 10221.1386719, Time: 24.5464s, Throughput: 203.370226 im/s
Epoch 27, Iter 39, Loss: 10179.0449219, Throughput: 205.193155 im/s
Epoch 27, Train Loss: 10172.9931641, Time: 24.3285s, Throughput: 205.191579 im/s
Epoch 28, Iter 39, Loss: 10267.5595703, Throughput: 204.726877 im/s
Epoch 28, Train Loss: 10125.6474609, Time: 24.3839s, Throughput: 204.725634 im/s
Epoch 29, Iter 39, Loss: 10710.0410156, Throughput: 204.263132 im/s
Epoch 29, Train Loss: 10018.9433594, Time: 24.4392s, Throughput: 204.261763 im/s
Epoch 30, Iter 39, Loss: 9728.7587891, Throughput: 204.998026 im/s
Epoch 30, Train Loss: 9928.8359375, Time: 24.3516s, Throughput: 204.996663 im/s
Epoch 31, Iter 39, Loss: 9714.4570312, Throughput: 203.801834 im/s
Epoch 31, Train Loss: 9920.2333984, Time: 24.4945s, Throughput: 203.800461 im/s
Epoch 32, Iter 39, Loss: 9744.4970703, Throughput: 203.889646 im/s
Epoch 32, Train Loss: 9811.3955078, Time: 24.4840s, Throughput: 203.888147 im/s
Epoch 33, Iter 39, Loss: 9318.5361328, Throughput: 204.080499 im/s
Epoch 33, Train Loss: 9784.4013672, Time: 24.4611s, Throughput: 204.079230 im/s
Epoch 34, Iter 39, Loss: 9676.9687500, Throughput: 203.997750 im/s
Epoch 34, Train Loss: 9770.3945312, Time: 24.4710s, Throughput: 203.996333 im/s
Epoch 35, Iter 39, Loss: 9627.4511719, Throughput: 203.443858 im/s
Epoch 35, Train Loss: 9682.3593750, Time: 24.5376s, Throughput: 203.442498 im/s
Epoch 36, Iter 39, Loss: 9579.0625000, Throughput: 203.552531 im/s
Epoch 36, Train Loss: 9679.3144531, Time: 24.5245s, Throughput: 203.551320 im/s
Epoch 37, Iter 39, Loss: 9730.5810547, Throughput: 202.690367 im/s
Epoch 37, Train Loss: 9732.3955078, Time: 24.6288s, Throughput: 202.689148 im/s
Epoch 38, Iter 39, Loss: 9520.4892578, Throughput: 203.223206 im/s
Epoch 38, Train Loss: 9615.0029297, Time: 24.5643s, Throughput: 203.221965 im/s
Epoch 39, Iter 39, Loss: 9739.9716797, Throughput: 203.473719 im/s
Epoch 39, Train Loss: 9611.7929688, Time: 24.5340s, Throughput: 203.472467 im/s
Epoch 40, Iter 39, Loss: 9060.3271484, Throughput: 203.632115 im/s
Epoch 40, Train Loss: 9542.1923828, Time: 24.5149s, Throughput: 203.630936 im/s
Epoch 41, Iter 39, Loss: 9582.4707031, Throughput: 201.628424 im/s
Epoch 41, Train Loss: 9536.8476562, Time: 24.7586s, Throughput: 201.626933 im/s
Epoch 42, Iter 39, Loss: 9138.6943359, Throughput: 203.949345 im/s
Epoch 42, Train Loss: 9528.1230469, Time: 24.4768s, Throughput: 203.948169 im/s
Epoch 43, Iter 39, Loss: 10074.7509766, Throughput: 205.100577 im/s
Epoch 43, Train Loss: 9456.1103516, Time: 24.3394s, Throughput: 205.099239 im/s
Epoch 44, Iter 39, Loss: 9800.9531250, Throughput: 205.395313 im/s
Epoch 44, Train Loss: 9440.6748047, Time: 24.3045s, Throughput: 205.393927 im/s
Epoch 45, Iter 39, Loss: 9158.3574219, Throughput: 204.604646 im/s
Epoch 45, Train Loss: 9391.1943359, Time: 24.3984s, Throughput: 204.603346 im/s
Epoch 46, Iter 39, Loss: 9942.6396484, Throughput: 204.534679 im/s
Epoch 46, Train Loss: 9378.8662109, Time: 24.4068s, Throughput: 204.533231 im/s
Epoch 47, Iter 39, Loss: 9118.2548828, Throughput: 204.400291 im/s
Epoch 47, Train Loss: 9325.1572266, Time: 24.4228s, Throughput: 204.398942 im/s
Epoch 48, Iter 39, Loss: 9553.3339844, Throughput: 204.756848 im/s
Epoch 48, Train Loss: 9322.6875000, Time: 24.3803s, Throughput: 204.755378 im/s
Epoch 49, Iter 39, Loss: 9103.6689453, Throughput: 205.339130 im/s
Epoch 49, Train Loss: 9280.9628906, Time: 24.3112s, Throughput: 205.337652 im/s
Epoch 50, Iter 39, Loss: 9457.8417969, Throughput: 205.026063 im/s
Epoch 50, Train Loss: 9244.5810547, Time: 24.3483s, Throughput: 205.024908 im/s

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
