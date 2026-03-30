---
url: /previews/PR1691/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24193.9824219, Throughput: 4.192510 im/s
Epoch 1, Train Loss: 39607.9921875, Time: 1191.0038s, Throughput: 4.191422 im/s
Epoch 2, Iter 39, Loss: 17586.8632812, Throughput: 72.520549 im/s
Epoch 2, Train Loss: 20225.2167969, Time: 68.8358s, Throughput: 72.520362 im/s
Epoch 3, Iter 39, Loss: 15639.7089844, Throughput: 72.440721 im/s
Epoch 3, Train Loss: 16551.3496094, Time: 68.9117s, Throughput: 72.440540 im/s
Epoch 4, Iter 39, Loss: 15036.6865234, Throughput: 72.948728 im/s
Epoch 4, Train Loss: 15004.6005859, Time: 68.4318s, Throughput: 72.948544 im/s
Epoch 5, Iter 39, Loss: 13516.3691406, Throughput: 72.529211 im/s
Epoch 5, Train Loss: 14048.2080078, Time: 68.8276s, Throughput: 72.529013 im/s
Epoch 6, Iter 39, Loss: 12727.0908203, Throughput: 72.407278 im/s
Epoch 6, Train Loss: 13311.5634766, Time: 68.9435s, Throughput: 72.407115 im/s
Epoch 7, Iter 39, Loss: 12857.2500000, Throughput: 72.504446 im/s
Epoch 7, Train Loss: 12888.9082031, Time: 68.8511s, Throughput: 72.504253 im/s
Epoch 8, Iter 39, Loss: 12897.4423828, Throughput: 73.037996 im/s
Epoch 8, Train Loss: 12538.4716797, Time: 68.3482s, Throughput: 73.037812 im/s
Epoch 9, Iter 39, Loss: 12209.9257812, Throughput: 72.600616 im/s
Epoch 9, Train Loss: 12178.0517578, Time: 68.7599s, Throughput: 72.600451 im/s
Epoch 10, Iter 39, Loss: 12160.5507812, Throughput: 72.824486 im/s
Epoch 10, Train Loss: 11913.4765625, Time: 68.5485s, Throughput: 72.824301 im/s
Epoch 11, Iter 39, Loss: 11425.6386719, Throughput: 72.900815 im/s
Epoch 11, Train Loss: 11696.7500000, Time: 68.4768s, Throughput: 72.900633 im/s
Epoch 12, Iter 39, Loss: 11638.3515625, Throughput: 72.781572 im/s
Epoch 12, Train Loss: 11472.1308594, Time: 68.5889s, Throughput: 72.781407 im/s
Epoch 13, Iter 39, Loss: 10879.6640625, Throughput: 72.796677 im/s
Epoch 13, Train Loss: 11370.2050781, Time: 68.5747s, Throughput: 72.796500 im/s
Epoch 14, Iter 39, Loss: 11147.1230469, Throughput: 72.921428 im/s
Epoch 14, Train Loss: 11269.9824219, Time: 68.4574s, Throughput: 72.921247 im/s
Epoch 15, Iter 39, Loss: 10422.0087891, Throughput: 72.980910 im/s
Epoch 15, Train Loss: 11148.9931641, Time: 68.4016s, Throughput: 72.980732 im/s
Epoch 16, Iter 39, Loss: 11354.0078125, Throughput: 72.885521 im/s
Epoch 16, Train Loss: 11031.1406250, Time: 68.4911s, Throughput: 72.885339 im/s
Epoch 17, Iter 39, Loss: 11655.7089844, Throughput: 72.680141 im/s
Epoch 17, Train Loss: 10940.3769531, Time: 68.6847s, Throughput: 72.679968 im/s
Epoch 18, Iter 39, Loss: 10684.9072266, Throughput: 72.865405 im/s
Epoch 18, Train Loss: 10788.3066406, Time: 68.5100s, Throughput: 72.865238 im/s
Epoch 19, Iter 39, Loss: 10586.2978516, Throughput: 72.688411 im/s
Epoch 19, Train Loss: 10691.9746094, Time: 68.6769s, Throughput: 72.688241 im/s
Epoch 20, Iter 39, Loss: 10920.5654297, Throughput: 72.885659 im/s
Epoch 20, Train Loss: 10646.0253906, Time: 68.4910s, Throughput: 72.885477 im/s
Epoch 21, Iter 39, Loss: 10738.3964844, Throughput: 72.986217 im/s
Epoch 21, Train Loss: 10610.5859375, Time: 68.3967s, Throughput: 72.986016 im/s
Epoch 22, Iter 39, Loss: 10803.6113281, Throughput: 72.589297 im/s
Epoch 22, Train Loss: 10490.0888672, Time: 68.7706s, Throughput: 72.589121 im/s
Epoch 23, Iter 39, Loss: 9947.0556641, Throughput: 73.214205 im/s
Epoch 23, Train Loss: 10448.3388672, Time: 68.1837s, Throughput: 73.214029 im/s
Epoch 24, Iter 39, Loss: 11238.3447266, Throughput: 72.457704 im/s
Epoch 24, Train Loss: 10396.8007812, Time: 68.8955s, Throughput: 72.457546 im/s
Epoch 25, Iter 39, Loss: 10430.6992188, Throughput: 72.981032 im/s
Epoch 25, Train Loss: 10256.0576172, Time: 68.4015s, Throughput: 72.980838 im/s
Epoch 26, Iter 39, Loss: 10891.0830078, Throughput: 72.631854 im/s
Epoch 26, Train Loss: 10239.3769531, Time: 68.7304s, Throughput: 72.631657 im/s
Epoch 27, Iter 39, Loss: 9536.9580078, Throughput: 72.717539 im/s
Epoch 27, Train Loss: 10280.5917969, Time: 68.6494s, Throughput: 72.717357 im/s
Epoch 28, Iter 39, Loss: 10720.5693359, Throughput: 72.859133 im/s
Epoch 28, Train Loss: 10174.3691406, Time: 68.5159s, Throughput: 72.858957 im/s
Epoch 29, Iter 39, Loss: 10343.1396484, Throughput: 72.840518 im/s
Epoch 29, Train Loss: 10043.3291016, Time: 68.5334s, Throughput: 72.840361 im/s
Epoch 30, Iter 39, Loss: 9468.3808594, Throughput: 72.343722 im/s
Epoch 30, Train Loss: 10047.3056641, Time: 69.0041s, Throughput: 72.343566 im/s
Epoch 31, Iter 39, Loss: 9898.7578125, Throughput: 72.729389 im/s
Epoch 31, Train Loss: 9963.2050781, Time: 68.6382s, Throughput: 72.729217 im/s
Epoch 32, Iter 39, Loss: 10395.5878906, Throughput: 72.770558 im/s
Epoch 32, Train Loss: 9923.6865234, Time: 68.5993s, Throughput: 72.770387 im/s
Epoch 33, Iter 39, Loss: 9717.3652344, Throughput: 73.105588 im/s
Epoch 33, Train Loss: 9887.1669922, Time: 68.2850s, Throughput: 73.105392 im/s
Epoch 34, Iter 39, Loss: 10297.4472656, Throughput: 72.901265 im/s
Epoch 34, Train Loss: 9856.4667969, Time: 68.4764s, Throughput: 72.901075 im/s
Epoch 35, Iter 39, Loss: 10165.6523438, Throughput: 72.248793 im/s
Epoch 35, Train Loss: 9728.1318359, Time: 69.0947s, Throughput: 72.248628 im/s
Epoch 36, Iter 39, Loss: 9809.1757812, Throughput: 72.427189 im/s
Epoch 36, Train Loss: 9764.1542969, Time: 68.9246s, Throughput: 72.426997 im/s
Epoch 37, Iter 39, Loss: 9659.7324219, Throughput: 72.697770 im/s
Epoch 37, Train Loss: 9695.1748047, Time: 68.6680s, Throughput: 72.697602 im/s
Epoch 38, Iter 39, Loss: 9817.7832031, Throughput: 72.818907 im/s
Epoch 38, Train Loss: 9685.0419922, Time: 68.5538s, Throughput: 72.818716 im/s
Epoch 39, Iter 39, Loss: 9260.7890625, Throughput: 72.665568 im/s
Epoch 39, Train Loss: 9629.7939453, Time: 68.6984s, Throughput: 72.665397 im/s
Epoch 40, Iter 39, Loss: 9744.1816406, Throughput: 72.570415 im/s
Epoch 40, Train Loss: 9678.7031250, Time: 68.7885s, Throughput: 72.570246 im/s
Epoch 41, Iter 39, Loss: 10011.2685547, Throughput: 72.454374 im/s
Epoch 41, Train Loss: 9623.9169922, Time: 68.8987s, Throughput: 72.454180 im/s
Epoch 42, Iter 39, Loss: 9623.6445312, Throughput: 72.786750 im/s
Epoch 42, Train Loss: 9594.1572266, Time: 68.5841s, Throughput: 72.786568 im/s
Epoch 43, Iter 39, Loss: 10378.0761719, Throughput: 72.743893 im/s
Epoch 43, Train Loss: 9531.0253906, Time: 68.6245s, Throughput: 72.743713 im/s
Epoch 44, Iter 39, Loss: 9185.0947266, Throughput: 73.239529 im/s
Epoch 44, Train Loss: 9473.5605469, Time: 68.1601s, Throughput: 73.239362 im/s
Epoch 45, Iter 39, Loss: 9669.9208984, Throughput: 72.576907 im/s
Epoch 45, Train Loss: 9463.4189453, Time: 68.7824s, Throughput: 72.576718 im/s
Epoch 46, Iter 39, Loss: 9588.8789062, Throughput: 72.981279 im/s
Epoch 46, Train Loss: 9435.0761719, Time: 68.4013s, Throughput: 72.981109 im/s
Epoch 47, Iter 39, Loss: 9490.5234375, Throughput: 72.452903 im/s
Epoch 47, Train Loss: 9485.6035156, Time: 68.9001s, Throughput: 72.452713 im/s
Epoch 48, Iter 39, Loss: 9319.1855469, Throughput: 72.879070 im/s
Epoch 48, Train Loss: 9398.6142578, Time: 68.4972s, Throughput: 72.878880 im/s
Epoch 49, Iter 39, Loss: 9035.8750000, Throughput: 72.520761 im/s
Epoch 49, Train Loss: 9381.6347656, Time: 68.8356s, Throughput: 72.520570 im/s
Epoch 50, Iter 39, Loss: 9177.4472656, Throughput: 72.550854 im/s
Epoch 50, Train Loss: 9438.2695312, Time: 68.8071s, Throughput: 72.550691 im/s

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
Julia Version 1.12.5
Commit 5fe89b8ddc1 (2026-02-09 16:05 UTC)
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
