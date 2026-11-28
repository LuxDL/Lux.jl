---
url: /previews/PR1834/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24730.7480469, Throughput: 16.606445 im/s
Epoch 1, Train Loss: 39706.4375000, Time: 300.8706s, Throughput: 16.591853 im/s
Epoch 2, Iter 39, Loss: 16725.8085938, Throughput: 172.906885 im/s
Epoch 2, Train Loss: 20224.1015625, Time: 28.8712s, Throughput: 172.905848 im/s
Epoch 3, Iter 39, Loss: 16007.0976562, Throughput: 173.681075 im/s
Epoch 3, Train Loss: 16587.7734375, Time: 28.7425s, Throughput: 173.680169 im/s
Epoch 4, Iter 39, Loss: 14697.3701172, Throughput: 172.932765 im/s
Epoch 4, Train Loss: 15014.8378906, Time: 28.8669s, Throughput: 172.931789 im/s
Epoch 5, Iter 39, Loss: 13574.2792969, Throughput: 174.088983 im/s
Epoch 5, Train Loss: 14011.2626953, Time: 28.6751s, Throughput: 174.088029 im/s
Epoch 6, Iter 39, Loss: 13872.4687500, Throughput: 172.482551 im/s
Epoch 6, Train Loss: 13334.4082031, Time: 28.9422s, Throughput: 172.481491 im/s
Epoch 7, Iter 39, Loss: 12811.9853516, Throughput: 173.573064 im/s
Epoch 7, Train Loss: 12943.7460938, Time: 28.7604s, Throughput: 173.572194 im/s
Epoch 8, Iter 39, Loss: 12668.6796875, Throughput: 172.864791 im/s
Epoch 8, Train Loss: 12458.7333984, Time: 28.8782s, Throughput: 172.863858 im/s
Epoch 9, Iter 39, Loss: 11335.0332031, Throughput: 173.401158 im/s
Epoch 9, Train Loss: 12205.7294922, Time: 28.7889s, Throughput: 173.400290 im/s
Epoch 10, Iter 39, Loss: 11581.7226562, Throughput: 172.905423 im/s
Epoch 10, Train Loss: 11902.2539062, Time: 28.8715s, Throughput: 172.904176 im/s
Epoch 11, Iter 39, Loss: 12122.6796875, Throughput: 174.029883 im/s
Epoch 11, Train Loss: 11694.4384766, Time: 28.6849s, Throughput: 174.028960 im/s
Epoch 12, Iter 39, Loss: 11752.1777344, Throughput: 173.251556 im/s
Epoch 12, Train Loss: 11538.8046875, Time: 28.8138s, Throughput: 173.250486 im/s
Epoch 13, Iter 39, Loss: 11490.9257812, Throughput: 173.608576 im/s
Epoch 13, Train Loss: 11417.4501953, Time: 28.7545s, Throughput: 173.607585 im/s
Epoch 14, Iter 39, Loss: 11425.6230469, Throughput: 172.615696 im/s
Epoch 14, Train Loss: 11267.7294922, Time: 28.9199s, Throughput: 172.614818 im/s
Epoch 15, Iter 39, Loss: 10772.4873047, Throughput: 173.418597 im/s
Epoch 15, Train Loss: 11200.6591797, Time: 28.7860s, Throughput: 173.417434 im/s
Epoch 16, Iter 39, Loss: 10680.6289062, Throughput: 173.530550 im/s
Epoch 16, Train Loss: 10907.2216797, Time: 28.7674s, Throughput: 173.529579 im/s
Epoch 17, Iter 39, Loss: 11232.2167969, Throughput: 172.203541 im/s
Epoch 17, Train Loss: 10865.5917969, Time: 28.9891s, Throughput: 172.202602 im/s
Epoch 18, Iter 39, Loss: 10863.1162109, Throughput: 172.971342 im/s
Epoch 18, Train Loss: 10813.9404297, Time: 28.8605s, Throughput: 172.970263 im/s
Epoch 19, Iter 39, Loss: 10675.4355469, Throughput: 173.554761 im/s
Epoch 19, Train Loss: 10712.9980469, Time: 28.7634s, Throughput: 173.553849 im/s
Epoch 20, Iter 39, Loss: 10984.2070312, Throughput: 172.481694 im/s
Epoch 20, Train Loss: 10549.1845703, Time: 28.9424s, Throughput: 172.480507 im/s
Epoch 21, Iter 39, Loss: 10574.4316406, Throughput: 173.751139 im/s
Epoch 21, Train Loss: 10507.0957031, Time: 28.7309s, Throughput: 173.749976 im/s
Epoch 22, Iter 39, Loss: 10091.7207031, Throughput: 173.124260 im/s
Epoch 22, Train Loss: 10438.2343750, Time: 28.8350s, Throughput: 173.123217 im/s
Epoch 23, Iter 39, Loss: 10477.1621094, Throughput: 173.757235 im/s
Epoch 23, Train Loss: 10301.8984375, Time: 28.7299s, Throughput: 173.756091 im/s
Epoch 24, Iter 39, Loss: 9859.6640625, Throughput: 174.263569 im/s
Epoch 24, Train Loss: 10316.7304688, Time: 28.6464s, Throughput: 174.262504 im/s
Epoch 25, Iter 39, Loss: 10439.7734375, Throughput: 172.944040 im/s
Epoch 25, Train Loss: 10230.1259766, Time: 28.8650s, Throughput: 172.943076 im/s
Epoch 26, Iter 39, Loss: 10284.0234375, Throughput: 173.994810 im/s
Epoch 26, Train Loss: 10172.3505859, Time: 28.6907s, Throughput: 173.993881 im/s
Epoch 27, Iter 39, Loss: 10478.1562500, Throughput: 173.138317 im/s
Epoch 27, Train Loss: 10228.0449219, Time: 28.8326s, Throughput: 173.137273 im/s
Epoch 28, Iter 39, Loss: 10809.6357422, Throughput: 173.426285 im/s
Epoch 28, Train Loss: 10146.2333984, Time: 28.7847s, Throughput: 173.425387 im/s
Epoch 29, Iter 39, Loss: 10346.1123047, Throughput: 172.999667 im/s
Epoch 29, Train Loss: 10066.1425781, Time: 28.8557s, Throughput: 172.998935 im/s
Epoch 30, Iter 39, Loss: 10035.2001953, Throughput: 174.598262 im/s
Epoch 30, Train Loss: 9929.2890625, Time: 28.5915s, Throughput: 174.597339 im/s
Epoch 31, Iter 39, Loss: 9346.6552734, Throughput: 173.217674 im/s
Epoch 31, Train Loss: 9869.6357422, Time: 28.8194s, Throughput: 173.216760 im/s
Epoch 32, Iter 39, Loss: 10509.4824219, Throughput: 174.587917 im/s
Epoch 32, Train Loss: 9923.1279297, Time: 28.5932s, Throughput: 174.586971 im/s
Epoch 33, Iter 39, Loss: 9950.0771484, Throughput: 173.586976 im/s
Epoch 33, Train Loss: 9846.2744141, Time: 28.7581s, Throughput: 173.585997 im/s
Epoch 34, Iter 39, Loss: 9449.0087891, Throughput: 174.249955 im/s
Epoch 34, Train Loss: 9747.6503906, Time: 28.6487s, Throughput: 174.249005 im/s
Epoch 35, Iter 39, Loss: 9632.4980469, Throughput: 173.450978 im/s
Epoch 35, Train Loss: 9734.3369141, Time: 28.7806s, Throughput: 173.450044 im/s
Epoch 36, Iter 39, Loss: 10228.6005859, Throughput: 174.130616 im/s
Epoch 36, Train Loss: 9730.4414062, Time: 28.6683s, Throughput: 174.129620 im/s
Epoch 37, Iter 39, Loss: 10681.3046875, Throughput: 174.175751 im/s
Epoch 37, Train Loss: 9644.7890625, Time: 28.6609s, Throughput: 174.174845 im/s
Epoch 38, Iter 39, Loss: 9535.1718750, Throughput: 173.276506 im/s
Epoch 38, Train Loss: 9729.9931641, Time: 28.8096s, Throughput: 173.275508 im/s
Epoch 39, Iter 39, Loss: 9256.8281250, Throughput: 172.742392 im/s
Epoch 39, Train Loss: 9612.1074219, Time: 28.8987s, Throughput: 172.741424 im/s
Epoch 40, Iter 39, Loss: 9745.6162109, Throughput: 172.988185 im/s
Epoch 40, Train Loss: 9622.4130859, Time: 28.8576s, Throughput: 172.987256 im/s
Epoch 41, Iter 39, Loss: 9409.7441406, Throughput: 171.462386 im/s
Epoch 41, Train Loss: 9550.7304688, Time: 29.1144s, Throughput: 171.461291 im/s
Epoch 42, Iter 39, Loss: 9387.7128906, Throughput: 170.667151 im/s
Epoch 42, Train Loss: 9512.9638672, Time: 29.2501s, Throughput: 170.666276 im/s
Epoch 43, Iter 39, Loss: 9786.9707031, Throughput: 172.063498 im/s
Epoch 43, Train Loss: 9582.8681641, Time: 29.0127s, Throughput: 172.062644 im/s
Epoch 44, Iter 39, Loss: 9938.6445312, Throughput: 173.645443 im/s
Epoch 44, Train Loss: 9525.1826172, Time: 28.7484s, Throughput: 173.644494 im/s
Epoch 45, Iter 39, Loss: 9003.6035156, Throughput: 172.114517 im/s
Epoch 45, Train Loss: 9455.4785156, Time: 29.0041s, Throughput: 172.113515 im/s
Epoch 46, Iter 39, Loss: 9667.1933594, Throughput: 172.782498 im/s
Epoch 46, Train Loss: 9419.4082031, Time: 28.8920s, Throughput: 172.781380 im/s
Epoch 47, Iter 39, Loss: 9781.4980469, Throughput: 173.124520 im/s
Epoch 47, Train Loss: 9359.4101562, Time: 28.8349s, Throughput: 173.123445 im/s
Epoch 48, Iter 39, Loss: 8887.2929688, Throughput: 172.566360 im/s
Epoch 48, Train Loss: 9293.5566406, Time: 28.9282s, Throughput: 172.565186 im/s
Epoch 49, Iter 39, Loss: 9533.9101562, Throughput: 173.310851 im/s
Epoch 49, Train Loss: 9327.0517578, Time: 28.8039s, Throughput: 173.309684 im/s
Epoch 50, Iter 39, Loss: 9381.4921875, Throughput: 173.330240 im/s
Epoch 50, Train Loss: 9321.4267578, Time: 28.8007s, Throughput: 173.329253 im/s

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
