---
url: /previews/PR1632/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 22975.1132812, Throughput: 19.165963 im/s
Epoch 1, Train Loss: 39532.4960938, Time: 260.6824s, Throughput: 19.149737 im/s
Epoch 2, Iter 39, Loss: 17990.7363281, Throughput: 203.480818 im/s
Epoch 2, Train Loss: 20273.7734375, Time: 24.5332s, Throughput: 203.479491 im/s
Epoch 3, Iter 39, Loss: 15712.6308594, Throughput: 202.807644 im/s
Epoch 3, Train Loss: 16653.0273438, Time: 24.6146s, Throughput: 202.806449 im/s
Epoch 4, Iter 39, Loss: 15548.3164062, Throughput: 202.201290 im/s
Epoch 4, Train Loss: 15026.6474609, Time: 24.6884s, Throughput: 202.200059 im/s
Epoch 5, Iter 39, Loss: 14063.7294922, Throughput: 203.262063 im/s
Epoch 5, Train Loss: 14116.6972656, Time: 24.5596s, Throughput: 203.260871 im/s
Epoch 6, Iter 39, Loss: 14254.9902344, Throughput: 203.361436 im/s
Epoch 6, Train Loss: 13385.5742188, Time: 24.5476s, Throughput: 203.360028 im/s
Epoch 7, Iter 39, Loss: 12980.7343750, Throughput: 202.156415 im/s
Epoch 7, Train Loss: 12863.8750000, Time: 24.6939s, Throughput: 202.155146 im/s
Epoch 8, Iter 39, Loss: 12382.8027344, Throughput: 203.548122 im/s
Epoch 8, Train Loss: 12552.5576172, Time: 24.5251s, Throughput: 203.546885 im/s
Epoch 9, Iter 39, Loss: 12825.7509766, Throughput: 203.765923 im/s
Epoch 9, Train Loss: 12227.7431641, Time: 24.4988s, Throughput: 203.764683 im/s
Epoch 10, Iter 39, Loss: 12913.8730469, Throughput: 203.694499 im/s
Epoch 10, Train Loss: 11934.3906250, Time: 24.5075s, Throughput: 203.693120 im/s
Epoch 11, Iter 39, Loss: 11590.1015625, Throughput: 203.827082 im/s
Epoch 11, Train Loss: 11858.1201172, Time: 24.4915s, Throughput: 203.825824 im/s
Epoch 12, Iter 39, Loss: 11007.8769531, Throughput: 203.747344 im/s
Epoch 12, Train Loss: 11633.0537109, Time: 24.5011s, Throughput: 203.745880 im/s
Epoch 13, Iter 39, Loss: 11871.9199219, Throughput: 203.769907 im/s
Epoch 13, Train Loss: 11370.2158203, Time: 24.4984s, Throughput: 203.768402 im/s
Epoch 14, Iter 39, Loss: 10824.0664062, Throughput: 199.672996 im/s
Epoch 14, Train Loss: 11322.4521484, Time: 25.0010s, Throughput: 199.671726 im/s
Epoch 15, Iter 39, Loss: 10595.5908203, Throughput: 202.847859 im/s
Epoch 15, Train Loss: 11137.3964844, Time: 24.6097s, Throughput: 202.846672 im/s
Epoch 16, Iter 39, Loss: 11443.9365234, Throughput: 202.962652 im/s
Epoch 16, Train Loss: 11108.8251953, Time: 24.5958s, Throughput: 202.961431 im/s
Epoch 17, Iter 39, Loss: 11862.4921875, Throughput: 203.472143 im/s
Epoch 17, Train Loss: 10985.2734375, Time: 24.5342s, Throughput: 203.470842 im/s
Epoch 18, Iter 39, Loss: 10686.7519531, Throughput: 204.191254 im/s
Epoch 18, Train Loss: 10834.3105469, Time: 24.4478s, Throughput: 204.190051 im/s
Epoch 19, Iter 39, Loss: 10430.0117188, Throughput: 202.499908 im/s
Epoch 19, Train Loss: 10851.3417969, Time: 24.6520s, Throughput: 202.498668 im/s
Epoch 20, Iter 39, Loss: 11117.5957031, Throughput: 204.161558 im/s
Epoch 20, Train Loss: 10722.4267578, Time: 24.4514s, Throughput: 204.160306 im/s
Epoch 21, Iter 39, Loss: 10612.8203125, Throughput: 204.143282 im/s
Epoch 21, Train Loss: 10581.3300781, Time: 24.4536s, Throughput: 204.142039 im/s
Epoch 22, Iter 39, Loss: 9913.6279297, Throughput: 202.752321 im/s
Epoch 22, Train Loss: 10452.2597656, Time: 24.6213s, Throughput: 202.751078 im/s
Epoch 23, Iter 39, Loss: 10111.5527344, Throughput: 202.359309 im/s
Epoch 23, Train Loss: 10387.2207031, Time: 24.6692s, Throughput: 202.357964 im/s
Epoch 24, Iter 39, Loss: 9941.8867188, Throughput: 203.063954 im/s
Epoch 24, Train Loss: 10313.4482422, Time: 24.5836s, Throughput: 203.062377 im/s
Epoch 25, Iter 39, Loss: 10091.6660156, Throughput: 203.274677 im/s
Epoch 25, Train Loss: 10332.4677734, Time: 24.5581s, Throughput: 203.273412 im/s
Epoch 26, Iter 39, Loss: 11182.9755859, Throughput: 204.105940 im/s
Epoch 26, Train Loss: 10351.5810547, Time: 24.4580s, Throughput: 204.104730 im/s
Epoch 27, Iter 39, Loss: 9853.7988281, Throughput: 203.349896 im/s
Epoch 27, Train Loss: 10275.2431641, Time: 24.5490s, Throughput: 203.348539 im/s
Epoch 28, Iter 39, Loss: 10436.3945312, Throughput: 203.325256 im/s
Epoch 28, Train Loss: 10191.2890625, Time: 24.5519s, Throughput: 203.323996 im/s
Epoch 29, Iter 39, Loss: 11162.0830078, Throughput: 203.926315 im/s
Epoch 29, Train Loss: 10185.2392578, Time: 24.4796s, Throughput: 203.925042 im/s
Epoch 30, Iter 39, Loss: 9818.8642578, Throughput: 203.519610 im/s
Epoch 30, Train Loss: 10080.0644531, Time: 24.5285s, Throughput: 203.518431 im/s
Epoch 31, Iter 39, Loss: 10527.1679688, Throughput: 203.747984 im/s
Epoch 31, Train Loss: 9989.5859375, Time: 24.5010s, Throughput: 203.746763 im/s
Epoch 32, Iter 39, Loss: 9832.1376953, Throughput: 203.745048 im/s
Epoch 32, Train Loss: 10026.8291016, Time: 24.5014s, Throughput: 203.743850 im/s
Epoch 33, Iter 39, Loss: 10394.4140625, Throughput: 203.738321 im/s
Epoch 33, Train Loss: 9964.6601562, Time: 24.5022s, Throughput: 203.737100 im/s
Epoch 34, Iter 39, Loss: 10659.6367188, Throughput: 203.193731 im/s
Epoch 34, Train Loss: 9887.6708984, Time: 24.5678s, Throughput: 203.192408 im/s
Epoch 35, Iter 39, Loss: 9613.4609375, Throughput: 200.756602 im/s
Epoch 35, Train Loss: 9873.7656250, Time: 24.8661s, Throughput: 200.755125 im/s
Epoch 36, Iter 39, Loss: 9643.2001953, Throughput: 203.830977 im/s
Epoch 36, Train Loss: 9846.1943359, Time: 24.4911s, Throughput: 203.829421 im/s
Epoch 37, Iter 39, Loss: 9870.4892578, Throughput: 203.335168 im/s
Epoch 37, Train Loss: 9847.1210938, Time: 24.5507s, Throughput: 203.333926 im/s
Epoch 38, Iter 39, Loss: 9394.2343750, Throughput: 203.329738 im/s
Epoch 38, Train Loss: 9768.7343750, Time: 24.5514s, Throughput: 203.328379 im/s
Epoch 39, Iter 39, Loss: 10133.0625000, Throughput: 202.748229 im/s
Epoch 39, Train Loss: 9726.9960938, Time: 24.6218s, Throughput: 202.747085 im/s
Epoch 40, Iter 39, Loss: 10043.8701172, Throughput: 203.309382 im/s
Epoch 40, Train Loss: 9723.2792969, Time: 24.5539s, Throughput: 203.307925 im/s
Epoch 41, Iter 39, Loss: 9422.4218750, Throughput: 203.031622 im/s
Epoch 41, Train Loss: 9645.5781250, Time: 24.5875s, Throughput: 203.030309 im/s
Epoch 42, Iter 39, Loss: 9422.5214844, Throughput: 203.509545 im/s
Epoch 42, Train Loss: 9629.9960938, Time: 24.5297s, Throughput: 203.508325 im/s
Epoch 43, Iter 39, Loss: 9675.9140625, Throughput: 203.024519 im/s
Epoch 43, Train Loss: 9531.3593750, Time: 24.5883s, Throughput: 203.023257 im/s
Epoch 44, Iter 39, Loss: 9327.8808594, Throughput: 202.928775 im/s
Epoch 44, Train Loss: 9573.3076172, Time: 24.5999s, Throughput: 202.927473 im/s
Epoch 45, Iter 39, Loss: 9591.9150391, Throughput: 203.041085 im/s
Epoch 45, Train Loss: 9529.9716797, Time: 24.5863s, Throughput: 203.039911 im/s
Epoch 46, Iter 39, Loss: 9990.0800781, Throughput: 201.634053 im/s
Epoch 46, Train Loss: 9459.3515625, Time: 24.7579s, Throughput: 201.632824 im/s
Epoch 47, Iter 39, Loss: 9578.2675781, Throughput: 202.301282 im/s
Epoch 47, Train Loss: 9529.9931641, Time: 24.6762s, Throughput: 202.299947 im/s
Epoch 48, Iter 39, Loss: 9748.2568359, Throughput: 203.189837 im/s
Epoch 48, Train Loss: 9463.8681641, Time: 24.5684s, Throughput: 203.188159 im/s
Epoch 49, Iter 39, Loss: 9788.1484375, Throughput: 202.662239 im/s
Epoch 49, Train Loss: 9391.1562500, Time: 24.6323s, Throughput: 202.660998 im/s
Epoch 50, Iter 39, Loss: 8921.9199219, Throughput: 203.414811 im/s
Epoch 50, Train Loss: 9346.7373047, Time: 24.5412s, Throughput: 203.413211 im/s

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
