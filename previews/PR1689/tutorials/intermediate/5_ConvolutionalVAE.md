---
url: /previews/PR1689/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23153.0957031, Throughput: 5.158778 im/s
Epoch 1, Train Loss: 39716.8789062, Time: 967.9599s, Throughput: 5.157239 im/s
Epoch 2, Iter 39, Loss: 17787.1113281, Throughput: 69.187964 im/s
Epoch 2, Train Loss: 20219.2070312, Time: 72.1515s, Throughput: 69.187775 im/s
Epoch 3, Iter 39, Loss: 15789.5087891, Throughput: 69.178629 im/s
Epoch 3, Train Loss: 16558.8476562, Time: 72.1612s, Throughput: 69.178488 im/s
Epoch 4, Iter 39, Loss: 14774.6962891, Throughput: 69.653663 im/s
Epoch 4, Train Loss: 14961.8681641, Time: 71.6690s, Throughput: 69.653501 im/s
Epoch 5, Iter 39, Loss: 14642.7187500, Throughput: 69.212526 im/s
Epoch 5, Train Loss: 14105.1474609, Time: 72.1258s, Throughput: 69.212359 im/s
Epoch 6, Iter 39, Loss: 13903.0878906, Throughput: 69.629309 im/s
Epoch 6, Train Loss: 13345.8232422, Time: 71.6941s, Throughput: 69.629170 im/s
Epoch 7, Iter 39, Loss: 12414.4472656, Throughput: 68.944239 im/s
Epoch 7, Train Loss: 12829.6875000, Time: 72.4065s, Throughput: 68.944085 im/s
Epoch 8, Iter 39, Loss: 12257.6230469, Throughput: 68.854388 im/s
Epoch 8, Train Loss: 12556.7910156, Time: 72.5010s, Throughput: 68.854218 im/s
Epoch 9, Iter 39, Loss: 12989.4902344, Throughput: 69.055877 im/s
Epoch 9, Train Loss: 12217.0019531, Time: 72.2894s, Throughput: 69.055743 im/s
Epoch 10, Iter 39, Loss: 11714.9238281, Throughput: 69.421920 im/s
Epoch 10, Train Loss: 11901.3750000, Time: 71.9083s, Throughput: 69.421760 im/s
Epoch 11, Iter 39, Loss: 11662.1601562, Throughput: 69.255619 im/s
Epoch 11, Train Loss: 11662.0761719, Time: 72.0810s, Throughput: 69.255461 im/s
Epoch 12, Iter 39, Loss: 11925.5595703, Throughput: 69.326401 im/s
Epoch 12, Train Loss: 11526.9228516, Time: 72.0073s, Throughput: 69.326256 im/s
Epoch 13, Iter 39, Loss: 10992.4316406, Throughput: 69.682742 im/s
Epoch 13, Train Loss: 11487.2636719, Time: 71.6391s, Throughput: 69.682607 im/s
Epoch 14, Iter 39, Loss: 11077.3232422, Throughput: 69.061645 im/s
Epoch 14, Train Loss: 11273.1162109, Time: 72.2834s, Throughput: 69.061513 im/s
Epoch 15, Iter 39, Loss: 11215.7421875, Throughput: 69.404019 im/s
Epoch 15, Train Loss: 11043.2031250, Time: 71.9268s, Throughput: 69.403855 im/s
Epoch 16, Iter 39, Loss: 10389.1513672, Throughput: 69.406663 im/s
Epoch 16, Train Loss: 10961.2753906, Time: 71.9241s, Throughput: 69.406501 im/s
Epoch 17, Iter 39, Loss: 11620.2695312, Throughput: 68.817533 im/s
Epoch 17, Train Loss: 10847.9804688, Time: 72.5398s, Throughput: 68.817375 im/s
Epoch 18, Iter 39, Loss: 10853.4560547, Throughput: 69.595325 im/s
Epoch 18, Train Loss: 10722.1181641, Time: 71.7291s, Throughput: 69.595174 im/s
Epoch 19, Iter 39, Loss: 10233.9785156, Throughput: 68.863018 im/s
Epoch 19, Train Loss: 10622.5859375, Time: 72.4919s, Throughput: 68.862870 im/s
Epoch 20, Iter 39, Loss: 11731.3154297, Throughput: 69.027953 im/s
Epoch 20, Train Loss: 10651.7958984, Time: 72.3187s, Throughput: 69.027797 im/s
Epoch 21, Iter 39, Loss: 10204.4257812, Throughput: 69.295213 im/s
Epoch 21, Train Loss: 10495.7802734, Time: 72.0398s, Throughput: 69.295076 im/s
Epoch 22, Iter 39, Loss: 10309.1230469, Throughput: 69.247399 im/s
Epoch 22, Train Loss: 10489.3681641, Time: 72.0895s, Throughput: 69.247247 im/s
Epoch 23, Iter 39, Loss: 10554.0869141, Throughput: 68.974865 im/s
Epoch 23, Train Loss: 10329.2724609, Time: 72.3744s, Throughput: 68.974711 im/s
Epoch 24, Iter 39, Loss: 10896.2929688, Throughput: 69.195881 im/s
Epoch 24, Train Loss: 10292.3847656, Time: 72.1432s, Throughput: 69.195739 im/s
Epoch 25, Iter 39, Loss: 11234.0742188, Throughput: 69.184873 im/s
Epoch 25, Train Loss: 10219.9306641, Time: 72.1547s, Throughput: 69.184728 im/s
Epoch 26, Iter 39, Loss: 10378.5839844, Throughput: 69.000008 im/s
Epoch 26, Train Loss: 10168.9707031, Time: 72.3480s, Throughput: 68.999863 im/s
Epoch 27, Iter 39, Loss: 10134.2167969, Throughput: 68.880483 im/s
Epoch 27, Train Loss: 10125.5849609, Time: 72.4735s, Throughput: 68.880327 im/s
Epoch 28, Iter 39, Loss: 9806.5400391, Throughput: 69.188351 im/s
Epoch 28, Train Loss: 10034.8457031, Time: 72.1510s, Throughput: 69.188192 im/s
Epoch 29, Iter 39, Loss: 9869.7910156, Throughput: 68.914894 im/s
Epoch 29, Train Loss: 10064.7080078, Time: 72.4373s, Throughput: 68.914741 im/s
Epoch 30, Iter 39, Loss: 9869.6015625, Throughput: 69.154312 im/s
Epoch 30, Train Loss: 9995.3574219, Time: 72.1866s, Throughput: 69.154155 im/s
Epoch 31, Iter 39, Loss: 10210.3203125, Throughput: 68.841446 im/s
Epoch 31, Train Loss: 9865.1591797, Time: 72.5146s, Throughput: 68.841288 im/s
Epoch 32, Iter 39, Loss: 9616.7597656, Throughput: 68.944299 im/s
Epoch 32, Train Loss: 9834.6181641, Time: 72.4064s, Throughput: 68.944147 im/s
Epoch 33, Iter 39, Loss: 10693.8339844, Throughput: 69.163204 im/s
Epoch 33, Train Loss: 9857.4482422, Time: 72.1773s, Throughput: 69.163044 im/s
Epoch 34, Iter 39, Loss: 9741.6074219, Throughput: 69.009683 im/s
Epoch 34, Train Loss: 9799.0429688, Time: 72.3378s, Throughput: 69.009517 im/s
Epoch 35, Iter 39, Loss: 9488.0253906, Throughput: 69.166703 im/s
Epoch 35, Train Loss: 9765.4169922, Time: 72.1736s, Throughput: 69.166550 im/s
Epoch 36, Iter 39, Loss: 10130.9550781, Throughput: 69.278096 im/s
Epoch 36, Train Loss: 9676.9707031, Time: 72.0576s, Throughput: 69.277956 im/s
Epoch 37, Iter 39, Loss: 10594.6308594, Throughput: 68.931629 im/s
Epoch 37, Train Loss: 9654.3603516, Time: 72.4198s, Throughput: 68.931459 im/s
Epoch 38, Iter 39, Loss: 9291.8261719, Throughput: 69.744283 im/s
Epoch 38, Train Loss: 9633.4287109, Time: 71.5759s, Throughput: 69.744140 im/s
Epoch 39, Iter 39, Loss: 9730.8164062, Throughput: 69.548777 im/s
Epoch 39, Train Loss: 9637.4541016, Time: 71.7771s, Throughput: 69.548632 im/s
Epoch 40, Iter 39, Loss: 9472.3906250, Throughput: 69.472898 im/s
Epoch 40, Train Loss: 9473.6318359, Time: 71.8555s, Throughput: 69.472745 im/s
Epoch 41, Iter 39, Loss: 10199.6503906, Throughput: 68.743799 im/s
Epoch 41, Train Loss: 9543.7744141, Time: 72.6176s, Throughput: 68.743647 im/s
Epoch 42, Iter 39, Loss: 10002.5957031, Throughput: 69.112832 im/s
Epoch 42, Train Loss: 9520.4667969, Time: 72.2299s, Throughput: 69.112678 im/s
Epoch 43, Iter 39, Loss: 9765.5156250, Throughput: 69.103457 im/s
Epoch 43, Train Loss: 9513.2070312, Time: 72.2397s, Throughput: 69.103292 im/s
Epoch 44, Iter 39, Loss: 9342.9814453, Throughput: 69.504904 im/s
Epoch 44, Train Loss: 9395.2597656, Time: 71.8224s, Throughput: 69.504750 im/s
Epoch 45, Iter 39, Loss: 9283.9726562, Throughput: 69.140345 im/s
Epoch 45, Train Loss: 9423.1503906, Time: 72.2011s, Throughput: 69.140205 im/s
Epoch 46, Iter 39, Loss: 9680.7343750, Throughput: 69.030285 im/s
Epoch 46, Train Loss: 9414.7529297, Time: 72.3162s, Throughput: 69.030129 im/s
Epoch 47, Iter 39, Loss: 10057.1777344, Throughput: 68.888698 im/s
Epoch 47, Train Loss: 9403.4042969, Time: 72.4649s, Throughput: 68.888555 im/s
Epoch 48, Iter 39, Loss: 9357.1503906, Throughput: 69.034477 im/s
Epoch 48, Train Loss: 9348.9511719, Time: 72.3118s, Throughput: 69.034329 im/s
Epoch 49, Iter 39, Loss: 9551.6015625, Throughput: 68.836792 im/s
Epoch 49, Train Loss: 9352.9472656, Time: 72.5195s, Throughput: 68.836626 im/s
Epoch 50, Iter 39, Loss: 9388.6855469, Throughput: 69.005845 im/s
Epoch 50, Train Loss: 9295.9130859, Time: 72.3419s, Throughput: 69.005681 im/s

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
