---
url: /previews/PR1822/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23700.3359375, Throughput: 19.797715 im/s
Epoch 1, Train Loss: 39441.6406250, Time: 252.3545s, Throughput: 19.781700 im/s
Epoch 2, Iter 39, Loss: 17052.0722656, Throughput: 120.114200 im/s
Epoch 2, Train Loss: 19906.9375000, Time: 41.5607s, Throughput: 120.113611 im/s
Epoch 3, Iter 39, Loss: 15888.0693359, Throughput: 120.293145 im/s
Epoch 3, Train Loss: 16611.3789062, Time: 41.4988s, Throughput: 120.292589 im/s
Epoch 4, Iter 39, Loss: 14544.8007812, Throughput: 120.662708 im/s
Epoch 4, Train Loss: 15094.5673828, Time: 41.3717s, Throughput: 120.662181 im/s
Epoch 5, Iter 39, Loss: 12780.3125000, Throughput: 120.492426 im/s
Epoch 5, Train Loss: 14171.4228516, Time: 41.4302s, Throughput: 120.491824 im/s
Epoch 6, Iter 39, Loss: 13650.1445312, Throughput: 120.075867 im/s
Epoch 6, Train Loss: 13499.6357422, Time: 41.5739s, Throughput: 120.075336 im/s
Epoch 7, Iter 39, Loss: 12903.3173828, Throughput: 120.817792 im/s
Epoch 7, Train Loss: 12861.2001953, Time: 41.3186s, Throughput: 120.817154 im/s
Epoch 8, Iter 39, Loss: 12189.7597656, Throughput: 120.513916 im/s
Epoch 8, Train Loss: 12518.7861328, Time: 41.4228s, Throughput: 120.513430 im/s
Epoch 9, Iter 39, Loss: 11813.7021484, Throughput: 120.274254 im/s
Epoch 9, Train Loss: 12264.9707031, Time: 41.5053s, Throughput: 120.273718 im/s
Epoch 10, Iter 39, Loss: 11796.2050781, Throughput: 120.424892 im/s
Epoch 10, Train Loss: 11965.8535156, Time: 41.4534s, Throughput: 120.424296 im/s
Epoch 11, Iter 39, Loss: 11938.5546875, Throughput: 121.169737 im/s
Epoch 11, Train Loss: 11737.9248047, Time: 41.1986s, Throughput: 121.169157 im/s
Epoch 12, Iter 39, Loss: 11238.1181641, Throughput: 120.797454 im/s
Epoch 12, Train Loss: 11549.2343750, Time: 41.3256s, Throughput: 120.796870 im/s
Epoch 13, Iter 39, Loss: 11530.2431641, Throughput: 120.436475 im/s
Epoch 13, Train Loss: 11338.3564453, Time: 41.4494s, Throughput: 120.435981 im/s
Epoch 14, Iter 39, Loss: 10911.6132812, Throughput: 121.054110 im/s
Epoch 14, Train Loss: 11245.4121094, Time: 41.2379s, Throughput: 121.053623 im/s
Epoch 15, Iter 39, Loss: 10961.5498047, Throughput: 120.955273 im/s
Epoch 15, Train Loss: 11225.4726562, Time: 41.2717s, Throughput: 120.954646 im/s
Epoch 16, Iter 39, Loss: 11111.2832031, Throughput: 121.113034 im/s
Epoch 16, Train Loss: 11071.0048828, Time: 41.2179s, Throughput: 121.112456 im/s
Epoch 17, Iter 39, Loss: 11151.9326172, Throughput: 120.612279 im/s
Epoch 17, Train Loss: 10858.0664062, Time: 41.3890s, Throughput: 120.611728 im/s
Epoch 18, Iter 39, Loss: 10338.1523438, Throughput: 120.881923 im/s
Epoch 18, Train Loss: 10820.7226562, Time: 41.2967s, Throughput: 120.881347 im/s
Epoch 19, Iter 39, Loss: 11262.7558594, Throughput: 121.044243 im/s
Epoch 19, Train Loss: 10683.2080078, Time: 41.2413s, Throughput: 121.043676 im/s
Epoch 20, Iter 39, Loss: 10837.0761719, Throughput: 120.832460 im/s
Epoch 20, Train Loss: 10554.9316406, Time: 41.3136s, Throughput: 120.831890 im/s
Epoch 21, Iter 39, Loss: 11039.4121094, Throughput: 120.899835 im/s
Epoch 21, Train Loss: 10536.8232422, Time: 41.2906s, Throughput: 120.899319 im/s
Epoch 22, Iter 39, Loss: 10302.1562500, Throughput: 119.589652 im/s
Epoch 22, Train Loss: 10459.0703125, Time: 41.7429s, Throughput: 119.589097 im/s
Epoch 23, Iter 39, Loss: 11046.0009766, Throughput: 122.097473 im/s
Epoch 23, Train Loss: 10342.2861328, Time: 40.8855s, Throughput: 122.096977 im/s
Epoch 24, Iter 39, Loss: 11287.8652344, Throughput: 120.945016 im/s
Epoch 24, Train Loss: 10342.6416016, Time: 41.2752s, Throughput: 120.944360 im/s
Epoch 25, Iter 39, Loss: 9975.8632812, Throughput: 121.896636 im/s
Epoch 25, Train Loss: 10301.3750000, Time: 40.9550s, Throughput: 121.889757 im/s
Epoch 26, Iter 39, Loss: 9819.4199219, Throughput: 121.076900 im/s
Epoch 26, Train Loss: 10229.0058594, Time: 41.2302s, Throughput: 121.076381 im/s
Epoch 27, Iter 39, Loss: 10092.2080078, Throughput: 121.422739 im/s
Epoch 27, Train Loss: 10138.2128906, Time: 41.1127s, Throughput: 121.422225 im/s
Epoch 28, Iter 39, Loss: 9729.2421875, Throughput: 121.626414 im/s
Epoch 28, Train Loss: 10062.6328125, Time: 41.0439s, Throughput: 121.625831 im/s
Epoch 29, Iter 39, Loss: 10039.6523438, Throughput: 121.367621 im/s
Epoch 29, Train Loss: 10081.0253906, Time: 41.1314s, Throughput: 121.367084 im/s
Epoch 30, Iter 39, Loss: 10033.9414062, Throughput: 121.040084 im/s
Epoch 30, Train Loss: 9936.3593750, Time: 41.2437s, Throughput: 121.036741 im/s
Epoch 31, Iter 39, Loss: 10147.7929688, Throughput: 120.731663 im/s
Epoch 31, Train Loss: 9924.9453125, Time: 41.3481s, Throughput: 120.731137 im/s
Epoch 32, Iter 39, Loss: 9945.3583984, Throughput: 120.759149 im/s
Epoch 32, Train Loss: 9876.3994141, Time: 41.3387s, Throughput: 120.758643 im/s
Epoch 33, Iter 39, Loss: 9246.5595703, Throughput: 121.433192 im/s
Epoch 33, Train Loss: 9830.9697266, Time: 41.1101s, Throughput: 121.430028 im/s
Epoch 34, Iter 39, Loss: 9803.3632812, Throughput: 120.889137 im/s
Epoch 34, Train Loss: 9776.1738281, Time: 41.2942s, Throughput: 120.888607 im/s
Epoch 35, Iter 39, Loss: 10421.7470703, Throughput: 121.472672 im/s
Epoch 35, Train Loss: 9809.8105469, Time: 41.0958s, Throughput: 121.472205 im/s
Epoch 36, Iter 39, Loss: 9411.9472656, Throughput: 121.006598 im/s
Epoch 36, Train Loss: 9710.2744141, Time: 41.2541s, Throughput: 121.006123 im/s
Epoch 37, Iter 39, Loss: 9697.3867188, Throughput: 121.640954 im/s
Epoch 37, Train Loss: 9796.6347656, Time: 41.0390s, Throughput: 121.640447 im/s
Epoch 38, Iter 39, Loss: 9548.8984375, Throughput: 121.150901 im/s
Epoch 38, Train Loss: 9668.2753906, Time: 41.2050s, Throughput: 121.150372 im/s
Epoch 39, Iter 39, Loss: 9590.2890625, Throughput: 122.260236 im/s
Epoch 39, Train Loss: 9598.0976562, Time: 40.8311s, Throughput: 122.259808 im/s
Epoch 40, Iter 39, Loss: 10522.7265625, Throughput: 121.960793 im/s
Epoch 40, Train Loss: 9574.3652344, Time: 40.9313s, Throughput: 121.960313 im/s
Epoch 41, Iter 39, Loss: 9662.6748047, Throughput: 121.583721 im/s
Epoch 41, Train Loss: 9523.9531250, Time: 41.0583s, Throughput: 121.583132 im/s
Epoch 42, Iter 39, Loss: 9729.8906250, Throughput: 122.197750 im/s
Epoch 42, Train Loss: 9536.4902344, Time: 40.8520s, Throughput: 122.197245 im/s
Epoch 43, Iter 39, Loss: 8665.1757812, Throughput: 121.640693 im/s
Epoch 43, Train Loss: 9515.6533203, Time: 41.0391s, Throughput: 121.640166 im/s
Epoch 44, Iter 39, Loss: 9674.8173828, Throughput: 121.682871 im/s
Epoch 44, Train Loss: 9442.9863281, Time: 41.0249s, Throughput: 121.682337 im/s
Epoch 45, Iter 39, Loss: 10141.1132812, Throughput: 120.560604 im/s
Epoch 45, Train Loss: 9408.3593750, Time: 41.4067s, Throughput: 120.560120 im/s
Epoch 46, Iter 39, Loss: 9025.3964844, Throughput: 121.235477 im/s
Epoch 46, Train Loss: 9399.0878906, Time: 41.1763s, Throughput: 121.234871 im/s
Epoch 47, Iter 39, Loss: 9553.6093750, Throughput: 121.401511 im/s
Epoch 47, Train Loss: 9352.2373047, Time: 41.1199s, Throughput: 121.400944 im/s
Epoch 48, Iter 39, Loss: 9079.1699219, Throughput: 122.077746 im/s
Epoch 48, Train Loss: 9353.9970703, Time: 40.8922s, Throughput: 122.077062 im/s
Epoch 49, Iter 39, Loss: 9011.0683594, Throughput: 121.145821 im/s
Epoch 49, Train Loss: 9383.6005859, Time: 41.2067s, Throughput: 121.145301 im/s
Epoch 50, Iter 39, Loss: 9117.6152344, Throughput: 121.597937 im/s
Epoch 50, Train Loss: 9327.8798828, Time: 41.0535s, Throughput: 121.597398 im/s

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
