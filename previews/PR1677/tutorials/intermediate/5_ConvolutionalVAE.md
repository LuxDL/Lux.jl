---
url: /previews/PR1677/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24083.5585938, Throughput: 5.208745 im/s
Epoch 1, Train Loss: 39758.1054688, Time: 958.6775s, Throughput: 5.207173 im/s
Epoch 2, Iter 39, Loss: 17704.9785156, Throughput: 72.540588 im/s
Epoch 2, Train Loss: 20059.3261719, Time: 68.8168s, Throughput: 72.540394 im/s
Epoch 3, Iter 39, Loss: 16458.1796875, Throughput: 71.799428 im/s
Epoch 3, Train Loss: 16420.7636719, Time: 69.5272s, Throughput: 71.799247 im/s
Epoch 4, Iter 39, Loss: 14909.4433594, Throughput: 72.188445 im/s
Epoch 4, Train Loss: 14955.5429688, Time: 69.1525s, Throughput: 72.188280 im/s
Epoch 5, Iter 39, Loss: 12601.1611328, Throughput: 72.338343 im/s
Epoch 5, Train Loss: 14006.6474609, Time: 69.0092s, Throughput: 72.338197 im/s
Epoch 6, Iter 39, Loss: 12714.6435547, Throughput: 72.360319 im/s
Epoch 6, Train Loss: 13364.7802734, Time: 68.9882s, Throughput: 72.360153 im/s
Epoch 7, Iter 39, Loss: 12985.5312500, Throughput: 72.839414 im/s
Epoch 7, Train Loss: 12895.2509766, Time: 68.5345s, Throughput: 72.839246 im/s
Epoch 8, Iter 39, Loss: 12724.4580078, Throughput: 72.528338 im/s
Epoch 8, Train Loss: 12435.3417969, Time: 68.8285s, Throughput: 72.528143 im/s
Epoch 9, Iter 39, Loss: 11971.6328125, Throughput: 73.182531 im/s
Epoch 9, Train Loss: 12254.0615234, Time: 68.2132s, Throughput: 73.182346 im/s
Epoch 10, Iter 39, Loss: 12280.3554688, Throughput: 72.627729 im/s
Epoch 10, Train Loss: 11979.8671875, Time: 68.7342s, Throughput: 72.627575 im/s
Epoch 11, Iter 39, Loss: 11567.9130859, Throughput: 72.887515 im/s
Epoch 11, Train Loss: 11676.0126953, Time: 68.4892s, Throughput: 72.887360 im/s
Epoch 12, Iter 39, Loss: 11387.6250000, Throughput: 72.228775 im/s
Epoch 12, Train Loss: 11566.0107422, Time: 69.1139s, Throughput: 72.228590 im/s
Epoch 13, Iter 39, Loss: 10810.8066406, Throughput: 73.004536 im/s
Epoch 13, Train Loss: 11355.2792969, Time: 68.3795s, Throughput: 73.004367 im/s
Epoch 14, Iter 39, Loss: 11380.8398438, Throughput: 72.622422 im/s
Epoch 14, Train Loss: 11194.2880859, Time: 68.7392s, Throughput: 72.622297 im/s
Epoch 15, Iter 39, Loss: 11065.3037109, Throughput: 72.548954 im/s
Epoch 15, Train Loss: 11173.1093750, Time: 68.8089s, Throughput: 72.548780 im/s
Epoch 16, Iter 39, Loss: 10793.8046875, Throughput: 72.776731 im/s
Epoch 16, Train Loss: 11098.8544922, Time: 68.5935s, Throughput: 72.776535 im/s
Epoch 17, Iter 39, Loss: 10709.6435547, Throughput: 72.530820 im/s
Epoch 17, Train Loss: 10924.1611328, Time: 68.8261s, Throughput: 72.530639 im/s
Epoch 18, Iter 39, Loss: 10490.4335938, Throughput: 71.867063 im/s
Epoch 18, Train Loss: 10750.4951172, Time: 69.4617s, Throughput: 71.866920 im/s
Epoch 19, Iter 39, Loss: 10668.8154297, Throughput: 71.871681 im/s
Epoch 19, Train Loss: 10713.2705078, Time: 69.4573s, Throughput: 71.871520 im/s
Epoch 20, Iter 39, Loss: 10827.0937500, Throughput: 72.056945 im/s
Epoch 20, Train Loss: 10612.9335938, Time: 69.2787s, Throughput: 72.056778 im/s
Epoch 21, Iter 39, Loss: 10487.4375000, Throughput: 72.494159 im/s
Epoch 21, Train Loss: 10490.9746094, Time: 68.8609s, Throughput: 72.493972 im/s
Epoch 22, Iter 39, Loss: 11635.8671875, Throughput: 72.745385 im/s
Epoch 22, Train Loss: 10461.5078125, Time: 68.6231s, Throughput: 72.745223 im/s
Epoch 23, Iter 39, Loss: 10164.1425781, Throughput: 72.533435 im/s
Epoch 23, Train Loss: 10371.9619141, Time: 68.8236s, Throughput: 72.533257 im/s
Epoch 24, Iter 39, Loss: 10358.7783203, Throughput: 72.313237 im/s
Epoch 24, Train Loss: 10435.0683594, Time: 69.0332s, Throughput: 72.313073 im/s
Epoch 25, Iter 39, Loss: 9964.0937500, Throughput: 72.225493 im/s
Epoch 25, Train Loss: 10265.5595703, Time: 69.1170s, Throughput: 72.225310 im/s
Epoch 26, Iter 39, Loss: 9963.7255859, Throughput: 72.231728 im/s
Epoch 26, Train Loss: 10189.9746094, Time: 69.1111s, Throughput: 72.231570 im/s
Epoch 27, Iter 39, Loss: 10638.8369141, Throughput: 71.798135 im/s
Epoch 27, Train Loss: 10217.1103516, Time: 69.5284s, Throughput: 71.797972 im/s
Epoch 28, Iter 39, Loss: 10618.6875000, Throughput: 72.701005 im/s
Epoch 28, Train Loss: 10147.5410156, Time: 68.6650s, Throughput: 72.700825 im/s
Epoch 29, Iter 39, Loss: 9827.7958984, Throughput: 72.247303 im/s
Epoch 29, Train Loss: 9992.2763672, Time: 69.0962s, Throughput: 72.247108 im/s
Epoch 30, Iter 39, Loss: 10206.3925781, Throughput: 72.046117 im/s
Epoch 30, Train Loss: 9989.3232422, Time: 69.2891s, Throughput: 72.045928 im/s
Epoch 31, Iter 39, Loss: 10168.6435547, Throughput: 72.468448 im/s
Epoch 31, Train Loss: 10033.1386719, Time: 68.8853s, Throughput: 72.468256 im/s
Epoch 32, Iter 39, Loss: 10627.9199219, Throughput: 72.577409 im/s
Epoch 32, Train Loss: 9904.0263672, Time: 68.7819s, Throughput: 72.577229 im/s
Epoch 33, Iter 39, Loss: 9994.5312500, Throughput: 72.517582 im/s
Epoch 33, Train Loss: 9916.0839844, Time: 68.8386s, Throughput: 72.517421 im/s
Epoch 34, Iter 39, Loss: 10207.1259766, Throughput: 72.934263 im/s
Epoch 34, Train Loss: 9876.6406250, Time: 68.4453s, Throughput: 72.934106 im/s
Epoch 35, Iter 39, Loss: 9401.8261719, Throughput: 72.351143 im/s
Epoch 35, Train Loss: 9772.5117188, Time: 68.9970s, Throughput: 72.350979 im/s
Epoch 36, Iter 39, Loss: 9791.5595703, Throughput: 71.608023 im/s
Epoch 36, Train Loss: 9707.1943359, Time: 69.7130s, Throughput: 71.607855 im/s
Epoch 37, Iter 39, Loss: 9628.0742188, Throughput: 72.470713 im/s
Epoch 37, Train Loss: 9744.2236328, Time: 68.8832s, Throughput: 72.470530 im/s
Epoch 38, Iter 39, Loss: 10238.7460938, Throughput: 72.575986 im/s
Epoch 38, Train Loss: 9686.6250000, Time: 68.7833s, Throughput: 72.575797 im/s
Epoch 39, Iter 39, Loss: 9065.8906250, Throughput: 72.297850 im/s
Epoch 39, Train Loss: 9615.2031250, Time: 69.0479s, Throughput: 72.297659 im/s
Epoch 40, Iter 39, Loss: 9368.8808594, Throughput: 72.815063 im/s
Epoch 40, Train Loss: 9562.9208984, Time: 68.5574s, Throughput: 72.814906 im/s
Epoch 41, Iter 39, Loss: 10691.0185547, Throughput: 72.224726 im/s
Epoch 41, Train Loss: 9609.9765625, Time: 69.1178s, Throughput: 72.224545 im/s
Epoch 42, Iter 39, Loss: 9411.8671875, Throughput: 72.690664 im/s
Epoch 42, Train Loss: 9723.4990234, Time: 68.6747s, Throughput: 72.690486 im/s
Epoch 43, Iter 39, Loss: 10104.4960938, Throughput: 72.306006 im/s
Epoch 43, Train Loss: 9534.8232422, Time: 69.0401s, Throughput: 72.305831 im/s
Epoch 44, Iter 39, Loss: 9607.7529297, Throughput: 72.357472 im/s
Epoch 44, Train Loss: 9453.1728516, Time: 68.9910s, Throughput: 72.357281 im/s
Epoch 45, Iter 39, Loss: 9246.4433594, Throughput: 72.141542 im/s
Epoch 45, Train Loss: 9487.9042969, Time: 69.1975s, Throughput: 72.141353 im/s
Epoch 46, Iter 39, Loss: 9382.2929688, Throughput: 71.668241 im/s
Epoch 46, Train Loss: 9497.3994141, Time: 69.6545s, Throughput: 71.668062 im/s
Epoch 47, Iter 39, Loss: 10047.2382812, Throughput: 71.545661 im/s
Epoch 47, Train Loss: 9454.7460938, Time: 69.7738s, Throughput: 71.545497 im/s
Epoch 48, Iter 39, Loss: 9394.8730469, Throughput: 71.843705 im/s
Epoch 48, Train Loss: 9359.3066406, Time: 69.4843s, Throughput: 71.843548 im/s
Epoch 49, Iter 39, Loss: 9074.8662109, Throughput: 72.673283 im/s
Epoch 49, Train Loss: 9386.6425781, Time: 68.6912s, Throughput: 72.673082 im/s
Epoch 50, Iter 39, Loss: 9380.8417969, Throughput: 72.700139 im/s
Epoch 50, Train Loss: 9298.2080078, Time: 68.6658s, Throughput: 72.699952 im/s

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
