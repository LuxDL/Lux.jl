---
url: /previews/PR1807/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24037.9316406, Throughput: 13.302558 im/s
Epoch 1, Train Loss: 39782.5937500, Time: 375.5419s, Throughput: 13.292791 im/s
Epoch 2, Iter 39, Loss: 18198.1718750, Throughput: 169.173035 im/s
Epoch 2, Train Loss: 20166.8535156, Time: 29.5084s, Throughput: 169.171939 im/s
Epoch 3, Iter 39, Loss: 16263.5966797, Throughput: 169.144110 im/s
Epoch 3, Train Loss: 16673.9335938, Time: 29.5134s, Throughput: 169.143278 im/s
Epoch 4, Iter 39, Loss: 14918.4179688, Throughput: 169.228692 im/s
Epoch 4, Train Loss: 15057.6943359, Time: 29.4987s, Throughput: 169.227785 im/s
Epoch 5, Iter 39, Loss: 13813.0000000, Throughput: 169.262488 im/s
Epoch 5, Train Loss: 14105.2128906, Time: 29.4928s, Throughput: 169.261472 im/s
Epoch 6, Iter 39, Loss: 13039.3232422, Throughput: 169.288944 im/s
Epoch 6, Train Loss: 13443.4667969, Time: 29.4882s, Throughput: 169.288078 im/s
Epoch 7, Iter 39, Loss: 13564.6669922, Throughput: 168.698568 im/s
Epoch 7, Train Loss: 13125.3867188, Time: 29.5914s, Throughput: 168.697661 im/s
Epoch 8, Iter 39, Loss: 12728.5927734, Throughput: 169.069117 im/s
Epoch 8, Train Loss: 12655.3300781, Time: 29.5265s, Throughput: 169.068321 im/s
Epoch 9, Iter 39, Loss: 12003.4033203, Throughput: 168.650465 im/s
Epoch 9, Train Loss: 12212.1474609, Time: 29.5999s, Throughput: 168.649457 im/s
Epoch 10, Iter 39, Loss: 12318.0625000, Throughput: 168.930832 im/s
Epoch 10, Train Loss: 11987.7060547, Time: 29.5507s, Throughput: 168.929917 im/s
Epoch 11, Iter 39, Loss: 11996.6210938, Throughput: 165.769139 im/s
Epoch 11, Train Loss: 11858.5595703, Time: 30.1143s, Throughput: 165.768302 im/s
Epoch 12, Iter 39, Loss: 10601.5410156, Throughput: 169.312783 im/s
Epoch 12, Train Loss: 11587.7255859, Time: 29.4841s, Throughput: 169.311807 im/s
Epoch 13, Iter 39, Loss: 10943.8515625, Throughput: 169.389234 im/s
Epoch 13, Train Loss: 11527.1513672, Time: 29.4727s, Throughput: 169.377267 im/s
Epoch 14, Iter 39, Loss: 10877.8837891, Throughput: 169.196245 im/s
Epoch 14, Train Loss: 11410.5361328, Time: 29.5043s, Throughput: 169.195539 im/s
Epoch 15, Iter 39, Loss: 11521.3222656, Throughput: 168.970435 im/s
Epoch 15, Train Loss: 11204.4248047, Time: 29.5438s, Throughput: 168.969503 im/s
Epoch 16, Iter 39, Loss: 11520.3242188, Throughput: 169.420704 im/s
Epoch 16, Train Loss: 11061.0556641, Time: 29.4653s, Throughput: 169.419802 im/s
Epoch 17, Iter 39, Loss: 10648.2939453, Throughput: 169.673505 im/s
Epoch 17, Train Loss: 10932.3125000, Time: 29.4214s, Throughput: 169.672664 im/s
Epoch 18, Iter 39, Loss: 11097.5791016, Throughput: 169.659101 im/s
Epoch 18, Train Loss: 10922.8984375, Time: 29.4239s, Throughput: 169.658143 im/s
Epoch 19, Iter 39, Loss: 9693.2666016, Throughput: 169.432848 im/s
Epoch 19, Train Loss: 10822.5654297, Time: 29.4632s, Throughput: 169.431819 im/s
Epoch 20, Iter 39, Loss: 11262.2187500, Throughput: 168.798713 im/s
Epoch 20, Train Loss: 10695.0205078, Time: 29.5738s, Throughput: 168.797818 im/s
Epoch 21, Iter 39, Loss: 10415.7500000, Throughput: 168.996118 im/s
Epoch 21, Train Loss: 10689.7539062, Time: 29.5393s, Throughput: 168.995226 im/s
Epoch 22, Iter 39, Loss: 10248.0449219, Throughput: 168.502542 im/s
Epoch 22, Train Loss: 10571.3652344, Time: 29.6258s, Throughput: 168.501586 im/s
Epoch 23, Iter 39, Loss: 10454.7431641, Throughput: 169.378568 im/s
Epoch 23, Train Loss: 10467.3232422, Time: 29.4726s, Throughput: 169.377711 im/s
Epoch 24, Iter 39, Loss: 10572.6347656, Throughput: 168.780412 im/s
Epoch 24, Train Loss: 10421.0185547, Time: 29.5770s, Throughput: 168.779567 im/s
Epoch 25, Iter 39, Loss: 10900.2109375, Throughput: 169.566398 im/s
Epoch 25, Train Loss: 10390.9443359, Time: 29.4399s, Throughput: 169.565569 im/s
Epoch 26, Iter 39, Loss: 10092.7490234, Throughput: 168.533829 im/s
Epoch 26, Train Loss: 10346.9443359, Time: 29.6203s, Throughput: 168.532908 im/s
Epoch 27, Iter 39, Loss: 10912.5019531, Throughput: 169.379406 im/s
Epoch 27, Train Loss: 10283.4121094, Time: 29.4724s, Throughput: 169.378586 im/s
Epoch 28, Iter 39, Loss: 9995.9072266, Throughput: 169.435850 im/s
Epoch 28, Train Loss: 10168.3867188, Time: 29.4626s, Throughput: 169.435022 im/s
Epoch 29, Iter 39, Loss: 10410.9804688, Throughput: 169.901090 im/s
Epoch 29, Train Loss: 10123.9589844, Time: 29.3820s, Throughput: 169.900165 im/s
Epoch 30, Iter 39, Loss: 10150.3613281, Throughput: 168.956619 im/s
Epoch 30, Train Loss: 10029.7890625, Time: 29.5462s, Throughput: 168.955738 im/s
Epoch 31, Iter 39, Loss: 10106.4960938, Throughput: 168.617129 im/s
Epoch 31, Train Loss: 10031.3144531, Time: 29.6057s, Throughput: 168.616241 im/s
Epoch 32, Iter 39, Loss: 10402.2832031, Throughput: 169.597744 im/s
Epoch 32, Train Loss: 10093.6269531, Time: 29.4345s, Throughput: 169.596925 im/s
Epoch 33, Iter 39, Loss: 9455.6728516, Throughput: 170.396164 im/s
Epoch 33, Train Loss: 10030.4375000, Time: 29.2966s, Throughput: 170.395268 im/s
Epoch 34, Iter 39, Loss: 9988.1230469, Throughput: 170.335395 im/s
Epoch 34, Train Loss: 9927.2167969, Time: 29.3070s, Throughput: 170.334604 im/s
Epoch 35, Iter 39, Loss: 10458.3750000, Throughput: 170.341167 im/s
Epoch 35, Train Loss: 9881.7255859, Time: 29.3061s, Throughput: 170.340219 im/s
Epoch 36, Iter 39, Loss: 10587.6171875, Throughput: 169.842988 im/s
Epoch 36, Train Loss: 9833.7031250, Time: 29.3920s, Throughput: 169.842122 im/s
Epoch 37, Iter 39, Loss: 9855.1191406, Throughput: 169.972482 im/s
Epoch 37, Train Loss: 9841.6201172, Time: 29.3696s, Throughput: 169.971533 im/s
Epoch 38, Iter 39, Loss: 9765.1289062, Throughput: 169.795554 im/s
Epoch 38, Train Loss: 9858.0732422, Time: 29.4002s, Throughput: 169.794607 im/s
Epoch 39, Iter 39, Loss: 10137.4257812, Throughput: 170.005199 im/s
Epoch 39, Train Loss: 9755.4287109, Time: 29.3640s, Throughput: 170.004371 im/s
Epoch 40, Iter 39, Loss: 9611.5351562, Throughput: 169.665979 im/s
Epoch 40, Train Loss: 9771.6289062, Time: 29.4227s, Throughput: 169.664872 im/s
Epoch 41, Iter 39, Loss: 10247.2871094, Throughput: 169.953119 im/s
Epoch 41, Train Loss: 9689.3847656, Time: 29.3730s, Throughput: 169.952037 im/s
Epoch 42, Iter 39, Loss: 9698.4619141, Throughput: 169.440371 im/s
Epoch 42, Train Loss: 9636.0419922, Time: 29.4619s, Throughput: 169.439399 im/s
Epoch 43, Iter 39, Loss: 8895.6718750, Throughput: 170.401202 im/s
Epoch 43, Train Loss: 9727.8574219, Time: 29.2957s, Throughput: 170.400324 im/s
Epoch 44, Iter 39, Loss: 9345.2587891, Throughput: 170.179157 im/s
Epoch 44, Train Loss: 9641.3486328, Time: 29.3339s, Throughput: 170.178305 im/s
Epoch 45, Iter 39, Loss: 10236.2929688, Throughput: 170.206701 im/s
Epoch 45, Train Loss: 9612.3623047, Time: 29.3292s, Throughput: 170.205721 im/s
Epoch 46, Iter 39, Loss: 9713.1562500, Throughput: 169.774876 im/s
Epoch 46, Train Loss: 9570.8330078, Time: 29.4038s, Throughput: 169.774016 im/s
Epoch 47, Iter 39, Loss: 9743.7119141, Throughput: 170.383468 im/s
Epoch 47, Train Loss: 9513.8652344, Time: 29.2988s, Throughput: 170.382625 im/s
Epoch 48, Iter 39, Loss: 9742.8544922, Throughput: 169.623889 im/s
Epoch 48, Train Loss: 9485.0322266, Time: 29.4300s, Throughput: 169.622833 im/s
Epoch 49, Iter 39, Loss: 9703.3740234, Throughput: 170.169093 im/s
Epoch 49, Train Loss: 9482.0917969, Time: 29.3357s, Throughput: 170.168217 im/s
Epoch 50, Iter 39, Loss: 9481.3232422, Throughput: 167.826362 im/s
Epoch 50, Train Loss: 9522.7968750, Time: 29.7452s, Throughput: 167.825470 im/s

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
