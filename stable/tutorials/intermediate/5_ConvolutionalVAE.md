---
url: /stable/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23960.9375000, Throughput: 16.245079 im/s
Epoch 1, Train Loss: 39446.6718750, Time: 307.5731s, Throughput: 16.230286 im/s
Epoch 2, Iter 39, Loss: 17504.9902344, Throughput: 172.931501 im/s
Epoch 2, Train Loss: 20127.1953125, Time: 28.8671s, Throughput: 172.930387 im/s
Epoch 3, Iter 39, Loss: 15611.4570312, Throughput: 173.611969 im/s
Epoch 3, Train Loss: 16679.9414062, Time: 28.7540s, Throughput: 173.610695 im/s
Epoch 4, Iter 39, Loss: 15154.9707031, Throughput: 173.751495 im/s
Epoch 4, Train Loss: 15086.8925781, Time: 28.7309s, Throughput: 173.750448 im/s
Epoch 5, Iter 39, Loss: 13444.3808594, Throughput: 174.039651 im/s
Epoch 5, Train Loss: 14048.2919922, Time: 28.6833s, Throughput: 174.038662 im/s
Epoch 6, Iter 39, Loss: 13439.3261719, Throughput: 173.887788 im/s
Epoch 6, Train Loss: 13409.3105469, Time: 28.7083s, Throughput: 173.886867 im/s
Epoch 7, Iter 39, Loss: 12155.5019531, Throughput: 173.675257 im/s
Epoch 7, Train Loss: 12958.3564453, Time: 28.7434s, Throughput: 173.674362 im/s
Epoch 8, Iter 39, Loss: 11929.2587891, Throughput: 174.475426 im/s
Epoch 8, Train Loss: 12687.9521484, Time: 28.6116s, Throughput: 174.474523 im/s
Epoch 9, Iter 39, Loss: 11907.5732422, Throughput: 173.888503 im/s
Epoch 9, Train Loss: 12273.1748047, Time: 28.7082s, Throughput: 173.887479 im/s
Epoch 10, Iter 39, Loss: 11813.6464844, Throughput: 174.647446 im/s
Epoch 10, Train Loss: 11978.7285156, Time: 28.5835s, Throughput: 174.646494 im/s
Epoch 11, Iter 39, Loss: 11749.1894531, Throughput: 174.081425 im/s
Epoch 11, Train Loss: 11776.8916016, Time: 28.6764s, Throughput: 174.080422 im/s
Epoch 12, Iter 39, Loss: 11851.6044922, Throughput: 174.039141 im/s
Epoch 12, Train Loss: 11590.5156250, Time: 28.6834s, Throughput: 174.038231 im/s
Epoch 13, Iter 39, Loss: 11143.7666016, Throughput: 174.866292 im/s
Epoch 13, Train Loss: 11432.1523438, Time: 28.5486s, Throughput: 174.859799 im/s
Epoch 14, Iter 39, Loss: 11092.8330078, Throughput: 173.071952 im/s
Epoch 14, Train Loss: 11229.0302734, Time: 28.8436s, Throughput: 173.071088 im/s
Epoch 15, Iter 39, Loss: 10748.5283203, Throughput: 173.861412 im/s
Epoch 15, Train Loss: 11111.1162109, Time: 28.7127s, Throughput: 173.860432 im/s
Epoch 16, Iter 39, Loss: 10992.7500000, Throughput: 173.314486 im/s
Epoch 16, Train Loss: 10977.2656250, Time: 28.8033s, Throughput: 173.313565 im/s
Epoch 17, Iter 39, Loss: 11748.3125000, Throughput: 173.827419 im/s
Epoch 17, Train Loss: 10953.3388672, Time: 28.7183s, Throughput: 173.826420 im/s
Epoch 18, Iter 39, Loss: 11449.3437500, Throughput: 173.063691 im/s
Epoch 18, Train Loss: 10860.9804688, Time: 28.8450s, Throughput: 173.062731 im/s
Epoch 19, Iter 39, Loss: 10123.6171875, Throughput: 173.954544 im/s
Epoch 19, Train Loss: 10670.9394531, Time: 28.6973s, Throughput: 173.953507 im/s
Epoch 20, Iter 39, Loss: 10621.0703125, Throughput: 173.981881 im/s
Epoch 20, Train Loss: 10596.3662109, Time: 28.6928s, Throughput: 173.980929 im/s
Epoch 21, Iter 39, Loss: 10791.8867188, Throughput: 171.751756 im/s
Epoch 21, Train Loss: 10539.2031250, Time: 29.0654s, Throughput: 171.750781 im/s
Epoch 22, Iter 39, Loss: 9768.4589844, Throughput: 173.310478 im/s
Epoch 22, Train Loss: 10478.3857422, Time: 28.8040s, Throughput: 173.309546 im/s
Epoch 23, Iter 39, Loss: 10740.1406250, Throughput: 173.579841 im/s
Epoch 23, Train Loss: 10420.1669922, Time: 28.7593s, Throughput: 173.578828 im/s
Epoch 24, Iter 39, Loss: 9839.5742188, Throughput: 173.117099 im/s
Epoch 24, Train Loss: 10403.9394531, Time: 28.8361s, Throughput: 173.116090 im/s
Epoch 25, Iter 39, Loss: 10077.9755859, Throughput: 174.019339 im/s
Epoch 25, Train Loss: 10275.1494141, Time: 28.6866s, Throughput: 174.018349 im/s
Epoch 26, Iter 39, Loss: 9812.8007812, Throughput: 173.477609 im/s
Epoch 26, Train Loss: 10326.8906250, Time: 28.7762s, Throughput: 173.476626 im/s
Epoch 27, Iter 39, Loss: 10776.1542969, Throughput: 173.297795 im/s
Epoch 27, Train Loss: 10174.2724609, Time: 28.8061s, Throughput: 173.296604 im/s
Epoch 28, Iter 39, Loss: 10069.6640625, Throughput: 173.951919 im/s
Epoch 28, Train Loss: 10166.0781250, Time: 28.6978s, Throughput: 173.950907 im/s
Epoch 29, Iter 39, Loss: 10674.3242188, Throughput: 173.450810 im/s
Epoch 29, Train Loss: 10063.6445312, Time: 28.7806s, Throughput: 173.449876 im/s
Epoch 30, Iter 39, Loss: 10122.6289062, Throughput: 173.247437 im/s
Epoch 30, Train Loss: 9971.8212891, Time: 28.8144s, Throughput: 173.246494 im/s
Epoch 31, Iter 39, Loss: 9642.1875000, Throughput: 173.369338 im/s
Epoch 31, Train Loss: 9923.6855469, Time: 28.7942s, Throughput: 173.368405 im/s
Epoch 32, Iter 39, Loss: 10125.9648438, Throughput: 172.526682 im/s
Epoch 32, Train Loss: 9916.8417969, Time: 28.9349s, Throughput: 172.525416 im/s
Epoch 33, Iter 39, Loss: 9527.6591797, Throughput: 173.499719 im/s
Epoch 33, Train Loss: 9788.3037109, Time: 28.7726s, Throughput: 173.498579 im/s
Epoch 34, Iter 39, Loss: 10027.4345703, Throughput: 173.182135 im/s
Epoch 34, Train Loss: 9775.9218750, Time: 28.8253s, Throughput: 173.181138 im/s
Epoch 35, Iter 39, Loss: 9903.9443359, Throughput: 173.550266 im/s
Epoch 35, Train Loss: 9766.3222656, Time: 28.7642s, Throughput: 173.549318 im/s
Epoch 36, Iter 39, Loss: 10111.3574219, Throughput: 172.945875 im/s
Epoch 36, Train Loss: 9845.2792969, Time: 28.8647s, Throughput: 172.944850 im/s
Epoch 37, Iter 39, Loss: 10073.6904297, Throughput: 173.175408 im/s
Epoch 37, Train Loss: 9718.3457031, Time: 28.8264s, Throughput: 173.174524 im/s
Epoch 38, Iter 39, Loss: 10128.5742188, Throughput: 173.415663 im/s
Epoch 38, Train Loss: 9638.7724609, Time: 28.7865s, Throughput: 173.414657 im/s
Epoch 39, Iter 39, Loss: 9675.0517578, Throughput: 172.798300 im/s
Epoch 39, Train Loss: 9597.3271484, Time: 28.8893s, Throughput: 172.797325 im/s
Epoch 40, Iter 39, Loss: 9320.7734375, Throughput: 173.241196 im/s
Epoch 40, Train Loss: 9659.9541016, Time: 28.8155s, Throughput: 173.240174 im/s
Epoch 41, Iter 39, Loss: 9689.3300781, Throughput: 173.181560 im/s
Epoch 41, Train Loss: 9529.6083984, Time: 28.8254s, Throughput: 173.180657 im/s
Epoch 42, Iter 39, Loss: 9508.8457031, Throughput: 173.175359 im/s
Epoch 42, Train Loss: 9523.8603516, Time: 28.8264s, Throughput: 173.174300 im/s
Epoch 43, Iter 39, Loss: 9054.2636719, Throughput: 170.596619 im/s
Epoch 43, Train Loss: 9498.9980469, Time: 29.2622s, Throughput: 170.595652 im/s
Epoch 44, Iter 39, Loss: 9163.6845703, Throughput: 173.255241 im/s
Epoch 44, Train Loss: 9516.9580078, Time: 28.8131s, Throughput: 173.254298 im/s
Epoch 45, Iter 39, Loss: 9139.3027344, Throughput: 173.120431 im/s
Epoch 45, Train Loss: 9433.5742188, Time: 28.8356s, Throughput: 173.119399 im/s
Epoch 46, Iter 39, Loss: 9579.6699219, Throughput: 173.109121 im/s
Epoch 46, Train Loss: 9411.8388672, Time: 28.8375s, Throughput: 173.108136 im/s
Epoch 47, Iter 39, Loss: 9606.2011719, Throughput: 172.853711 im/s
Epoch 47, Train Loss: 9426.5361328, Time: 28.8801s, Throughput: 172.852773 im/s
Epoch 48, Iter 39, Loss: 9038.8281250, Throughput: 173.487333 im/s
Epoch 48, Train Loss: 9365.3593750, Time: 28.7746s, Throughput: 173.486182 im/s
Epoch 49, Iter 39, Loss: 9319.2451172, Throughput: 173.397984 im/s
Epoch 49, Train Loss: 9365.2119141, Time: 28.7894s, Throughput: 173.396941 im/s
Epoch 50, Iter 39, Loss: 10453.4277344, Throughput: 173.548130 im/s
Epoch 50, Train Loss: 9341.4941406, Time: 28.7645s, Throughput: 173.547272 im/s

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
