---
url: /previews/PR1832/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24302.0937500, Throughput: 21.130062 im/s
Epoch 1, Train Loss: 40349.0820312, Time: 236.4550s, Throughput: 21.111837 im/s
Epoch 2, Iter 39, Loss: 18036.6015625, Throughput: 187.133311 im/s
Epoch 2, Train Loss: 20545.2988281, Time: 26.6764s, Throughput: 187.131950 im/s
Epoch 3, Iter 39, Loss: 16003.2558594, Throughput: 187.066594 im/s
Epoch 3, Train Loss: 17011.0703125, Time: 26.6859s, Throughput: 187.065205 im/s
Epoch 4, Iter 39, Loss: 15371.6367188, Throughput: 187.983382 im/s
Epoch 4, Train Loss: 15614.6953125, Time: 26.5557s, Throughput: 187.982291 im/s
Epoch 5, Iter 39, Loss: 13289.1835938, Throughput: 187.569502 im/s
Epoch 5, Train Loss: 14588.9423828, Time: 26.6143s, Throughput: 187.568310 im/s
Epoch 6, Iter 39, Loss: 13716.9628906, Throughput: 187.586649 im/s
Epoch 6, Train Loss: 13988.9482422, Time: 26.6119s, Throughput: 187.585464 im/s
Epoch 7, Iter 39, Loss: 13053.1015625, Throughput: 187.711161 im/s
Epoch 7, Train Loss: 13451.3095703, Time: 26.5942s, Throughput: 187.709877 im/s
Epoch 8, Iter 39, Loss: 12922.6113281, Throughput: 187.534988 im/s
Epoch 8, Train Loss: 12999.3896484, Time: 26.6192s, Throughput: 187.533852 im/s
Epoch 9, Iter 39, Loss: 12070.3759766, Throughput: 187.421580 im/s
Epoch 9, Train Loss: 12772.9003906, Time: 26.6353s, Throughput: 187.420487 im/s
Epoch 10, Iter 39, Loss: 13243.5712891, Throughput: 187.739863 im/s
Epoch 10, Train Loss: 12417.8642578, Time: 26.5901s, Throughput: 187.738840 im/s
Epoch 11, Iter 39, Loss: 12354.7099609, Throughput: 187.418954 im/s
Epoch 11, Train Loss: 12344.9482422, Time: 26.6357s, Throughput: 187.417857 im/s
Epoch 12, Iter 39, Loss: 13135.7597656, Throughput: 187.531916 im/s
Epoch 12, Train Loss: 12114.2714844, Time: 26.6196s, Throughput: 187.530711 im/s
Epoch 13, Iter 39, Loss: 11044.0273438, Throughput: 187.666380 im/s
Epoch 13, Train Loss: 11914.4257812, Time: 26.6006s, Throughput: 187.665194 im/s
Epoch 14, Iter 39, Loss: 11576.6699219, Throughput: 187.696417 im/s
Epoch 14, Train Loss: 11690.1542969, Time: 26.5963s, Throughput: 187.695331 im/s
Epoch 15, Iter 39, Loss: 11748.5214844, Throughput: 187.764636 im/s
Epoch 15, Train Loss: 11653.3593750, Time: 26.5866s, Throughput: 187.763513 im/s
Epoch 16, Iter 39, Loss: 11322.2519531, Throughput: 187.366141 im/s
Epoch 16, Train Loss: 11572.3779297, Time: 26.6432s, Throughput: 187.364932 im/s
Epoch 17, Iter 39, Loss: 11916.7441406, Throughput: 188.118854 im/s
Epoch 17, Train Loss: 11471.7265625, Time: 26.5366s, Throughput: 188.117735 im/s
Epoch 18, Iter 39, Loss: 12116.5322266, Throughput: 187.756431 im/s
Epoch 18, Train Loss: 11276.7285156, Time: 26.5878s, Throughput: 187.755321 im/s
Epoch 19, Iter 39, Loss: 11784.9560547, Throughput: 185.427866 im/s
Epoch 19, Train Loss: 11319.3955078, Time: 26.9217s, Throughput: 185.426667 im/s
Epoch 20, Iter 39, Loss: 11320.8808594, Throughput: 187.756796 im/s
Epoch 20, Train Loss: 11095.1435547, Time: 26.5877s, Throughput: 187.755673 im/s
Epoch 21, Iter 39, Loss: 10928.8974609, Throughput: 187.435814 im/s
Epoch 21, Train Loss: 10979.2792969, Time: 26.6333s, Throughput: 187.434787 im/s
Epoch 22, Iter 39, Loss: 11003.1044922, Throughput: 187.805686 im/s
Epoch 22, Train Loss: 10908.9306641, Time: 26.5809s, Throughput: 187.804364 im/s
Epoch 23, Iter 39, Loss: 10672.1445312, Throughput: 188.105215 im/s
Epoch 23, Train Loss: 10783.4072266, Time: 26.5385s, Throughput: 188.103955 im/s
Epoch 24, Iter 39, Loss: 10215.2773438, Throughput: 187.541771 im/s
Epoch 24, Train Loss: 10806.5146484, Time: 26.6182s, Throughput: 187.540694 im/s
Epoch 25, Iter 39, Loss: 10449.2304688, Throughput: 187.924007 im/s
Epoch 25, Train Loss: 10728.7587891, Time: 26.5641s, Throughput: 187.922734 im/s
Epoch 26, Iter 39, Loss: 11019.0839844, Throughput: 187.785401 im/s
Epoch 26, Train Loss: 10641.0722656, Time: 26.5837s, Throughput: 187.784349 im/s
Epoch 27, Iter 39, Loss: 9846.6787109, Throughput: 187.365921 im/s
Epoch 27, Train Loss: 10541.6679688, Time: 26.6432s, Throughput: 187.364732 im/s
Epoch 28, Iter 39, Loss: 10550.5244141, Throughput: 187.793822 im/s
Epoch 28, Train Loss: 10582.3505859, Time: 26.5825s, Throughput: 187.792756 im/s
Epoch 29, Iter 39, Loss: 9763.7802734, Throughput: 187.845960 im/s
Epoch 29, Train Loss: 10466.8427734, Time: 26.5751s, Throughput: 187.844921 im/s
Epoch 30, Iter 39, Loss: 10695.3837891, Throughput: 187.410385 im/s
Epoch 30, Train Loss: 10436.4794922, Time: 26.6369s, Throughput: 187.409407 im/s
Epoch 31, Iter 39, Loss: 11293.1201172, Throughput: 187.586790 im/s
Epoch 31, Train Loss: 10427.9111328, Time: 26.6118s, Throughput: 187.585627 im/s
Epoch 32, Iter 39, Loss: 10724.4375000, Throughput: 187.345797 im/s
Epoch 32, Train Loss: 10348.8076172, Time: 26.6461s, Throughput: 187.344756 im/s
Epoch 33, Iter 39, Loss: 9598.4541016, Throughput: 187.891357 im/s
Epoch 33, Train Loss: 10293.0263672, Time: 26.5687s, Throughput: 187.890310 im/s
Epoch 34, Iter 39, Loss: 10512.9882812, Throughput: 187.562214 im/s
Epoch 34, Train Loss: 10236.2363281, Time: 26.6153s, Throughput: 187.561072 im/s
Epoch 35, Iter 39, Loss: 9987.0478516, Throughput: 187.573756 im/s
Epoch 35, Train Loss: 10099.2568359, Time: 26.6137s, Throughput: 187.572573 im/s
Epoch 36, Iter 39, Loss: 10498.1191406, Throughput: 185.320439 im/s
Epoch 36, Train Loss: 10081.6494141, Time: 26.9373s, Throughput: 185.319310 im/s
Epoch 37, Iter 39, Loss: 9544.6855469, Throughput: 187.963838 im/s
Epoch 37, Train Loss: 10149.1337891, Time: 26.5585s, Throughput: 187.962719 im/s
Epoch 38, Iter 39, Loss: 10264.4101562, Throughput: 187.438378 im/s
Epoch 38, Train Loss: 10033.2968750, Time: 26.6329s, Throughput: 187.437140 im/s
Epoch 39, Iter 39, Loss: 9849.1455078, Throughput: 187.719384 im/s
Epoch 39, Train Loss: 10007.7285156, Time: 26.5931s, Throughput: 187.718156 im/s
Epoch 40, Iter 39, Loss: 9641.7910156, Throughput: 187.935442 im/s
Epoch 40, Train Loss: 9976.3525391, Time: 26.5625s, Throughput: 187.934281 im/s
Epoch 41, Iter 39, Loss: 9864.2529297, Throughput: 187.121170 im/s
Epoch 41, Train Loss: 9980.0302734, Time: 26.6781s, Throughput: 187.120096 im/s
Epoch 42, Iter 39, Loss: 9931.8593750, Throughput: 187.474375 im/s
Epoch 42, Train Loss: 9907.8642578, Time: 26.6278s, Throughput: 187.473200 im/s
Epoch 43, Iter 39, Loss: 10300.0595703, Throughput: 187.111327 im/s
Epoch 43, Train Loss: 9908.8642578, Time: 26.6795s, Throughput: 187.110073 im/s
Epoch 44, Iter 39, Loss: 9660.0810547, Throughput: 187.683404 im/s
Epoch 44, Train Loss: 9850.8076172, Time: 26.5981s, Throughput: 187.682394 im/s
Epoch 45, Iter 39, Loss: 10040.7353516, Throughput: 187.581940 im/s
Epoch 45, Train Loss: 9768.7099609, Time: 26.6127s, Throughput: 187.579537 im/s
Epoch 46, Iter 39, Loss: 9770.1894531, Throughput: 188.031905 im/s
Epoch 46, Train Loss: 9783.3154297, Time: 26.5488s, Throughput: 188.030856 im/s
Epoch 47, Iter 39, Loss: 9576.1162109, Throughput: 187.739165 im/s
Epoch 47, Train Loss: 9743.2001953, Time: 26.5902s, Throughput: 187.738141 im/s
Epoch 48, Iter 39, Loss: 9370.7109375, Throughput: 187.523441 im/s
Epoch 48, Train Loss: 9676.4580078, Time: 26.6209s, Throughput: 187.522080 im/s
Epoch 49, Iter 39, Loss: 9261.1601562, Throughput: 187.676136 im/s
Epoch 49, Train Loss: 9693.3535156, Time: 26.5992s, Throughput: 187.674972 im/s
Epoch 50, Iter 39, Loss: 9954.7832031, Throughput: 187.275219 im/s
Epoch 50, Train Loss: 9698.2333984, Time: 26.6561s, Throughput: 187.274214 im/s

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
