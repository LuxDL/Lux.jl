---
url: /previews/PR1857/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24095.3105469, Throughput: 20.317673 im/s
Epoch 1, Train Loss: 39846.2304688, Time: 245.8964s, Throughput: 20.301236 im/s
Epoch 2, Iter 39, Loss: 17134.9296875, Throughput: 121.590029 im/s
Epoch 2, Train Loss: 20207.8925781, Time: 41.0562s, Throughput: 121.589449 im/s
Epoch 3, Iter 39, Loss: 15695.8242188, Throughput: 121.508014 im/s
Epoch 3, Train Loss: 16623.3222656, Time: 41.0839s, Throughput: 121.507508 im/s
Epoch 4, Iter 39, Loss: 14075.8476562, Throughput: 121.584328 im/s
Epoch 4, Train Loss: 15075.0781250, Time: 41.0581s, Throughput: 121.583816 im/s
Epoch 5, Iter 39, Loss: 13659.1718750, Throughput: 120.892975 im/s
Epoch 5, Train Loss: 14068.8330078, Time: 41.2929s, Throughput: 120.892474 im/s
Epoch 6, Iter 39, Loss: 13437.2207031, Throughput: 120.697649 im/s
Epoch 6, Train Loss: 13277.9218750, Time: 41.3597s, Throughput: 120.697148 im/s
Epoch 7, Iter 39, Loss: 13379.6796875, Throughput: 120.698032 im/s
Epoch 7, Train Loss: 12901.8984375, Time: 41.3596s, Throughput: 120.697445 im/s
Epoch 8, Iter 39, Loss: 12937.8583984, Throughput: 119.961352 im/s
Epoch 8, Train Loss: 12524.0429688, Time: 41.6136s, Throughput: 119.960799 im/s
Epoch 9, Iter 39, Loss: 12030.1445312, Throughput: 120.554248 im/s
Epoch 9, Train Loss: 12171.5556641, Time: 41.4089s, Throughput: 120.553718 im/s
Epoch 10, Iter 39, Loss: 12631.0234375, Throughput: 120.869095 im/s
Epoch 10, Train Loss: 11861.0458984, Time: 41.3011s, Throughput: 120.868583 im/s
Epoch 11, Iter 39, Loss: 11943.0898438, Throughput: 121.080742 im/s
Epoch 11, Train Loss: 11663.5380859, Time: 41.2289s, Throughput: 121.080234 im/s
Epoch 12, Iter 39, Loss: 11361.1201172, Throughput: 121.088069 im/s
Epoch 12, Train Loss: 11456.0273438, Time: 41.2264s, Throughput: 121.087356 im/s
Epoch 13, Iter 39, Loss: 10983.6347656, Throughput: 121.130496 im/s
Epoch 13, Train Loss: 11363.7011719, Time: 41.2119s, Throughput: 121.129953 im/s
Epoch 14, Iter 39, Loss: 10992.2031250, Throughput: 120.806400 im/s
Epoch 14, Train Loss: 11127.8554688, Time: 41.3225s, Throughput: 120.805847 im/s
Epoch 15, Iter 39, Loss: 11357.6220703, Throughput: 120.352891 im/s
Epoch 15, Train Loss: 11026.0156250, Time: 41.4782s, Throughput: 120.352389 im/s
Epoch 16, Iter 39, Loss: 10713.4316406, Throughput: 120.420708 im/s
Epoch 16, Train Loss: 10905.9199219, Time: 41.4548s, Throughput: 120.420273 im/s
Epoch 17, Iter 39, Loss: 11436.2236328, Throughput: 120.262470 im/s
Epoch 17, Train Loss: 10951.0117188, Time: 41.5094s, Throughput: 120.261951 im/s
Epoch 18, Iter 39, Loss: 10978.4589844, Throughput: 120.183705 im/s
Epoch 18, Train Loss: 10875.2265625, Time: 41.5366s, Throughput: 120.183245 im/s
Epoch 19, Iter 39, Loss: 10950.1279297, Throughput: 120.789747 im/s
Epoch 19, Train Loss: 10703.4638672, Time: 41.3282s, Throughput: 120.789206 im/s
Epoch 20, Iter 39, Loss: 10544.8046875, Throughput: 120.492996 im/s
Epoch 20, Train Loss: 10530.2285156, Time: 41.4300s, Throughput: 120.492431 im/s
Epoch 21, Iter 39, Loss: 10349.6845703, Throughput: 119.700992 im/s
Epoch 21, Train Loss: 10471.4013672, Time: 41.7041s, Throughput: 119.700527 im/s
Epoch 22, Iter 39, Loss: 10352.1210938, Throughput: 120.309403 im/s
Epoch 22, Train Loss: 10431.3750000, Time: 41.4932s, Throughput: 120.308888 im/s
Epoch 23, Iter 39, Loss: 10545.6660156, Throughput: 121.428185 im/s
Epoch 23, Train Loss: 10373.0146484, Time: 41.1109s, Throughput: 121.427659 im/s
Epoch 24, Iter 39, Loss: 10176.1523438, Throughput: 121.358621 im/s
Epoch 24, Train Loss: 10286.6875000, Time: 41.1345s, Throughput: 121.357999 im/s
Epoch 25, Iter 39, Loss: 10402.1240234, Throughput: 121.400645 im/s
Epoch 25, Train Loss: 10213.9482422, Time: 41.1203s, Throughput: 121.400034 im/s
Epoch 26, Iter 39, Loss: 10323.5410156, Throughput: 121.076478 im/s
Epoch 26, Train Loss: 10172.3730469, Time: 41.2303s, Throughput: 121.075861 im/s
Epoch 27, Iter 39, Loss: 10319.7861328, Throughput: 121.222196 im/s
Epoch 27, Train Loss: 10103.0332031, Time: 41.1807s, Throughput: 121.221702 im/s
Epoch 28, Iter 39, Loss: 10671.9267578, Throughput: 121.011464 im/s
Epoch 28, Train Loss: 10088.8740234, Time: 41.2525s, Throughput: 121.010889 im/s
Epoch 29, Iter 39, Loss: 9798.8398438, Throughput: 122.059251 im/s
Epoch 29, Train Loss: 10087.6533203, Time: 40.8983s, Throughput: 122.058780 im/s
Epoch 30, Iter 39, Loss: 9423.8623047, Throughput: 121.992233 im/s
Epoch 30, Train Loss: 9968.5312500, Time: 40.9208s, Throughput: 121.991756 im/s
Epoch 31, Iter 39, Loss: 10179.5488281, Throughput: 121.385950 im/s
Epoch 31, Train Loss: 9864.2255859, Time: 41.1252s, Throughput: 121.385507 im/s
Epoch 32, Iter 39, Loss: 10067.7607422, Throughput: 122.226520 im/s
Epoch 32, Train Loss: 9887.4267578, Time: 40.8424s, Throughput: 122.226072 im/s
Epoch 33, Iter 39, Loss: 9886.2158203, Throughput: 121.037694 im/s
Epoch 33, Train Loss: 9774.4267578, Time: 41.2435s, Throughput: 121.037201 im/s
Epoch 34, Iter 39, Loss: 10051.7050781, Throughput: 121.467227 im/s
Epoch 34, Train Loss: 9775.0205078, Time: 41.0977s, Throughput: 121.466589 im/s
Epoch 35, Iter 39, Loss: 10028.2539062, Throughput: 121.117663 im/s
Epoch 35, Train Loss: 9748.7705078, Time: 41.2163s, Throughput: 121.117131 im/s
Epoch 36, Iter 39, Loss: 9975.2998047, Throughput: 120.425844 im/s
Epoch 36, Train Loss: 9733.0126953, Time: 41.4531s, Throughput: 120.425263 im/s
Epoch 37, Iter 39, Loss: 9521.3750000, Throughput: 121.436648 im/s
Epoch 37, Train Loss: 9739.9062500, Time: 41.1081s, Throughput: 121.436043 im/s
Epoch 38, Iter 39, Loss: 9738.5195312, Throughput: 121.094608 im/s
Epoch 38, Train Loss: 9646.6416016, Time: 41.2241s, Throughput: 121.094100 im/s
Epoch 39, Iter 39, Loss: 9680.2685547, Throughput: 121.898850 im/s
Epoch 39, Train Loss: 9679.3251953, Time: 40.9522s, Throughput: 121.898308 im/s
Epoch 40, Iter 39, Loss: 9746.3330078, Throughput: 121.211024 im/s
Epoch 40, Train Loss: 9614.7255859, Time: 41.1846s, Throughput: 121.210467 im/s
Epoch 41, Iter 39, Loss: 10171.0585938, Throughput: 122.338615 im/s
Epoch 41, Train Loss: 9510.1562500, Time: 40.8049s, Throughput: 122.338134 im/s
Epoch 42, Iter 39, Loss: 9772.9404297, Throughput: 122.485757 im/s
Epoch 42, Train Loss: 9488.6855469, Time: 40.7559s, Throughput: 122.485297 im/s
Epoch 43, Iter 39, Loss: 10110.7500000, Throughput: 121.205199 im/s
Epoch 43, Train Loss: 9510.4843750, Time: 41.1865s, Throughput: 121.204634 im/s
Epoch 44, Iter 39, Loss: 9342.6132812, Throughput: 122.248290 im/s
Epoch 44, Train Loss: 9462.2226562, Time: 40.8351s, Throughput: 122.247733 im/s
Epoch 45, Iter 39, Loss: 8891.9492188, Throughput: 122.063148 im/s
Epoch 45, Train Loss: 9415.2480469, Time: 40.8970s, Throughput: 122.062629 im/s
Epoch 46, Iter 39, Loss: 9017.7177734, Throughput: 122.065570 im/s
Epoch 46, Train Loss: 9352.6923828, Time: 40.8963s, Throughput: 122.064970 im/s
Epoch 47, Iter 39, Loss: 9556.0664062, Throughput: 121.370483 im/s
Epoch 47, Train Loss: 9438.0400391, Time: 41.1305s, Throughput: 121.369864 im/s
Epoch 48, Iter 39, Loss: 9405.5898438, Throughput: 121.587548 im/s
Epoch 48, Train Loss: 9458.0449219, Time: 41.0571s, Throughput: 121.586852 im/s
Epoch 49, Iter 39, Loss: 8770.9912109, Throughput: 121.245727 im/s
Epoch 49, Train Loss: 9319.9345703, Time: 41.1728s, Throughput: 121.245218 im/s
Epoch 50, Iter 39, Loss: 9554.6943359, Throughput: 121.153306 im/s
Epoch 50, Train Loss: 9237.3007812, Time: 41.2042s, Throughput: 121.152754 im/s

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
