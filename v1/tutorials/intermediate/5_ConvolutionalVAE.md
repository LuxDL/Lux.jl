---
url: /v1/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24785.4238281, Throughput: 22.844700 im/s
Epoch 1, Train Loss: 40406.2890625, Time: 218.7085s, Throughput: 22.824898 im/s
Epoch 2, Iter 39, Loss: 18464.6738281, Throughput: 100.618891 im/s
Epoch 2, Train Loss: 20811.5078125, Time: 49.6131s, Throughput: 100.618526 im/s
Epoch 3, Iter 39, Loss: 16283.3740234, Throughput: 100.372496 im/s
Epoch 3, Train Loss: 17158.5898438, Time: 49.7452s, Throughput: 100.351302 im/s
Epoch 4, Iter 39, Loss: 15416.5097656, Throughput: 100.616126 im/s
Epoch 4, Train Loss: 15620.4951172, Time: 49.6145s, Throughput: 100.615788 im/s
Epoch 5, Iter 39, Loss: 13865.8017578, Throughput: 100.440369 im/s
Epoch 5, Train Loss: 14651.1748047, Time: 49.7013s, Throughput: 100.440017 im/s
Epoch 6, Iter 39, Loss: 14187.8183594, Throughput: 100.824275 im/s
Epoch 6, Train Loss: 14022.8876953, Time: 49.5121s, Throughput: 100.823918 im/s
Epoch 7, Iter 39, Loss: 13345.2441406, Throughput: 100.507240 im/s
Epoch 7, Train Loss: 13427.3095703, Time: 49.6682s, Throughput: 100.506892 im/s
Epoch 8, Iter 39, Loss: 13411.6220703, Throughput: 100.527646 im/s
Epoch 8, Train Loss: 13109.1464844, Time: 49.6582s, Throughput: 100.527251 im/s
Epoch 9, Iter 39, Loss: 13138.2890625, Throughput: 100.375251 im/s
Epoch 9, Train Loss: 12815.4316406, Time: 49.7336s, Throughput: 100.374874 im/s
Epoch 10, Iter 39, Loss: 13111.6923828, Throughput: 100.265752 im/s
Epoch 10, Train Loss: 12600.2744141, Time: 49.7879s, Throughput: 100.265384 im/s
Epoch 11, Iter 39, Loss: 12258.9179688, Throughput: 100.508925 im/s
Epoch 11, Train Loss: 12198.2099609, Time: 49.6674s, Throughput: 100.508607 im/s
Epoch 12, Iter 39, Loss: 11520.1074219, Throughput: 100.172203 im/s
Epoch 12, Train Loss: 12011.0693359, Time: 49.8344s, Throughput: 100.171863 im/s
Epoch 13, Iter 39, Loss: 11735.9882812, Throughput: 99.935590 im/s
Epoch 13, Train Loss: 11842.4287109, Time: 49.9523s, Throughput: 99.935255 im/s
Epoch 14, Iter 39, Loss: 12539.5546875, Throughput: 100.442666 im/s
Epoch 14, Train Loss: 11710.0273438, Time: 49.7002s, Throughput: 100.442310 im/s
Epoch 15, Iter 39, Loss: 11867.1435547, Throughput: 100.369501 im/s
Epoch 15, Train Loss: 11544.2880859, Time: 49.7364s, Throughput: 100.369158 im/s
Epoch 16, Iter 39, Loss: 10943.1191406, Throughput: 100.286731 im/s
Epoch 16, Train Loss: 11412.9541016, Time: 49.7775s, Throughput: 100.286365 im/s
Epoch 17, Iter 39, Loss: 10566.1093750, Throughput: 100.012758 im/s
Epoch 17, Train Loss: 11306.1845703, Time: 49.9138s, Throughput: 100.012392 im/s
Epoch 18, Iter 39, Loss: 10909.0625000, Throughput: 100.266091 im/s
Epoch 18, Train Loss: 11199.4873047, Time: 49.7877s, Throughput: 100.265740 im/s
Epoch 19, Iter 39, Loss: 10493.2207031, Throughput: 100.052880 im/s
Epoch 19, Train Loss: 11090.8066406, Time: 49.8938s, Throughput: 100.052475 im/s
Epoch 20, Iter 39, Loss: 11201.6171875, Throughput: 99.669635 im/s
Epoch 20, Train Loss: 11099.7109375, Time: 50.0856s, Throughput: 99.669305 im/s
Epoch 21, Iter 39, Loss: 11303.4101562, Throughput: 100.576477 im/s
Epoch 21, Train Loss: 10911.2558594, Time: 49.6340s, Throughput: 100.576131 im/s
Epoch 22, Iter 39, Loss: 10889.8466797, Throughput: 100.294754 im/s
Epoch 22, Train Loss: 10818.9599609, Time: 49.7735s, Throughput: 100.294369 im/s
Epoch 23, Iter 39, Loss: 10306.9394531, Throughput: 100.262728 im/s
Epoch 23, Train Loss: 10749.7607422, Time: 49.7894s, Throughput: 100.262386 im/s
Epoch 24, Iter 39, Loss: 11380.6748047, Throughput: 99.795138 im/s
Epoch 24, Train Loss: 10606.8574219, Time: 50.0227s, Throughput: 99.794775 im/s
Epoch 25, Iter 39, Loss: 10677.2265625, Throughput: 99.950217 im/s
Epoch 25, Train Loss: 10572.6367188, Time: 49.9451s, Throughput: 99.949819 im/s
Epoch 26, Iter 39, Loss: 10325.8085938, Throughput: 100.226598 im/s
Epoch 26, Train Loss: 10573.3027344, Time: 49.8073s, Throughput: 100.226266 im/s
Epoch 27, Iter 39, Loss: 10404.2333984, Throughput: 100.461054 im/s
Epoch 27, Train Loss: 10492.1845703, Time: 49.6911s, Throughput: 100.460698 im/s
Epoch 28, Iter 39, Loss: 10603.1093750, Throughput: 100.087402 im/s
Epoch 28, Train Loss: 10384.6005859, Time: 49.8766s, Throughput: 100.087021 im/s
Epoch 29, Iter 39, Loss: 10407.2441406, Throughput: 100.577108 im/s
Epoch 29, Train Loss: 10454.4072266, Time: 49.6337s, Throughput: 100.576767 im/s
Epoch 30, Iter 39, Loss: 10013.7851562, Throughput: 100.749217 im/s
Epoch 30, Train Loss: 10310.4912109, Time: 49.5490s, Throughput: 100.748847 im/s
Epoch 31, Iter 39, Loss: 9538.3164062, Throughput: 100.559975 im/s
Epoch 31, Train Loss: 10215.6240234, Time: 49.6422s, Throughput: 100.559619 im/s
Epoch 32, Iter 39, Loss: 10486.8183594, Throughput: 100.025174 im/s
Epoch 32, Train Loss: 10180.6054688, Time: 49.9076s, Throughput: 100.024800 im/s
Epoch 33, Iter 39, Loss: 10114.9873047, Throughput: 99.939888 im/s
Epoch 33, Train Loss: 10180.2324219, Time: 49.9502s, Throughput: 99.939482 im/s
Epoch 34, Iter 39, Loss: 10918.2978516, Throughput: 100.181352 im/s
Epoch 34, Train Loss: 10076.9394531, Time: 49.8298s, Throughput: 100.180958 im/s
Epoch 35, Iter 39, Loss: 9947.0175781, Throughput: 100.236357 im/s
Epoch 35, Train Loss: 10017.9365234, Time: 49.8025s, Throughput: 100.236002 im/s
Epoch 36, Iter 39, Loss: 9825.4648438, Throughput: 100.157258 im/s
Epoch 36, Train Loss: 10043.7822266, Time: 49.8418s, Throughput: 100.156951 im/s
Epoch 37, Iter 39, Loss: 10535.2216797, Throughput: 100.040650 im/s
Epoch 37, Train Loss: 10004.1142578, Time: 49.8999s, Throughput: 100.040293 im/s
Epoch 38, Iter 39, Loss: 10025.5527344, Throughput: 100.224785 im/s
Epoch 38, Train Loss: 9959.6806641, Time: 49.8082s, Throughput: 100.224393 im/s
Epoch 39, Iter 39, Loss: 9861.8867188, Throughput: 100.154648 im/s
Epoch 39, Train Loss: 9941.8183594, Time: 49.8431s, Throughput: 100.154306 im/s
Epoch 40, Iter 39, Loss: 10208.5312500, Throughput: 100.039341 im/s
Epoch 40, Train Loss: 9875.2353516, Time: 49.9006s, Throughput: 100.038964 im/s
Epoch 41, Iter 39, Loss: 9787.5742188, Throughput: 100.105414 im/s
Epoch 41, Train Loss: 9775.8681641, Time: 49.8676s, Throughput: 100.105041 im/s
Epoch 42, Iter 39, Loss: 9597.6494141, Throughput: 100.319640 im/s
Epoch 42, Train Loss: 9800.3642578, Time: 49.7611s, Throughput: 100.319243 im/s
Epoch 43, Iter 39, Loss: 9625.2978516, Throughput: 100.450752 im/s
Epoch 43, Train Loss: 9717.3613281, Time: 49.6962s, Throughput: 100.450402 im/s
Epoch 44, Iter 39, Loss: 9807.7705078, Throughput: 100.295977 im/s
Epoch 44, Train Loss: 9738.2675781, Time: 49.7729s, Throughput: 100.295603 im/s
Epoch 45, Iter 39, Loss: 9875.7363281, Throughput: 100.161338 im/s
Epoch 45, Train Loss: 9732.6660156, Time: 49.8398s, Throughput: 100.160972 im/s
Epoch 46, Iter 39, Loss: 9270.5791016, Throughput: 100.218698 im/s
Epoch 46, Train Loss: 9694.2626953, Time: 49.8112s, Throughput: 100.218324 im/s
Epoch 47, Iter 39, Loss: 10364.0566406, Throughput: 100.102288 im/s
Epoch 47, Train Loss: 9674.5947266, Time: 49.8692s, Throughput: 100.101919 im/s
Epoch 48, Iter 39, Loss: 9967.1083984, Throughput: 100.183831 im/s
Epoch 48, Train Loss: 9669.5917969, Time: 49.8286s, Throughput: 100.183391 im/s
Epoch 49, Iter 39, Loss: 9332.9531250, Throughput: 100.131971 im/s
Epoch 49, Train Loss: 9633.5820312, Time: 49.8544s, Throughput: 100.131662 im/s
Epoch 50, Iter 39, Loss: 8960.3320312, Throughput: 99.992945 im/s
Epoch 50, Train Loss: 9567.5000000, Time: 49.9237s, Throughput: 99.992523 im/s

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
Julia Version 1.13.4
Commit c1ac32ab733 (2026-11-13 17:30 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 4 × Intel(R) Xeon(R) 6975P-C
  WORD_SIZE: 64
  LLVM: libLLVM-20.1.8 (ORCJIT, graniterapids)
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
