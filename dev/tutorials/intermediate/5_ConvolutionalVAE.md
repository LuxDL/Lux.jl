---
url: /dev/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23669.0214844, Throughput: 10.124549 im/s
Epoch 1, Train Loss: 39975.1601562, Time: 493.3634s, Throughput: 10.118301 im/s
Epoch 2, Iter 39, Loss: 18302.7792969, Throughput: 68.049236 im/s
Epoch 2, Train Loss: 20375.8925781, Time: 73.3589s, Throughput: 68.049023 im/s
Epoch 3, Iter 39, Loss: 16283.4511719, Throughput: 68.054038 im/s
Epoch 3, Train Loss: 16709.9121094, Time: 73.3537s, Throughput: 68.053838 im/s
Epoch 4, Iter 39, Loss: 13466.8798828, Throughput: 68.032190 im/s
Epoch 4, Train Loss: 15122.2822266, Time: 73.3772s, Throughput: 68.032016 im/s
Epoch 5, Iter 39, Loss: 14304.7539062, Throughput: 68.043028 im/s
Epoch 5, Train Loss: 14137.8144531, Time: 73.3655s, Throughput: 68.042864 im/s
Epoch 6, Iter 39, Loss: 12600.3085938, Throughput: 68.001696 im/s
Epoch 6, Train Loss: 13351.3906250, Time: 73.4102s, Throughput: 68.001472 im/s
Epoch 7, Iter 39, Loss: 12405.7949219, Throughput: 69.244548 im/s
Epoch 7, Train Loss: 12894.4794922, Time: 72.0925s, Throughput: 69.244354 im/s
Epoch 8, Iter 39, Loss: 12413.3720703, Throughput: 68.602105 im/s
Epoch 8, Train Loss: 12526.3232422, Time: 72.7676s, Throughput: 68.601936 im/s
Epoch 9, Iter 39, Loss: 11890.4316406, Throughput: 68.033301 im/s
Epoch 9, Train Loss: 12207.2285156, Time: 73.3760s, Throughput: 68.033115 im/s
Epoch 10, Iter 39, Loss: 12109.4013672, Throughput: 67.690280 im/s
Epoch 10, Train Loss: 11937.8769531, Time: 73.7479s, Throughput: 67.690100 im/s
Epoch 11, Iter 39, Loss: 11775.3535156, Throughput: 68.061594 im/s
Epoch 11, Train Loss: 11712.8369141, Time: 73.3455s, Throughput: 68.061436 im/s
Epoch 12, Iter 39, Loss: 11403.8300781, Throughput: 68.355921 im/s
Epoch 12, Train Loss: 11601.0283203, Time: 73.0297s, Throughput: 68.355734 im/s
Epoch 13, Iter 39, Loss: 11250.1308594, Throughput: 68.534802 im/s
Epoch 13, Train Loss: 11406.9287109, Time: 72.8391s, Throughput: 68.534622 im/s
Epoch 14, Iter 39, Loss: 11398.3183594, Throughput: 68.165342 im/s
Epoch 14, Train Loss: 11303.6406250, Time: 73.2339s, Throughput: 68.165161 im/s
Epoch 15, Iter 39, Loss: 11767.1406250, Throughput: 68.562296 im/s
Epoch 15, Train Loss: 11192.1826172, Time: 72.8099s, Throughput: 68.562135 im/s
Epoch 16, Iter 39, Loss: 10836.2226562, Throughput: 69.231020 im/s
Epoch 16, Train Loss: 10999.6738281, Time: 72.1066s, Throughput: 69.230829 im/s
Epoch 17, Iter 39, Loss: 10251.7041016, Throughput: 69.279984 im/s
Epoch 17, Train Loss: 10919.7597656, Time: 72.0556s, Throughput: 69.279818 im/s
Epoch 18, Iter 39, Loss: 10644.7070312, Throughput: 69.279546 im/s
Epoch 18, Train Loss: 10823.3603516, Time: 72.0561s, Throughput: 69.279341 im/s
Epoch 19, Iter 39, Loss: 11289.3828125, Throughput: 68.848847 im/s
Epoch 19, Train Loss: 10718.5107422, Time: 72.5068s, Throughput: 68.848679 im/s
Epoch 20, Iter 39, Loss: 10420.6542969, Throughput: 68.845076 im/s
Epoch 20, Train Loss: 10539.7568359, Time: 72.5108s, Throughput: 68.844940 im/s
Epoch 21, Iter 39, Loss: 10542.1757812, Throughput: 68.777805 im/s
Epoch 21, Train Loss: 10445.1005859, Time: 72.5817s, Throughput: 68.777634 im/s
Epoch 22, Iter 39, Loss: 10361.1679688, Throughput: 68.743540 im/s
Epoch 22, Train Loss: 10374.2812500, Time: 72.6179s, Throughput: 68.743382 im/s
Epoch 23, Iter 39, Loss: 10557.1943359, Throughput: 68.776692 im/s
Epoch 23, Train Loss: 10406.1728516, Time: 72.5829s, Throughput: 68.776532 im/s
Epoch 24, Iter 39, Loss: 10106.6113281, Throughput: 68.653364 im/s
Epoch 24, Train Loss: 10377.8544922, Time: 72.7133s, Throughput: 68.653187 im/s
Epoch 25, Iter 39, Loss: 10092.6054688, Throughput: 68.915060 im/s
Epoch 25, Train Loss: 10258.3574219, Time: 72.4372s, Throughput: 68.914895 im/s
Epoch 26, Iter 39, Loss: 10645.6298828, Throughput: 68.927061 im/s
Epoch 26, Train Loss: 10262.6572266, Time: 72.4246s, Throughput: 68.926895 im/s
Epoch 27, Iter 39, Loss: 10337.3339844, Throughput: 68.412689 im/s
Epoch 27, Train Loss: 10140.3837891, Time: 72.9691s, Throughput: 68.412527 im/s
Epoch 28, Iter 39, Loss: 10293.0546875, Throughput: 68.378347 im/s
Epoch 28, Train Loss: 10080.1660156, Time: 73.0057s, Throughput: 68.378184 im/s
Epoch 29, Iter 39, Loss: 10612.8369141, Throughput: 68.715038 im/s
Epoch 29, Train Loss: 10073.8808594, Time: 72.6480s, Throughput: 68.714876 im/s
Epoch 30, Iter 39, Loss: 10239.8535156, Throughput: 68.469637 im/s
Epoch 30, Train Loss: 9966.2177734, Time: 72.9084s, Throughput: 68.469470 im/s
Epoch 31, Iter 39, Loss: 10710.3222656, Throughput: 68.559571 im/s
Epoch 31, Train Loss: 10012.6406250, Time: 72.8128s, Throughput: 68.559358 im/s
Epoch 32, Iter 39, Loss: 9745.9951172, Throughput: 68.676341 im/s
Epoch 32, Train Loss: 9882.7548828, Time: 72.6890s, Throughput: 68.676143 im/s
Epoch 33, Iter 39, Loss: 9787.8847656, Throughput: 69.333868 im/s
Epoch 33, Train Loss: 9899.9130859, Time: 71.9996s, Throughput: 69.333696 im/s
Epoch 34, Iter 39, Loss: 10539.0322266, Throughput: 68.515452 im/s
Epoch 34, Train Loss: 9837.1962891, Time: 72.8597s, Throughput: 68.515289 im/s
Epoch 35, Iter 39, Loss: 9744.6542969, Throughput: 67.934433 im/s
Epoch 35, Train Loss: 9805.7861328, Time: 73.4828s, Throughput: 67.934235 im/s
Epoch 36, Iter 39, Loss: 9546.8281250, Throughput: 67.951674 im/s
Epoch 36, Train Loss: 9718.6044922, Time: 73.4642s, Throughput: 67.951502 im/s
Epoch 37, Iter 39, Loss: 10119.2109375, Throughput: 68.748607 im/s
Epoch 37, Train Loss: 9725.8994141, Time: 72.6126s, Throughput: 68.748417 im/s
Epoch 38, Iter 39, Loss: 9806.8525391, Throughput: 67.718084 im/s
Epoch 38, Train Loss: 9681.8994141, Time: 73.7175s, Throughput: 67.717933 im/s
Epoch 39, Iter 39, Loss: 9570.3046875, Throughput: 67.956688 im/s
Epoch 39, Train Loss: 9612.1855469, Time: 73.4587s, Throughput: 67.956508 im/s
Epoch 40, Iter 39, Loss: 9559.7421875, Throughput: 68.146191 im/s
Epoch 40, Train Loss: 9571.6318359, Time: 73.2545s, Throughput: 68.146002 im/s
Epoch 41, Iter 39, Loss: 9602.0146484, Throughput: 68.271227 im/s
Epoch 41, Train Loss: 9613.8828125, Time: 73.1203s, Throughput: 68.271059 im/s
Epoch 42, Iter 39, Loss: 9778.6142578, Throughput: 67.987427 im/s
Epoch 42, Train Loss: 9565.6318359, Time: 73.4255s, Throughput: 67.987251 im/s
Epoch 43, Iter 39, Loss: 9702.7597656, Throughput: 68.292909 im/s
Epoch 43, Train Loss: 9527.6259766, Time: 73.0971s, Throughput: 68.292736 im/s
Epoch 44, Iter 39, Loss: 9159.7744141, Throughput: 68.161749 im/s
Epoch 44, Train Loss: 9563.1494141, Time: 73.2377s, Throughput: 68.161586 im/s
Epoch 45, Iter 39, Loss: 9218.2128906, Throughput: 68.980294 im/s
Epoch 45, Train Loss: 9473.7011719, Time: 72.3687s, Throughput: 68.980122 im/s
Epoch 46, Iter 39, Loss: 9952.1992188, Throughput: 68.659445 im/s
Epoch 46, Train Loss: 9378.1591797, Time: 72.7069s, Throughput: 68.659256 im/s
Epoch 47, Iter 39, Loss: 9560.0312500, Throughput: 68.670714 im/s
Epoch 47, Train Loss: 9346.6210938, Time: 72.6949s, Throughput: 68.670546 im/s
Epoch 48, Iter 39, Loss: 9305.5390625, Throughput: 68.120295 im/s
Epoch 48, Train Loss: 9318.7275391, Time: 73.2823s, Throughput: 68.120108 im/s
Epoch 49, Iter 39, Loss: 9102.9833984, Throughput: 68.555589 im/s
Epoch 49, Train Loss: 9292.0361328, Time: 72.8170s, Throughput: 68.555394 im/s
Epoch 50, Iter 39, Loss: 9303.9628906, Throughput: 68.402689 im/s
Epoch 50, Train Loss: 9338.2158203, Time: 72.9798s, Throughput: 68.402518 im/s

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
Julia Version 1.12.6
Commit 15346901f00 (2026-04-09 19:20 UTC)
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
