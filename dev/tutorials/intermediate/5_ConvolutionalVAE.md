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
Epoch 1, Iter 39, Loss: 25857.8164062, Throughput: 17.379297 im/s
Epoch 1, Train Loss: 40049.4570312, Time: 287.4960s, Throughput: 17.363719 im/s
Epoch 2, Iter 39, Loss: 18202.8769531, Throughput: 75.502163 im/s
Epoch 2, Train Loss: 20681.1386719, Time: 66.1175s, Throughput: 75.501915 im/s
Epoch 3, Iter 39, Loss: 15772.9902344, Throughput: 75.485833 im/s
Epoch 3, Train Loss: 17088.1035156, Time: 66.1318s, Throughput: 75.485595 im/s
Epoch 4, Iter 39, Loss: 14639.9082031, Throughput: 75.575295 im/s
Epoch 4, Train Loss: 15638.6455078, Time: 66.0535s, Throughput: 75.575064 im/s
Epoch 5, Iter 39, Loss: 14652.9482422, Throughput: 75.566991 im/s
Epoch 5, Train Loss: 14592.5546875, Time: 66.0608s, Throughput: 75.566768 im/s
Epoch 6, Iter 39, Loss: 13207.8847656, Throughput: 75.569603 im/s
Epoch 6, Train Loss: 13956.4326172, Time: 66.0585s, Throughput: 75.569377 im/s
Epoch 7, Iter 39, Loss: 12254.5810547, Throughput: 75.514881 im/s
Epoch 7, Train Loss: 13477.7968750, Time: 66.1064s, Throughput: 75.514658 im/s
Epoch 8, Iter 39, Loss: 13489.4199219, Throughput: 75.542398 im/s
Epoch 8, Train Loss: 13075.5859375, Time: 66.0823s, Throughput: 75.542160 im/s
Epoch 9, Iter 39, Loss: 12613.7919922, Throughput: 75.633727 im/s
Epoch 9, Train Loss: 12750.4550781, Time: 66.0025s, Throughput: 75.633453 im/s
Epoch 10, Iter 39, Loss: 12539.3515625, Throughput: 75.639397 im/s
Epoch 10, Train Loss: 12459.5859375, Time: 65.9976s, Throughput: 75.639158 im/s
Epoch 11, Iter 39, Loss: 12439.5000000, Throughput: 75.682420 im/s
Epoch 11, Train Loss: 12374.4755859, Time: 65.9600s, Throughput: 75.682190 im/s
Epoch 12, Iter 39, Loss: 13526.4873047, Throughput: 75.693546 im/s
Epoch 12, Train Loss: 12035.1513672, Time: 65.9503s, Throughput: 75.693329 im/s
Epoch 13, Iter 39, Loss: 11118.9531250, Throughput: 75.583109 im/s
Epoch 13, Train Loss: 11814.3857422, Time: 66.0467s, Throughput: 75.582887 im/s
Epoch 14, Iter 39, Loss: 11814.6640625, Throughput: 75.533917 im/s
Epoch 14, Train Loss: 11684.9785156, Time: 66.0897s, Throughput: 75.533679 im/s
Epoch 15, Iter 39, Loss: 10986.2343750, Throughput: 75.645540 im/s
Epoch 15, Train Loss: 11591.4130859, Time: 65.9922s, Throughput: 75.645295 im/s
Epoch 16, Iter 39, Loss: 11379.6777344, Throughput: 75.663715 im/s
Epoch 16, Train Loss: 11481.9316406, Time: 65.9764s, Throughput: 75.663439 im/s
Epoch 17, Iter 39, Loss: 11181.9628906, Throughput: 75.764986 im/s
Epoch 17, Train Loss: 11529.8085938, Time: 65.8882s, Throughput: 75.764745 im/s
Epoch 18, Iter 39, Loss: 10647.3515625, Throughput: 75.739158 im/s
Epoch 18, Train Loss: 11264.4619141, Time: 65.9106s, Throughput: 75.738910 im/s
Epoch 19, Iter 39, Loss: 11392.3837891, Throughput: 75.743573 im/s
Epoch 19, Train Loss: 11235.5380859, Time: 65.9068s, Throughput: 75.743355 im/s
Epoch 20, Iter 39, Loss: 11962.2812500, Throughput: 75.726841 im/s
Epoch 20, Train Loss: 11085.9648438, Time: 65.9213s, Throughput: 75.726632 im/s
Epoch 21, Iter 39, Loss: 10632.3867188, Throughput: 75.714924 im/s
Epoch 21, Train Loss: 11091.0830078, Time: 65.9317s, Throughput: 75.714728 im/s
Epoch 22, Iter 39, Loss: 10560.4414062, Throughput: 75.836731 im/s
Epoch 22, Train Loss: 10917.3359375, Time: 65.8258s, Throughput: 75.836472 im/s
Epoch 23, Iter 39, Loss: 10925.0195312, Throughput: 75.799187 im/s
Epoch 23, Train Loss: 10872.1875000, Time: 65.8584s, Throughput: 75.798935 im/s
Epoch 24, Iter 39, Loss: 10675.7460938, Throughput: 75.709809 im/s
Epoch 24, Train Loss: 10706.3359375, Time: 65.9362s, Throughput: 75.709561 im/s
Epoch 25, Iter 39, Loss: 11280.7119141, Throughput: 75.720833 im/s
Epoch 25, Train Loss: 10718.6386719, Time: 65.9266s, Throughput: 75.720603 im/s
Epoch 26, Iter 39, Loss: 10686.4218750, Throughput: 75.765990 im/s
Epoch 26, Train Loss: 10696.6503906, Time: 65.8873s, Throughput: 75.765744 im/s
Epoch 27, Iter 39, Loss: 10724.5429688, Throughput: 75.853876 im/s
Epoch 27, Train Loss: 10550.4726562, Time: 65.8109s, Throughput: 75.853642 im/s
Epoch 28, Iter 39, Loss: 10138.4707031, Throughput: 75.788011 im/s
Epoch 28, Train Loss: 10514.5332031, Time: 65.8681s, Throughput: 75.787800 im/s
Epoch 29, Iter 39, Loss: 10420.4121094, Throughput: 75.788785 im/s
Epoch 29, Train Loss: 10493.5390625, Time: 65.8675s, Throughput: 75.788545 im/s
Epoch 30, Iter 39, Loss: 11371.3916016, Throughput: 75.750497 im/s
Epoch 30, Train Loss: 10407.1796875, Time: 65.9008s, Throughput: 75.750243 im/s
Epoch 31, Iter 39, Loss: 10432.4931641, Throughput: 75.623350 im/s
Epoch 31, Train Loss: 10504.0419922, Time: 66.0116s, Throughput: 75.623109 im/s
Epoch 32, Iter 39, Loss: 10416.0429688, Throughput: 75.769841 im/s
Epoch 32, Train Loss: 10350.3906250, Time: 65.8839s, Throughput: 75.769634 im/s
Epoch 33, Iter 39, Loss: 10410.6259766, Throughput: 75.873006 im/s
Epoch 33, Train Loss: 10242.6894531, Time: 65.7944s, Throughput: 75.872757 im/s
Epoch 34, Iter 39, Loss: 9762.8232422, Throughput: 75.822487 im/s
Epoch 34, Train Loss: 10207.3876953, Time: 65.8382s, Throughput: 75.822267 im/s
Epoch 35, Iter 39, Loss: 10152.5546875, Throughput: 75.774374 im/s
Epoch 35, Train Loss: 10226.8818359, Time: 65.8800s, Throughput: 75.774165 im/s
Epoch 36, Iter 39, Loss: 10093.5605469, Throughput: 75.714007 im/s
Epoch 36, Train Loss: 10186.4677734, Time: 65.9325s, Throughput: 75.713794 im/s
Epoch 37, Iter 39, Loss: 9936.0878906, Throughput: 75.705668 im/s
Epoch 37, Train Loss: 10111.9082031, Time: 65.9398s, Throughput: 75.705447 im/s
Epoch 38, Iter 39, Loss: 11028.1640625, Throughput: 75.778803 im/s
Epoch 38, Train Loss: 10175.6943359, Time: 65.8761s, Throughput: 75.778609 im/s
Epoch 39, Iter 39, Loss: 10550.4746094, Throughput: 75.720162 im/s
Epoch 39, Train Loss: 10167.8378906, Time: 65.9271s, Throughput: 75.719945 im/s
Epoch 40, Iter 39, Loss: 9463.3857422, Throughput: 75.635732 im/s
Epoch 40, Train Loss: 10025.7919922, Time: 66.0007s, Throughput: 75.635525 im/s
Epoch 41, Iter 39, Loss: 9789.3105469, Throughput: 75.624083 im/s
Epoch 41, Train Loss: 9984.6845703, Time: 66.0109s, Throughput: 75.623855 im/s
Epoch 42, Iter 39, Loss: 9832.6406250, Throughput: 75.538978 im/s
Epoch 42, Train Loss: 9927.9853516, Time: 66.0853s, Throughput: 75.538774 im/s
Epoch 43, Iter 39, Loss: 9631.9365234, Throughput: 75.387373 im/s
Epoch 43, Train Loss: 9940.9873047, Time: 66.2182s, Throughput: 75.387161 im/s
Epoch 44, Iter 39, Loss: 9986.0664062, Throughput: 75.489347 im/s
Epoch 44, Train Loss: 9912.0820312, Time: 66.1287s, Throughput: 75.489133 im/s
Epoch 45, Iter 39, Loss: 10060.5341797, Throughput: 75.658948 im/s
Epoch 45, Train Loss: 9870.2167969, Time: 65.9805s, Throughput: 75.658726 im/s
Epoch 46, Iter 39, Loss: 9778.9179688, Throughput: 75.559708 im/s
Epoch 46, Train Loss: 9827.4833984, Time: 66.0671s, Throughput: 75.559496 im/s
Epoch 47, Iter 39, Loss: 9152.4482422, Throughput: 75.762408 im/s
Epoch 47, Train Loss: 9753.7021484, Time: 65.8904s, Throughput: 75.762193 im/s
Epoch 48, Iter 39, Loss: 10002.3720703, Throughput: 75.619021 im/s
Epoch 48, Train Loss: 9777.3798828, Time: 66.0153s, Throughput: 75.618793 im/s
Epoch 49, Iter 39, Loss: 10227.7822266, Throughput: 75.562007 im/s
Epoch 49, Train Loss: 9763.8330078, Time: 66.0652s, Throughput: 75.561745 im/s
Epoch 50, Iter 39, Loss: 9786.4746094, Throughput: 75.581786 im/s
Epoch 50, Train Loss: 9722.8203125, Time: 66.0478s, Throughput: 75.581568 im/s

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
  CPU: 4 × INTEL(R) XEON(R) PLATINUM 8573C
  WORD_SIZE: 64
  LLVM: libLLVM-20.1.8 (ORCJIT, emeraldrapids)
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
