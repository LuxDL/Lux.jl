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
Epoch 1, Iter 39, Loss: 23340.9589844, Throughput: 14.165961 im/s
Epoch 1, Train Loss: 39393.4023438, Time: 352.6586s, Throughput: 14.155334 im/s
Epoch 2, Iter 39, Loss: 18393.0605469, Throughput: 169.092981 im/s
Epoch 2, Train Loss: 20050.3457031, Time: 29.5225s, Throughput: 169.091555 im/s
Epoch 3, Iter 39, Loss: 15438.5644531, Throughput: 169.328014 im/s
Epoch 3, Train Loss: 16525.0429688, Time: 29.4814s, Throughput: 169.326998 im/s
Epoch 4, Iter 39, Loss: 14179.1386719, Throughput: 169.375189 im/s
Epoch 4, Train Loss: 15013.1406250, Time: 29.4732s, Throughput: 169.373941 im/s
Epoch 5, Iter 39, Loss: 13730.6972656, Throughput: 169.629964 im/s
Epoch 5, Train Loss: 14082.6220703, Time: 29.4290s, Throughput: 169.628806 im/s
Epoch 6, Iter 39, Loss: 13192.6855469, Throughput: 169.899303 im/s
Epoch 6, Train Loss: 13351.5195312, Time: 29.3823s, Throughput: 169.898176 im/s
Epoch 7, Iter 39, Loss: 12694.6308594, Throughput: 169.840973 im/s
Epoch 7, Train Loss: 12846.6943359, Time: 29.3924s, Throughput: 169.840003 im/s
Epoch 8, Iter 39, Loss: 12400.9296875, Throughput: 169.575010 im/s
Epoch 8, Train Loss: 12530.3818359, Time: 29.4385s, Throughput: 169.573834 im/s
Epoch 9, Iter 39, Loss: 12689.7851562, Throughput: 169.871327 im/s
Epoch 9, Train Loss: 12202.6962891, Time: 29.3871s, Throughput: 169.870303 im/s
Epoch 10, Iter 39, Loss: 12255.8183594, Throughput: 170.196934 im/s
Epoch 10, Train Loss: 11947.1347656, Time: 29.3309s, Throughput: 170.195763 im/s
Epoch 11, Iter 39, Loss: 12143.2519531, Throughput: 170.017815 im/s
Epoch 11, Train Loss: 11680.9326172, Time: 29.3618s, Throughput: 170.016825 im/s
Epoch 12, Iter 39, Loss: 12196.1152344, Throughput: 170.252032 im/s
Epoch 12, Train Loss: 11590.6181641, Time: 29.3214s, Throughput: 170.250946 im/s
Epoch 13, Iter 39, Loss: 10892.0332031, Throughput: 170.345764 im/s
Epoch 13, Train Loss: 11460.9755859, Time: 29.3053s, Throughput: 170.344717 im/s
Epoch 14, Iter 39, Loss: 10178.1474609, Throughput: 170.437133 im/s
Epoch 14, Train Loss: 11286.6337891, Time: 29.2896s, Throughput: 170.436062 im/s
Epoch 15, Iter 39, Loss: 10391.2841797, Throughput: 169.815586 im/s
Epoch 15, Train Loss: 11060.6142578, Time: 29.3968s, Throughput: 169.814678 im/s
Epoch 16, Iter 39, Loss: 10502.8964844, Throughput: 170.403970 im/s
Epoch 16, Train Loss: 10928.9121094, Time: 29.2953s, Throughput: 170.402883 im/s
Epoch 17, Iter 39, Loss: 10510.7734375, Throughput: 169.904438 im/s
Epoch 17, Train Loss: 10821.7812500, Time: 29.3814s, Throughput: 169.903455 im/s
Epoch 18, Iter 39, Loss: 10671.1699219, Throughput: 169.556970 im/s
Epoch 18, Train Loss: 10754.7294922, Time: 29.4416s, Throughput: 169.555830 im/s
Epoch 19, Iter 39, Loss: 11157.9843750, Throughput: 168.456409 im/s
Epoch 19, Train Loss: 10733.4726562, Time: 29.6340s, Throughput: 168.455158 im/s
Epoch 20, Iter 39, Loss: 9919.9218750, Throughput: 170.044745 im/s
Epoch 20, Train Loss: 10745.3857422, Time: 29.3572s, Throughput: 170.043505 im/s
Epoch 21, Iter 39, Loss: 10824.5976562, Throughput: 170.634831 im/s
Epoch 21, Train Loss: 10513.7919922, Time: 29.2556s, Throughput: 170.633752 im/s
Epoch 22, Iter 39, Loss: 10681.4951172, Throughput: 170.095384 im/s
Epoch 22, Train Loss: 10461.4814453, Time: 29.3484s, Throughput: 170.094445 im/s
Epoch 23, Iter 39, Loss: 10480.0253906, Throughput: 169.911296 im/s
Epoch 23, Train Loss: 10353.4082031, Time: 29.3802s, Throughput: 169.910249 im/s
Epoch 24, Iter 39, Loss: 10725.0117188, Throughput: 170.266363 im/s
Epoch 24, Train Loss: 10360.6679688, Time: 29.3189s, Throughput: 170.265405 im/s
Epoch 25, Iter 39, Loss: 10369.4589844, Throughput: 169.551931 im/s
Epoch 25, Train Loss: 10220.8066406, Time: 29.4425s, Throughput: 169.550629 im/s
Epoch 26, Iter 39, Loss: 10036.3476562, Throughput: 170.795555 im/s
Epoch 26, Train Loss: 10231.8154297, Time: 29.2281s, Throughput: 170.794613 im/s
Epoch 27, Iter 39, Loss: 9944.4882812, Throughput: 170.369884 im/s
Epoch 27, Train Loss: 10120.5244141, Time: 29.3011s, Throughput: 170.368757 im/s
Epoch 28, Iter 39, Loss: 10080.6816406, Throughput: 170.133407 im/s
Epoch 28, Train Loss: 10055.7187500, Time: 29.3419s, Throughput: 170.132369 im/s
Epoch 29, Iter 39, Loss: 9533.4521484, Throughput: 170.061626 im/s
Epoch 29, Train Loss: 10067.4921875, Time: 29.3542s, Throughput: 170.060566 im/s
Epoch 30, Iter 39, Loss: 9892.1552734, Throughput: 170.341707 im/s
Epoch 30, Train Loss: 9885.3017578, Time: 29.3060s, Throughput: 170.340725 im/s
Epoch 31, Iter 39, Loss: 9547.0136719, Throughput: 170.345909 im/s
Epoch 31, Train Loss: 9900.1269531, Time: 29.3053s, Throughput: 170.344770 im/s
Epoch 32, Iter 39, Loss: 9385.5097656, Throughput: 170.410846 im/s
Epoch 32, Train Loss: 9864.2539062, Time: 29.2941s, Throughput: 170.409484 im/s
Epoch 33, Iter 39, Loss: 9314.4218750, Throughput: 170.462973 im/s
Epoch 33, Train Loss: 9762.7128906, Time: 29.2851s, Throughput: 170.461932 im/s
Epoch 34, Iter 39, Loss: 9468.4208984, Throughput: 169.896903 im/s
Epoch 34, Train Loss: 9804.9619141, Time: 29.3827s, Throughput: 169.895725 im/s
Epoch 35, Iter 39, Loss: 10214.1103516, Throughput: 169.157739 im/s
Epoch 35, Train Loss: 9828.8828125, Time: 29.5111s, Throughput: 169.156587 im/s
Epoch 36, Iter 39, Loss: 9920.0625000, Throughput: 169.242771 im/s
Epoch 36, Train Loss: 9712.5478516, Time: 29.4963s, Throughput: 169.241699 im/s
Epoch 37, Iter 39, Loss: 9862.1416016, Throughput: 169.483625 im/s
Epoch 37, Train Loss: 9723.0625000, Time: 29.4543s, Throughput: 169.482658 im/s
Epoch 38, Iter 39, Loss: 10161.5810547, Throughput: 169.944644 im/s
Epoch 38, Train Loss: 9770.3632812, Time: 29.3745s, Throughput: 169.943475 im/s
Epoch 39, Iter 39, Loss: 9386.4121094, Throughput: 169.821288 im/s
Epoch 39, Train Loss: 9612.5722656, Time: 29.3958s, Throughput: 169.820167 im/s
Epoch 40, Iter 39, Loss: 9577.4394531, Throughput: 169.943885 im/s
Epoch 40, Train Loss: 9600.2031250, Time: 29.3746s, Throughput: 169.942803 im/s
Epoch 41, Iter 39, Loss: 9254.4541016, Throughput: 168.032244 im/s
Epoch 41, Train Loss: 9556.8007812, Time: 29.7088s, Throughput: 168.031293 im/s
Epoch 42, Iter 39, Loss: 9297.2070312, Throughput: 169.784992 im/s
Epoch 42, Train Loss: 9499.0000000, Time: 29.4021s, Throughput: 169.783797 im/s
Epoch 43, Iter 39, Loss: 9352.5654297, Throughput: 169.207600 im/s
Epoch 43, Train Loss: 9475.6201172, Time: 29.5024s, Throughput: 169.206333 im/s
Epoch 44, Iter 39, Loss: 8934.7089844, Throughput: 169.197838 im/s
Epoch 44, Train Loss: 9455.9912109, Time: 29.5041s, Throughput: 169.196771 im/s
Epoch 45, Iter 39, Loss: 9479.8681641, Throughput: 169.558899 im/s
Epoch 45, Train Loss: 9394.1074219, Time: 29.4413s, Throughput: 169.557823 im/s
Epoch 46, Iter 39, Loss: 10209.1582031, Throughput: 169.789104 im/s
Epoch 46, Train Loss: 9370.0468750, Time: 29.4014s, Throughput: 169.788122 im/s
Epoch 47, Iter 39, Loss: 9983.4531250, Throughput: 169.316975 im/s
Epoch 47, Train Loss: 9385.4130859, Time: 29.4834s, Throughput: 169.315821 im/s
Epoch 48, Iter 39, Loss: 9288.8222656, Throughput: 169.024276 im/s
Epoch 48, Train Loss: 9343.0107422, Time: 29.5344s, Throughput: 169.023047 im/s
Epoch 49, Iter 39, Loss: 9703.0449219, Throughput: 169.277721 im/s
Epoch 49, Train Loss: 9326.3339844, Time: 29.4902s, Throughput: 169.276401 im/s
Epoch 50, Iter 39, Loss: 9441.0390625, Throughput: 169.740319 im/s
Epoch 50, Train Loss: 9377.4326172, Time: 29.4098s, Throughput: 169.739275 im/s

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
