---
url: /previews/PR1827/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23236.9121094, Throughput: 19.271445 im/s
Epoch 1, Train Loss: 39547.6484375, Time: 259.2730s, Throughput: 19.253840 im/s
Epoch 2, Iter 39, Loss: 18246.4980469, Throughput: 203.869926 im/s
Epoch 2, Train Loss: 20076.7441406, Time: 24.4864s, Throughput: 203.868560 im/s
Epoch 3, Iter 39, Loss: 16020.6386719, Throughput: 203.913273 im/s
Epoch 3, Train Loss: 16539.5546875, Time: 24.4812s, Throughput: 203.911865 im/s
Epoch 4, Iter 39, Loss: 13573.4638672, Throughput: 204.584414 im/s
Epoch 4, Train Loss: 14956.1767578, Time: 24.4008s, Throughput: 204.583147 im/s
Epoch 5, Iter 39, Loss: 13825.8867188, Throughput: 203.955744 im/s
Epoch 5, Train Loss: 14079.3857422, Time: 24.4761s, Throughput: 203.954445 im/s
Epoch 6, Iter 39, Loss: 12913.7187500, Throughput: 204.088066 im/s
Epoch 6, Train Loss: 13390.4921875, Time: 24.4602s, Throughput: 204.086557 im/s
Epoch 7, Iter 39, Loss: 12876.5468750, Throughput: 203.828580 im/s
Epoch 7, Train Loss: 12844.8251953, Time: 24.4913s, Throughput: 203.827082 im/s
Epoch 8, Iter 39, Loss: 12290.8349609, Throughput: 203.553212 im/s
Epoch 8, Train Loss: 12561.0527344, Time: 24.5244s, Throughput: 203.552008 im/s
Epoch 9, Iter 39, Loss: 11884.0849609, Throughput: 204.189635 im/s
Epoch 9, Train Loss: 12417.1044922, Time: 24.4480s, Throughput: 204.188474 im/s
Epoch 10, Iter 39, Loss: 11991.7998047, Throughput: 204.416235 im/s
Epoch 10, Train Loss: 12005.0419922, Time: 24.4209s, Throughput: 204.414862 im/s
Epoch 11, Iter 39, Loss: 11674.5019531, Throughput: 205.418936 im/s
Epoch 11, Train Loss: 11702.3671875, Time: 24.3017s, Throughput: 205.417499 im/s
Epoch 12, Iter 39, Loss: 12028.8134766, Throughput: 204.854820 im/s
Epoch 12, Train Loss: 11554.9521484, Time: 24.3686s, Throughput: 204.853494 im/s
Epoch 13, Iter 39, Loss: 11187.7099609, Throughput: 205.100420 im/s
Epoch 13, Train Loss: 11438.1748047, Time: 24.3394s, Throughput: 205.099172 im/s
Epoch 14, Iter 39, Loss: 10577.5097656, Throughput: 204.388363 im/s
Epoch 14, Train Loss: 11289.0292969, Time: 24.4242s, Throughput: 204.387226 im/s
Epoch 15, Iter 39, Loss: 10836.4902344, Throughput: 204.741151 im/s
Epoch 15, Train Loss: 11165.0791016, Time: 24.3822s, Throughput: 204.739917 im/s
Epoch 16, Iter 39, Loss: 10880.5419922, Throughput: 204.845104 im/s
Epoch 16, Train Loss: 10936.7226562, Time: 24.3698s, Throughput: 204.843683 im/s
Epoch 17, Iter 39, Loss: 11430.4414062, Throughput: 204.567586 im/s
Epoch 17, Train Loss: 10819.6162109, Time: 24.4028s, Throughput: 204.566379 im/s
Epoch 18, Iter 39, Loss: 11444.2119141, Throughput: 204.528362 im/s
Epoch 18, Train Loss: 10774.1337891, Time: 24.4076s, Throughput: 204.526837 im/s
Epoch 19, Iter 39, Loss: 11211.6455078, Throughput: 204.902936 im/s
Epoch 19, Train Loss: 10784.1855469, Time: 24.3629s, Throughput: 204.901473 im/s
Epoch 20, Iter 39, Loss: 10153.9335938, Throughput: 204.675835 im/s
Epoch 20, Train Loss: 10691.9218750, Time: 24.3900s, Throughput: 204.674424 im/s
Epoch 21, Iter 39, Loss: 9892.2226562, Throughput: 204.765423 im/s
Epoch 21, Train Loss: 10569.9072266, Time: 24.3793s, Throughput: 204.764129 im/s
Epoch 22, Iter 39, Loss: 10799.1757812, Throughput: 201.983828 im/s
Epoch 22, Train Loss: 10454.8085938, Time: 24.7150s, Throughput: 201.982372 im/s
Epoch 23, Iter 39, Loss: 10744.8789062, Throughput: 204.540579 im/s
Epoch 23, Train Loss: 10569.9794922, Time: 24.4061s, Throughput: 204.539247 im/s
Epoch 24, Iter 39, Loss: 10657.6142578, Throughput: 205.368503 im/s
Epoch 24, Train Loss: 10285.7490234, Time: 24.3077s, Throughput: 205.367000 im/s
Epoch 25, Iter 39, Loss: 10272.4003906, Throughput: 205.650545 im/s
Epoch 25, Train Loss: 10225.5332031, Time: 24.2743s, Throughput: 205.649182 im/s
Epoch 26, Iter 39, Loss: 10530.6191406, Throughput: 205.499000 im/s
Epoch 26, Train Loss: 10120.3925781, Time: 24.2922s, Throughput: 205.497681 im/s
Epoch 27, Iter 39, Loss: 10355.0966797, Throughput: 204.670631 im/s
Epoch 27, Train Loss: 10186.3779297, Time: 24.3906s, Throughput: 204.669322 im/s
Epoch 28, Iter 39, Loss: 9882.5546875, Throughput: 204.885711 im/s
Epoch 28, Train Loss: 10098.6279297, Time: 24.3649s, Throughput: 204.884508 im/s
Epoch 29, Iter 39, Loss: 9725.2070312, Throughput: 205.774632 im/s
Epoch 29, Train Loss: 9981.6328125, Time: 24.2597s, Throughput: 205.773328 im/s
Epoch 30, Iter 39, Loss: 9884.6796875, Throughput: 205.146201 im/s
Epoch 30, Train Loss: 10007.5693359, Time: 24.3340s, Throughput: 205.144979 im/s
Epoch 31, Iter 39, Loss: 9760.9335938, Throughput: 204.906552 im/s
Epoch 31, Train Loss: 9938.1210938, Time: 24.3625s, Throughput: 204.905180 im/s
Epoch 32, Iter 39, Loss: 10075.9892578, Throughput: 204.868416 im/s
Epoch 32, Train Loss: 9945.6621094, Time: 24.3670s, Throughput: 204.867079 im/s
Epoch 33, Iter 39, Loss: 10144.9726562, Throughput: 205.091942 im/s
Epoch 33, Train Loss: 9879.3505859, Time: 24.3405s, Throughput: 205.090594 im/s
Epoch 34, Iter 39, Loss: 9968.7128906, Throughput: 204.933823 im/s
Epoch 34, Train Loss: 9789.1494141, Time: 24.3592s, Throughput: 204.932502 im/s
Epoch 35, Iter 39, Loss: 10018.7500000, Throughput: 205.077366 im/s
Epoch 35, Train Loss: 9733.1328125, Time: 24.3422s, Throughput: 205.075934 im/s
Epoch 36, Iter 39, Loss: 9574.5820312, Throughput: 205.129023 im/s
Epoch 36, Train Loss: 9736.4873047, Time: 24.3361s, Throughput: 205.127743 im/s
Epoch 37, Iter 39, Loss: 10833.6357422, Throughput: 205.071333 im/s
Epoch 37, Train Loss: 9702.8017578, Time: 24.3429s, Throughput: 205.070119 im/s
Epoch 38, Iter 39, Loss: 9070.9892578, Throughput: 205.106849 im/s
Epoch 38, Train Loss: 9675.9794922, Time: 24.3387s, Throughput: 205.105543 im/s
Epoch 39, Iter 39, Loss: 10010.4843750, Throughput: 205.565176 im/s
Epoch 39, Train Loss: 9635.9550781, Time: 24.2844s, Throughput: 205.563787 im/s
Epoch 40, Iter 39, Loss: 9752.9394531, Throughput: 205.368005 im/s
Epoch 40, Train Loss: 9652.4257812, Time: 24.3078s, Throughput: 205.366535 im/s
Epoch 41, Iter 39, Loss: 10050.7089844, Throughput: 205.956516 im/s
Epoch 41, Train Loss: 9529.3486328, Time: 24.2383s, Throughput: 205.955183 im/s
Epoch 42, Iter 39, Loss: 9413.5039062, Throughput: 205.342771 im/s
Epoch 42, Train Loss: 9480.4433594, Time: 24.3107s, Throughput: 205.341504 im/s
Epoch 43, Iter 39, Loss: 9733.4072266, Throughput: 204.611866 im/s
Epoch 43, Train Loss: 9497.4511719, Time: 24.3976s, Throughput: 204.610306 im/s
Epoch 44, Iter 39, Loss: 9328.4990234, Throughput: 204.275655 im/s
Epoch 44, Train Loss: 9421.4951172, Time: 24.4378s, Throughput: 204.274058 im/s
Epoch 45, Iter 39, Loss: 9151.0166016, Throughput: 202.185589 im/s
Epoch 45, Train Loss: 9489.2343750, Time: 24.6904s, Throughput: 202.184238 im/s
Epoch 46, Iter 39, Loss: 8922.9433594, Throughput: 203.898739 im/s
Epoch 46, Train Loss: 9446.3017578, Time: 24.4829s, Throughput: 203.897290 im/s
Epoch 47, Iter 39, Loss: 8710.6074219, Throughput: 204.625293 im/s
Epoch 47, Train Loss: 9385.0849609, Time: 24.3960s, Throughput: 204.624044 im/s
Epoch 48, Iter 39, Loss: 9686.2734375, Throughput: 204.535141 im/s
Epoch 48, Train Loss: 9359.1591797, Time: 24.4068s, Throughput: 204.533480 im/s
Epoch 49, Iter 39, Loss: 9095.1542969, Throughput: 204.948985 im/s
Epoch 49, Train Loss: 9302.1621094, Time: 24.3574s, Throughput: 204.947689 im/s
Epoch 50, Iter 39, Loss: 9049.5761719, Throughput: 204.797991 im/s
Epoch 50, Train Loss: 9252.2509766, Time: 24.3754s, Throughput: 204.796681 im/s

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
