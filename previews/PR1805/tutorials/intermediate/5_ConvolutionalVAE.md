---
url: /previews/PR1805/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23993.0839844, Throughput: 18.380077 im/s
Epoch 1, Train Loss: 39610.5507812, Time: 271.7901s, Throughput: 18.367116 im/s
Epoch 2, Iter 39, Loss: 17850.8437500, Throughput: 122.537339 im/s
Epoch 2, Train Loss: 20200.3710938, Time: 40.7388s, Throughput: 122.536761 im/s
Epoch 3, Iter 39, Loss: 15908.2001953, Throughput: 123.935337 im/s
Epoch 3, Train Loss: 16731.5156250, Time: 40.2792s, Throughput: 123.934802 im/s
Epoch 4, Iter 39, Loss: 15034.6582031, Throughput: 123.211477 im/s
Epoch 4, Train Loss: 15148.6025391, Time: 40.5159s, Throughput: 123.210854 im/s
Epoch 5, Iter 39, Loss: 14532.5839844, Throughput: 123.574203 im/s
Epoch 5, Train Loss: 14036.6396484, Time: 40.3970s, Throughput: 123.573612 im/s
Epoch 6, Iter 39, Loss: 12915.1308594, Throughput: 123.710571 im/s
Epoch 6, Train Loss: 13513.4970703, Time: 40.3525s, Throughput: 123.709893 im/s
Epoch 7, Iter 39, Loss: 13480.0917969, Throughput: 123.660953 im/s
Epoch 7, Train Loss: 13052.1777344, Time: 40.3686s, Throughput: 123.660355 im/s
Epoch 8, Iter 39, Loss: 12635.4121094, Throughput: 124.233071 im/s
Epoch 8, Train Loss: 12494.1943359, Time: 40.1827s, Throughput: 124.232515 im/s
Epoch 9, Iter 39, Loss: 11917.4335938, Throughput: 123.945427 im/s
Epoch 9, Train Loss: 12292.1582031, Time: 40.2759s, Throughput: 123.944944 im/s
Epoch 10, Iter 39, Loss: 12653.7480469, Throughput: 123.862900 im/s
Epoch 10, Train Loss: 12010.3867188, Time: 40.3028s, Throughput: 123.862408 im/s
Epoch 11, Iter 39, Loss: 11545.1787109, Throughput: 124.345873 im/s
Epoch 11, Train Loss: 11747.4667969, Time: 40.1463s, Throughput: 124.345303 im/s
Epoch 12, Iter 39, Loss: 11375.6621094, Throughput: 124.331602 im/s
Epoch 12, Train Loss: 11567.1210938, Time: 40.1509s, Throughput: 124.331082 im/s
Epoch 13, Iter 39, Loss: 11180.5537109, Throughput: 124.567870 im/s
Epoch 13, Train Loss: 11365.6669922, Time: 40.0747s, Throughput: 124.567361 im/s
Epoch 14, Iter 39, Loss: 9675.4335938, Throughput: 124.250276 im/s
Epoch 14, Train Loss: 11319.8144531, Time: 40.1771s, Throughput: 124.249744 im/s
Epoch 15, Iter 39, Loss: 10984.2509766, Throughput: 124.529950 im/s
Epoch 15, Train Loss: 11188.2265625, Time: 40.0869s, Throughput: 124.529438 im/s
Epoch 16, Iter 39, Loss: 11034.6503906, Throughput: 124.383359 im/s
Epoch 16, Train Loss: 11003.7617188, Time: 40.1342s, Throughput: 124.382848 im/s
Epoch 17, Iter 39, Loss: 10569.1230469, Throughput: 124.524312 im/s
Epoch 17, Train Loss: 10916.6914062, Time: 40.0887s, Throughput: 124.523765 im/s
Epoch 18, Iter 39, Loss: 11032.2412109, Throughput: 124.443603 im/s
Epoch 18, Train Loss: 10764.9082031, Time: 40.1148s, Throughput: 124.442954 im/s
Epoch 19, Iter 39, Loss: 11212.0937500, Throughput: 124.200605 im/s
Epoch 19, Train Loss: 10704.5546875, Time: 40.1932s, Throughput: 124.200058 im/s
Epoch 20, Iter 39, Loss: 10096.6953125, Throughput: 124.310453 im/s
Epoch 20, Train Loss: 10543.1894531, Time: 40.1577s, Throughput: 124.309844 im/s
Epoch 21, Iter 39, Loss: 11087.5585938, Throughput: 124.426853 im/s
Epoch 21, Train Loss: 10479.9052734, Time: 40.1201s, Throughput: 124.426354 im/s
Epoch 22, Iter 39, Loss: 11345.8134766, Throughput: 124.065643 im/s
Epoch 22, Train Loss: 10530.8994141, Time: 40.2370s, Throughput: 124.065021 im/s
Epoch 23, Iter 39, Loss: 10090.9248047, Throughput: 124.846305 im/s
Epoch 23, Train Loss: 10421.8505859, Time: 39.9853s, Throughput: 124.845747 im/s
Epoch 24, Iter 39, Loss: 10559.7480469, Throughput: 124.215281 im/s
Epoch 24, Train Loss: 10284.8173828, Time: 40.1885s, Throughput: 124.214620 im/s
Epoch 25, Iter 39, Loss: 10582.1103516, Throughput: 124.458589 im/s
Epoch 25, Train Loss: 10298.5283203, Time: 40.1099s, Throughput: 124.458015 im/s
Epoch 26, Iter 39, Loss: 10598.6240234, Throughput: 124.415441 im/s
Epoch 26, Train Loss: 10181.2744141, Time: 40.1238s, Throughput: 124.414924 im/s
Epoch 27, Iter 39, Loss: 10136.6279297, Throughput: 124.078354 im/s
Epoch 27, Train Loss: 10174.3066406, Time: 40.2328s, Throughput: 124.077759 im/s
Epoch 28, Iter 39, Loss: 10021.9492188, Throughput: 124.092629 im/s
Epoch 28, Train Loss: 10138.3769531, Time: 40.2282s, Throughput: 124.092012 im/s
Epoch 29, Iter 39, Loss: 10447.7431641, Throughput: 123.227740 im/s
Epoch 29, Train Loss: 10086.7421875, Time: 40.5105s, Throughput: 123.227235 im/s
Epoch 30, Iter 39, Loss: 10527.1425781, Throughput: 124.619414 im/s
Epoch 30, Train Loss: 10058.2187500, Time: 40.0581s, Throughput: 124.618860 im/s
Epoch 31, Iter 39, Loss: 9834.9736328, Throughput: 124.416167 im/s
Epoch 31, Train Loss: 9878.6894531, Time: 40.1236s, Throughput: 124.415639 im/s
Epoch 32, Iter 39, Loss: 10155.7871094, Throughput: 124.571029 im/s
Epoch 32, Train Loss: 9902.7988281, Time: 40.0738s, Throughput: 124.570292 im/s
Epoch 33, Iter 39, Loss: 10201.6503906, Throughput: 124.744853 im/s
Epoch 33, Train Loss: 9830.0878906, Time: 40.0179s, Throughput: 124.744193 im/s
Epoch 34, Iter 39, Loss: 9172.6953125, Throughput: 124.972893 im/s
Epoch 34, Train Loss: 9760.0107422, Time: 39.9449s, Throughput: 124.972283 im/s
Epoch 35, Iter 39, Loss: 10030.2851562, Throughput: 125.057287 im/s
Epoch 35, Train Loss: 9816.4921875, Time: 39.9179s, Throughput: 125.056763 im/s
Epoch 36, Iter 39, Loss: 9436.9960938, Throughput: 124.755541 im/s
Epoch 36, Train Loss: 9755.9873047, Time: 40.0144s, Throughput: 124.754951 im/s
Epoch 37, Iter 39, Loss: 9624.7500000, Throughput: 124.380418 im/s
Epoch 37, Train Loss: 9661.3857422, Time: 40.1351s, Throughput: 124.379867 im/s
Epoch 38, Iter 39, Loss: 10272.5771484, Throughput: 124.536040 im/s
Epoch 38, Train Loss: 9722.3525391, Time: 40.0849s, Throughput: 124.535533 im/s
Epoch 39, Iter 39, Loss: 9529.5722656, Throughput: 124.566273 im/s
Epoch 39, Train Loss: 9636.4072266, Time: 40.0752s, Throughput: 124.565751 im/s
Epoch 40, Iter 39, Loss: 8910.2275391, Throughput: 124.609494 im/s
Epoch 40, Train Loss: 9600.5966797, Time: 40.0613s, Throughput: 124.608968 im/s
Epoch 41, Iter 39, Loss: 10355.6787109, Throughput: 124.759594 im/s
Epoch 41, Train Loss: 9591.4296875, Time: 40.0131s, Throughput: 124.758994 im/s
Epoch 42, Iter 39, Loss: 10082.2109375, Throughput: 124.572015 im/s
Epoch 42, Train Loss: 9545.4599609, Time: 40.0734s, Throughput: 124.571480 im/s
Epoch 43, Iter 39, Loss: 9422.8974609, Throughput: 124.727060 im/s
Epoch 43, Train Loss: 9459.8681641, Time: 40.0236s, Throughput: 124.726542 im/s
Epoch 44, Iter 39, Loss: 8954.9873047, Throughput: 124.427046 im/s
Epoch 44, Train Loss: 9429.0371094, Time: 40.1201s, Throughput: 124.426521 im/s
Epoch 45, Iter 39, Loss: 9823.9492188, Throughput: 124.491432 im/s
Epoch 45, Train Loss: 9389.2519531, Time: 40.0993s, Throughput: 124.490823 im/s
Epoch 46, Iter 39, Loss: 9718.9580078, Throughput: 124.499853 im/s
Epoch 46, Train Loss: 9412.4482422, Time: 40.0966s, Throughput: 124.499341 im/s
Epoch 47, Iter 39, Loss: 10149.1435547, Throughput: 124.343079 im/s
Epoch 47, Train Loss: 9351.1962891, Time: 40.1472s, Throughput: 124.342562 im/s
Epoch 48, Iter 39, Loss: 9320.2929688, Throughput: 124.368506 im/s
Epoch 48, Train Loss: 9435.0068359, Time: 40.1390s, Throughput: 124.367877 im/s
Epoch 49, Iter 39, Loss: 9650.5585938, Throughput: 124.462194 im/s
Epoch 49, Train Loss: 9338.1816406, Time: 40.1087s, Throughput: 124.461648 im/s
Epoch 50, Iter 39, Loss: 9598.1787109, Throughput: 124.746835 im/s
Epoch 50, Train Loss: 9280.2910156, Time: 40.0173s, Throughput: 124.746193 im/s

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
