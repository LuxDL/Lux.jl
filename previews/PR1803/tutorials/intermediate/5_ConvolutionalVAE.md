---
url: /previews/PR1803/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23763.7792969, Throughput: 13.967528 im/s
Epoch 1, Train Loss: 39327.9218750, Time: 357.6763s, Throughput: 13.956754 im/s
Epoch 2, Iter 39, Loss: 17736.2812500, Throughput: 157.241481 im/s
Epoch 2, Train Loss: 20096.5800781, Time: 31.7475s, Throughput: 157.240599 im/s
Epoch 3, Iter 39, Loss: 15755.1523438, Throughput: 156.827519 im/s
Epoch 3, Train Loss: 16621.0820312, Time: 31.8314s, Throughput: 156.826509 im/s
Epoch 4, Iter 39, Loss: 15146.1992188, Throughput: 157.167633 im/s
Epoch 4, Train Loss: 15046.5517578, Time: 31.7624s, Throughput: 157.166796 im/s
Epoch 5, Iter 39, Loss: 13802.1318359, Throughput: 156.948519 im/s
Epoch 5, Train Loss: 14220.2177734, Time: 31.8068s, Throughput: 156.947606 im/s
Epoch 6, Iter 39, Loss: 14188.8417969, Throughput: 156.851596 im/s
Epoch 6, Train Loss: 13432.9248047, Time: 31.8264s, Throughput: 156.850724 im/s
Epoch 7, Iter 39, Loss: 12257.3554688, Throughput: 156.308596 im/s
Epoch 7, Train Loss: 12911.3564453, Time: 31.9370s, Throughput: 156.307564 im/s
Epoch 8, Iter 39, Loss: 12516.6630859, Throughput: 156.033546 im/s
Epoch 8, Train Loss: 12573.8994141, Time: 31.9933s, Throughput: 156.032620 im/s
Epoch 9, Iter 39, Loss: 12025.6835938, Throughput: 156.013881 im/s
Epoch 9, Train Loss: 12253.4619141, Time: 31.9973s, Throughput: 156.013057 im/s
Epoch 10, Iter 39, Loss: 12016.8261719, Throughput: 156.731358 im/s
Epoch 10, Train Loss: 12008.6699219, Time: 31.8509s, Throughput: 156.730393 im/s
Epoch 11, Iter 39, Loss: 11221.3623047, Throughput: 156.817952 im/s
Epoch 11, Train Loss: 11806.4902344, Time: 31.8332s, Throughput: 156.817198 im/s
Epoch 12, Iter 39, Loss: 11206.1171875, Throughput: 156.605648 im/s
Epoch 12, Train Loss: 11638.2607422, Time: 31.8764s, Throughput: 156.604748 im/s
Epoch 13, Iter 39, Loss: 11731.8398438, Throughput: 154.637647 im/s
Epoch 13, Train Loss: 11498.0166016, Time: 32.2821s, Throughput: 154.636765 im/s
Epoch 14, Iter 39, Loss: 11440.3544922, Throughput: 157.196185 im/s
Epoch 14, Train Loss: 11309.9882812, Time: 31.7567s, Throughput: 157.195343 im/s
Epoch 15, Iter 39, Loss: 10961.7382812, Throughput: 156.716100 im/s
Epoch 15, Train Loss: 11131.4208984, Time: 31.8540s, Throughput: 156.715116 im/s
Epoch 16, Iter 39, Loss: 10857.3066406, Throughput: 156.927594 im/s
Epoch 16, Train Loss: 11129.9169922, Time: 31.8110s, Throughput: 156.926697 im/s
Epoch 17, Iter 39, Loss: 11540.6054688, Throughput: 157.189136 im/s
Epoch 17, Train Loss: 10904.9072266, Time: 31.7581s, Throughput: 157.188155 im/s
Epoch 18, Iter 39, Loss: 11046.7597656, Throughput: 156.698640 im/s
Epoch 18, Train Loss: 10871.9667969, Time: 31.8575s, Throughput: 156.697794 im/s
Epoch 19, Iter 39, Loss: 10747.3046875, Throughput: 157.466878 im/s
Epoch 19, Train Loss: 10740.9843750, Time: 31.7021s, Throughput: 157.465889 im/s
Epoch 20, Iter 39, Loss: 10621.7675781, Throughput: 157.113236 im/s
Epoch 20, Train Loss: 10701.3847656, Time: 31.7734s, Throughput: 157.112365 im/s
Epoch 21, Iter 39, Loss: 10414.8242188, Throughput: 157.296174 im/s
Epoch 21, Train Loss: 10620.2529297, Time: 31.7365s, Throughput: 157.295178 im/s
Epoch 22, Iter 39, Loss: 10210.5263672, Throughput: 157.403974 im/s
Epoch 22, Train Loss: 10470.0478516, Time: 31.7147s, Throughput: 157.403136 im/s
Epoch 23, Iter 39, Loss: 10366.0683594, Throughput: 157.572450 im/s
Epoch 23, Train Loss: 10456.5996094, Time: 31.6808s, Throughput: 157.571575 im/s
Epoch 24, Iter 39, Loss: 10805.1191406, Throughput: 157.321654 im/s
Epoch 24, Train Loss: 10412.9335938, Time: 31.7315s, Throughput: 157.320143 im/s
Epoch 25, Iter 39, Loss: 10231.4609375, Throughput: 157.507653 im/s
Epoch 25, Train Loss: 10266.7734375, Time: 31.6939s, Throughput: 157.506828 im/s
Epoch 26, Iter 39, Loss: 10046.8798828, Throughput: 157.698152 im/s
Epoch 26, Train Loss: 10269.2998047, Time: 31.6556s, Throughput: 157.697261 im/s
Epoch 27, Iter 39, Loss: 10824.8652344, Throughput: 157.574315 im/s
Epoch 27, Train Loss: 10220.2861328, Time: 31.6805s, Throughput: 157.573425 im/s
Epoch 28, Iter 39, Loss: 10480.9423828, Throughput: 157.760569 im/s
Epoch 28, Train Loss: 10240.6875000, Time: 31.6431s, Throughput: 157.759711 im/s
Epoch 29, Iter 39, Loss: 9685.9169922, Throughput: 157.127873 im/s
Epoch 29, Train Loss: 10037.2988281, Time: 31.7705s, Throughput: 157.126998 im/s
Epoch 30, Iter 39, Loss: 9997.7597656, Throughput: 157.254895 im/s
Epoch 30, Train Loss: 10062.1572266, Time: 31.7448s, Throughput: 157.254053 im/s
Epoch 31, Iter 39, Loss: 9880.8115234, Throughput: 157.053016 im/s
Epoch 31, Train Loss: 10044.8378906, Time: 31.7856s, Throughput: 157.052157 im/s
Epoch 32, Iter 39, Loss: 9925.8486328, Throughput: 157.188147 im/s
Epoch 32, Train Loss: 9933.9531250, Time: 31.7583s, Throughput: 157.187315 im/s
Epoch 33, Iter 39, Loss: 9848.7646484, Throughput: 156.902365 im/s
Epoch 33, Train Loss: 9881.1416016, Time: 31.8162s, Throughput: 156.901443 im/s
Epoch 34, Iter 39, Loss: 10505.8437500, Throughput: 157.386193 im/s
Epoch 34, Train Loss: 9854.2919922, Time: 31.7183s, Throughput: 157.385355 im/s
Epoch 35, Iter 39, Loss: 10247.9316406, Throughput: 155.568414 im/s
Epoch 35, Train Loss: 9838.1660156, Time: 32.0890s, Throughput: 155.567503 im/s
Epoch 36, Iter 39, Loss: 10126.7138672, Throughput: 156.815031 im/s
Epoch 36, Train Loss: 9787.8671875, Time: 31.8339s, Throughput: 156.814091 im/s
Epoch 37, Iter 39, Loss: 10295.6269531, Throughput: 156.516926 im/s
Epoch 37, Train Loss: 9806.6699219, Time: 31.8945s, Throughput: 156.516087 im/s
Epoch 38, Iter 39, Loss: 9184.5527344, Throughput: 156.348373 im/s
Epoch 38, Train Loss: 9705.0332031, Time: 31.9289s, Throughput: 156.347429 im/s
Epoch 39, Iter 39, Loss: 10277.2871094, Throughput: 156.665722 im/s
Epoch 39, Train Loss: 9623.2832031, Time: 31.8642s, Throughput: 156.664851 im/s
Epoch 40, Iter 39, Loss: 9702.7900391, Throughput: 156.795911 im/s
Epoch 40, Train Loss: 9658.0185547, Time: 31.8378s, Throughput: 156.794995 im/s
Epoch 41, Iter 39, Loss: 9196.7246094, Throughput: 156.971294 im/s
Epoch 41, Train Loss: 9629.2792969, Time: 31.8022s, Throughput: 156.970193 im/s
Epoch 42, Iter 39, Loss: 9688.6835938, Throughput: 157.077448 im/s
Epoch 42, Train Loss: 9552.6181641, Time: 31.7807s, Throughput: 157.076479 im/s
Epoch 43, Iter 39, Loss: 10105.7910156, Throughput: 156.354608 im/s
Epoch 43, Train Loss: 9610.2832031, Time: 31.9276s, Throughput: 156.353751 im/s
Epoch 44, Iter 39, Loss: 8860.9033203, Throughput: 157.457604 im/s
Epoch 44, Train Loss: 9510.8544922, Time: 31.7039s, Throughput: 157.456766 im/s
Epoch 45, Iter 39, Loss: 9846.9628906, Throughput: 157.282616 im/s
Epoch 45, Train Loss: 9561.3945312, Time: 31.7392s, Throughput: 157.281744 im/s
Epoch 46, Iter 39, Loss: 9430.7988281, Throughput: 157.063363 im/s
Epoch 46, Train Loss: 9480.8564453, Time: 31.7835s, Throughput: 157.062449 im/s
Epoch 47, Iter 39, Loss: 9700.8808594, Throughput: 156.657983 im/s
Epoch 47, Train Loss: 9390.6894531, Time: 31.8658s, Throughput: 156.656990 im/s
Epoch 48, Iter 39, Loss: 9663.2548828, Throughput: 157.246988 im/s
Epoch 48, Train Loss: 9413.9208984, Time: 31.7464s, Throughput: 157.245957 im/s
Epoch 49, Iter 39, Loss: 9545.7910156, Throughput: 156.779778 im/s
Epoch 49, Train Loss: 9408.3691406, Time: 31.8410s, Throughput: 156.778833 im/s
Epoch 50, Iter 39, Loss: 9130.4814453, Throughput: 156.641516 im/s
Epoch 50, Train Loss: 9405.4765625, Time: 31.8691s, Throughput: 156.640681 im/s

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
