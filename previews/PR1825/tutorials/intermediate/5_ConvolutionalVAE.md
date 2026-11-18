---
url: /previews/PR1825/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24528.3867188, Throughput: 20.895492 im/s
Epoch 1, Train Loss: 39522.4687500, Time: 239.0993s, Throughput: 20.878353 im/s
Epoch 2, Iter 39, Loss: 17567.9335938, Throughput: 123.352008 im/s
Epoch 2, Train Loss: 20205.1777344, Time: 40.4697s, Throughput: 123.351508 im/s
Epoch 3, Iter 39, Loss: 15772.7265625, Throughput: 123.231726 im/s
Epoch 3, Train Loss: 16607.9101562, Time: 40.5092s, Throughput: 123.231138 im/s
Epoch 4, Iter 39, Loss: 14294.6367188, Throughput: 123.757753 im/s
Epoch 4, Train Loss: 15042.5019531, Time: 40.3370s, Throughput: 123.757317 im/s
Epoch 5, Iter 39, Loss: 13548.2324219, Throughput: 122.824443 im/s
Epoch 5, Train Loss: 14069.1318359, Time: 40.6435s, Throughput: 122.824020 im/s
Epoch 6, Iter 39, Loss: 12725.4843750, Throughput: 122.571957 im/s
Epoch 6, Train Loss: 13465.0078125, Time: 40.7273s, Throughput: 122.571479 im/s
Epoch 7, Iter 39, Loss: 13045.9072266, Throughput: 123.464513 im/s
Epoch 7, Train Loss: 12936.4316406, Time: 40.4328s, Throughput: 123.464048 im/s
Epoch 8, Iter 39, Loss: 12007.4335938, Throughput: 124.017099 im/s
Epoch 8, Train Loss: 12645.3164062, Time: 40.2527s, Throughput: 124.016507 im/s
Epoch 9, Iter 39, Loss: 11855.8398438, Throughput: 123.647911 im/s
Epoch 9, Train Loss: 12287.6025391, Time: 40.3729s, Throughput: 123.647304 im/s
Epoch 10, Iter 39, Loss: 12435.5058594, Throughput: 123.568476 im/s
Epoch 10, Train Loss: 12096.8242188, Time: 40.3988s, Throughput: 123.567907 im/s
Epoch 11, Iter 39, Loss: 11304.7783203, Throughput: 123.730416 im/s
Epoch 11, Train Loss: 11761.6425781, Time: 40.3459s, Throughput: 123.729977 im/s
Epoch 12, Iter 39, Loss: 11425.8730469, Throughput: 123.604344 im/s
Epoch 12, Train Loss: 11537.9677734, Time: 40.3871s, Throughput: 123.603845 im/s
Epoch 13, Iter 39, Loss: 11277.8886719, Throughput: 123.506330 im/s
Epoch 13, Train Loss: 11335.4824219, Time: 40.4192s, Throughput: 123.505728 im/s
Epoch 14, Iter 39, Loss: 11033.5332031, Throughput: 123.591390 im/s
Epoch 14, Train Loss: 11175.2177734, Time: 40.3913s, Throughput: 123.590880 im/s
Epoch 15, Iter 39, Loss: 10809.5703125, Throughput: 123.540055 im/s
Epoch 15, Train Loss: 11133.8710938, Time: 40.4081s, Throughput: 123.539587 im/s
Epoch 16, Iter 39, Loss: 12125.8300781, Throughput: 123.380227 im/s
Epoch 16, Train Loss: 11040.2011719, Time: 40.4605s, Throughput: 123.379666 im/s
Epoch 17, Iter 39, Loss: 11808.9150391, Throughput: 123.396967 im/s
Epoch 17, Train Loss: 10937.3212891, Time: 40.4550s, Throughput: 123.396397 im/s
Epoch 18, Iter 39, Loss: 10838.4511719, Throughput: 123.403116 im/s
Epoch 18, Train Loss: 10839.0126953, Time: 40.4530s, Throughput: 123.402534 im/s
Epoch 19, Iter 39, Loss: 10006.8242188, Throughput: 123.248983 im/s
Epoch 19, Train Loss: 10787.5761719, Time: 40.5035s, Throughput: 123.248462 im/s
Epoch 20, Iter 39, Loss: 10674.3125000, Throughput: 123.351785 im/s
Epoch 20, Train Loss: 10628.4033203, Time: 40.4698s, Throughput: 123.351236 im/s
Epoch 21, Iter 39, Loss: 10414.5751953, Throughput: 123.214531 im/s
Epoch 21, Train Loss: 10473.4589844, Time: 40.5149s, Throughput: 123.214029 im/s
Epoch 22, Iter 39, Loss: 10153.2792969, Throughput: 123.603454 im/s
Epoch 22, Train Loss: 10435.5683594, Time: 40.3874s, Throughput: 123.602952 im/s
Epoch 23, Iter 39, Loss: 10970.8232422, Throughput: 123.410081 im/s
Epoch 23, Train Loss: 10483.8906250, Time: 40.4542s, Throughput: 123.398812 im/s
Epoch 24, Iter 39, Loss: 10304.6083984, Throughput: 123.150421 im/s
Epoch 24, Train Loss: 10374.1044922, Time: 40.5360s, Throughput: 123.149856 im/s
Epoch 25, Iter 39, Loss: 10448.4316406, Throughput: 122.245742 im/s
Epoch 25, Train Loss: 10250.2070312, Time: 40.8360s, Throughput: 122.245165 im/s
Epoch 26, Iter 39, Loss: 10675.5947266, Throughput: 123.675729 im/s
Epoch 26, Train Loss: 10227.4375000, Time: 40.3638s, Throughput: 123.675116 im/s
Epoch 27, Iter 39, Loss: 9262.5693359, Throughput: 123.789436 im/s
Epoch 27, Train Loss: 10162.7783203, Time: 40.3267s, Throughput: 123.788892 im/s
Epoch 28, Iter 39, Loss: 10470.8144531, Throughput: 123.672567 im/s
Epoch 28, Train Loss: 10086.0087891, Time: 40.3648s, Throughput: 123.672064 im/s
Epoch 29, Iter 39, Loss: 10066.4453125, Throughput: 122.892227 im/s
Epoch 29, Train Loss: 10078.5156250, Time: 40.6211s, Throughput: 122.891806 im/s
Epoch 30, Iter 39, Loss: 10815.3671875, Throughput: 122.490588 im/s
Epoch 30, Train Loss: 9972.1005859, Time: 40.7543s, Throughput: 122.490097 im/s
Epoch 31, Iter 39, Loss: 9758.8623047, Throughput: 123.035196 im/s
Epoch 31, Train Loss: 9945.8828125, Time: 40.5740s, Throughput: 123.034549 im/s
Epoch 32, Iter 39, Loss: 10112.5244141, Throughput: 122.775602 im/s
Epoch 32, Train Loss: 9862.0527344, Time: 40.6597s, Throughput: 122.775049 im/s
Epoch 33, Iter 39, Loss: 10281.6611328, Throughput: 122.638063 im/s
Epoch 33, Train Loss: 9868.6279297, Time: 40.7053s, Throughput: 122.637617 im/s
Epoch 34, Iter 39, Loss: 10932.9082031, Throughput: 123.011875 im/s
Epoch 34, Train Loss: 9803.4794922, Time: 40.5817s, Throughput: 123.011196 im/s
Epoch 35, Iter 39, Loss: 9594.3769531, Throughput: 123.585723 im/s
Epoch 35, Train Loss: 9848.9453125, Time: 40.3941s, Throughput: 123.582462 im/s
Epoch 36, Iter 39, Loss: 9708.1250000, Throughput: 123.619759 im/s
Epoch 36, Train Loss: 9734.9384766, Time: 40.3821s, Throughput: 123.619217 im/s
Epoch 37, Iter 39, Loss: 9618.1035156, Throughput: 123.634470 im/s
Epoch 37, Train Loss: 9689.9589844, Time: 40.3772s, Throughput: 123.634057 im/s
Epoch 38, Iter 39, Loss: 10156.8798828, Throughput: 123.440516 im/s
Epoch 38, Train Loss: 9715.2392578, Time: 40.4407s, Throughput: 123.439866 im/s
Epoch 39, Iter 39, Loss: 9831.3837891, Throughput: 123.508835 im/s
Epoch 39, Train Loss: 9658.8339844, Time: 40.4183s, Throughput: 123.508316 im/s
Epoch 40, Iter 39, Loss: 9337.0517578, Throughput: 123.372408 im/s
Epoch 40, Train Loss: 9693.6699219, Time: 40.4630s, Throughput: 123.371844 im/s
Epoch 41, Iter 39, Loss: 9401.3554688, Throughput: 123.031153 im/s
Epoch 41, Train Loss: 9568.2802734, Time: 40.5754s, Throughput: 123.030144 im/s
Epoch 42, Iter 39, Loss: 9392.2314453, Throughput: 123.185427 im/s
Epoch 42, Train Loss: 9534.2304688, Time: 40.5244s, Throughput: 123.184953 im/s
Epoch 43, Iter 39, Loss: 9342.3710938, Throughput: 123.310297 im/s
Epoch 43, Train Loss: 9495.9716797, Time: 40.4834s, Throughput: 123.309746 im/s
Epoch 44, Iter 39, Loss: 9532.3144531, Throughput: 123.455618 im/s
Epoch 44, Train Loss: 9518.5712891, Time: 40.4358s, Throughput: 123.455090 im/s
Epoch 45, Iter 39, Loss: 9602.0156250, Throughput: 123.313289 im/s
Epoch 45, Train Loss: 9452.6025391, Time: 40.4824s, Throughput: 123.312832 im/s
Epoch 46, Iter 39, Loss: 9396.1386719, Throughput: 123.690683 im/s
Epoch 46, Train Loss: 9351.1611328, Time: 40.3589s, Throughput: 123.690153 im/s
Epoch 47, Iter 39, Loss: 8599.8017578, Throughput: 123.844540 im/s
Epoch 47, Train Loss: 9444.3925781, Time: 40.3088s, Throughput: 123.843959 im/s
Epoch 48, Iter 39, Loss: 9887.9414062, Throughput: 123.719303 im/s
Epoch 48, Train Loss: 9339.6123047, Time: 40.3496s, Throughput: 123.718763 im/s
Epoch 49, Iter 39, Loss: 9389.8339844, Throughput: 123.177679 im/s
Epoch 49, Train Loss: 9395.9521484, Time: 40.5270s, Throughput: 123.177159 im/s
Epoch 50, Iter 39, Loss: 9954.9355469, Throughput: 124.494801 im/s
Epoch 50, Train Loss: 9409.0302734, Time: 40.0982s, Throughput: 124.494220 im/s

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
