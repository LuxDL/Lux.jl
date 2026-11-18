---
url: /previews/PR1846/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 24218.7480469, Throughput: 24.550902 im/s
Epoch 1, Train Loss: 39532.3945312, Time: 203.4939s, Throughput: 24.531448 im/s
Epoch 2, Iter 39, Loss: 17895.5703125, Throughput: 128.108794 im/s
Epoch 2, Train Loss: 20152.3496094, Time: 38.9670s, Throughput: 128.108242 im/s
Epoch 3, Iter 39, Loss: 15834.4316406, Throughput: 128.749142 im/s
Epoch 3, Train Loss: 16626.8847656, Time: 38.7732s, Throughput: 128.748657 im/s
Epoch 4, Iter 39, Loss: 15047.2568359, Throughput: 129.230775 im/s
Epoch 4, Train Loss: 15148.5351562, Time: 38.6287s, Throughput: 129.230239 im/s
Epoch 5, Iter 39, Loss: 13946.6503906, Throughput: 129.304711 im/s
Epoch 5, Train Loss: 14269.7695312, Time: 38.6066s, Throughput: 129.304162 im/s
Epoch 6, Iter 39, Loss: 14651.9082031, Throughput: 127.137483 im/s
Epoch 6, Train Loss: 13633.0253906, Time: 39.2647s, Throughput: 127.136994 im/s
Epoch 7, Iter 39, Loss: 13069.7441406, Throughput: 121.974253 im/s
Epoch 7, Train Loss: 13069.2519531, Time: 40.9268s, Throughput: 121.973719 im/s
Epoch 8, Iter 39, Loss: 12854.1914062, Throughput: 122.914616 im/s
Epoch 8, Train Loss: 12759.4326172, Time: 40.6137s, Throughput: 122.914147 im/s
Epoch 9, Iter 39, Loss: 12007.9335938, Throughput: 129.120370 im/s
Epoch 9, Train Loss: 12423.4033203, Time: 38.6618s, Throughput: 129.119830 im/s
Epoch 10, Iter 39, Loss: 12662.4423828, Throughput: 129.804459 im/s
Epoch 10, Train Loss: 12166.6181641, Time: 38.4580s, Throughput: 129.803881 im/s
Epoch 11, Iter 39, Loss: 12290.4248047, Throughput: 129.749789 im/s
Epoch 11, Train Loss: 12044.2001953, Time: 38.4742s, Throughput: 129.749246 im/s
Epoch 12, Iter 39, Loss: 11983.5048828, Throughput: 128.298930 im/s
Epoch 12, Train Loss: 11705.1777344, Time: 38.9093s, Throughput: 128.298392 im/s
Epoch 13, Iter 39, Loss: 11534.3828125, Throughput: 128.313479 im/s
Epoch 13, Train Loss: 11531.7421875, Time: 38.9049s, Throughput: 128.313008 im/s
Epoch 14, Iter 39, Loss: 11376.8232422, Throughput: 127.946079 im/s
Epoch 14, Train Loss: 11319.5683594, Time: 39.0166s, Throughput: 127.945495 im/s
Epoch 15, Iter 39, Loss: 11523.2128906, Throughput: 129.239868 im/s
Epoch 15, Train Loss: 11324.3857422, Time: 38.6260s, Throughput: 129.239382 im/s
Epoch 16, Iter 39, Loss: 10514.3906250, Throughput: 129.259658 im/s
Epoch 16, Train Loss: 11100.9453125, Time: 38.6201s, Throughput: 129.259089 im/s
Epoch 17, Iter 39, Loss: 10948.8457031, Throughput: 129.306493 im/s
Epoch 17, Train Loss: 10972.2167969, Time: 38.6061s, Throughput: 129.305940 im/s
Epoch 18, Iter 39, Loss: 11094.8369141, Throughput: 124.375655 im/s
Epoch 18, Train Loss: 10879.8906250, Time: 40.1367s, Throughput: 124.375097 im/s
Epoch 19, Iter 39, Loss: 11134.1347656, Throughput: 121.893037 im/s
Epoch 19, Train Loss: 10764.9257812, Time: 40.9541s, Throughput: 121.892608 im/s
Epoch 20, Iter 39, Loss: 10436.4902344, Throughput: 125.133741 im/s
Epoch 20, Train Loss: 10654.5996094, Time: 39.8935s, Throughput: 125.133252 im/s
Epoch 21, Iter 39, Loss: 10872.2626953, Throughput: 127.736764 im/s
Epoch 21, Train Loss: 10657.1845703, Time: 39.0805s, Throughput: 127.736251 im/s
Epoch 22, Iter 39, Loss: 10291.5937500, Throughput: 128.074155 im/s
Epoch 22, Train Loss: 10563.5937500, Time: 38.9776s, Throughput: 128.073672 im/s
Epoch 23, Iter 39, Loss: 11123.0976562, Throughput: 124.675748 im/s
Epoch 23, Train Loss: 10388.7988281, Time: 40.0400s, Throughput: 124.675268 im/s
Epoch 24, Iter 39, Loss: 10918.2558594, Throughput: 122.188962 im/s
Epoch 24, Train Loss: 10373.1914062, Time: 40.8549s, Throughput: 122.188460 im/s
Epoch 25, Iter 39, Loss: 10210.6406250, Throughput: 126.503397 im/s
Epoch 25, Train Loss: 10301.2177734, Time: 39.4615s, Throughput: 126.502897 im/s
Epoch 26, Iter 39, Loss: 10582.3828125, Throughput: 129.126446 im/s
Epoch 26, Train Loss: 10233.7646484, Time: 38.6599s, Throughput: 129.125945 im/s
Epoch 27, Iter 39, Loss: 9977.1621094, Throughput: 128.744666 im/s
Epoch 27, Train Loss: 10165.3349609, Time: 38.7746s, Throughput: 128.744151 im/s
Epoch 28, Iter 39, Loss: 10012.5839844, Throughput: 128.844331 im/s
Epoch 28, Train Loss: 10045.2666016, Time: 38.7446s, Throughput: 128.843752 im/s
Epoch 29, Iter 39, Loss: 10165.3896484, Throughput: 127.607254 im/s
Epoch 29, Train Loss: 10084.3564453, Time: 39.1202s, Throughput: 127.606768 im/s
Epoch 30, Iter 39, Loss: 9940.0644531, Throughput: 121.269046 im/s
Epoch 30, Train Loss: 10039.9433594, Time: 41.1648s, Throughput: 121.268566 im/s
Epoch 31, Iter 39, Loss: 10282.6210938, Throughput: 121.944480 im/s
Epoch 31, Train Loss: 9928.8789062, Time: 40.9368s, Throughput: 121.944036 im/s
Epoch 32, Iter 39, Loss: 10192.1152344, Throughput: 128.332942 im/s
Epoch 32, Train Loss: 9905.8339844, Time: 38.8990s, Throughput: 128.332434 im/s
Epoch 33, Iter 39, Loss: 10541.2626953, Throughput: 124.905636 im/s
Epoch 33, Train Loss: 9808.0878906, Time: 39.9663s, Throughput: 124.905095 im/s
Epoch 34, Iter 39, Loss: 9633.1806641, Throughput: 124.242917 im/s
Epoch 34, Train Loss: 9818.3457031, Time: 40.1795s, Throughput: 124.242403 im/s
Epoch 35, Iter 39, Loss: 9775.9843750, Throughput: 123.402870 im/s
Epoch 35, Train Loss: 9802.6093750, Time: 40.4530s, Throughput: 123.402363 im/s
Epoch 36, Iter 39, Loss: 9575.0751953, Throughput: 123.391959 im/s
Epoch 36, Train Loss: 9749.3574219, Time: 40.4566s, Throughput: 123.391468 im/s
Epoch 37, Iter 39, Loss: 9225.7412109, Throughput: 124.445092 im/s
Epoch 37, Train Loss: 9672.7275391, Time: 40.1142s, Throughput: 124.444599 im/s
Epoch 38, Iter 39, Loss: 9516.1113281, Throughput: 127.211209 im/s
Epoch 38, Train Loss: 9671.8271484, Time: 39.2420s, Throughput: 127.210681 im/s
Epoch 39, Iter 39, Loss: 9724.6406250, Throughput: 128.300450 im/s
Epoch 39, Train Loss: 9566.8955078, Time: 38.9088s, Throughput: 128.299939 im/s
Epoch 40, Iter 39, Loss: 9919.4970703, Throughput: 127.849954 im/s
Epoch 40, Train Loss: 9601.9873047, Time: 39.0460s, Throughput: 127.849364 im/s
Epoch 41, Iter 39, Loss: 10051.4375000, Throughput: 129.047815 im/s
Epoch 41, Train Loss: 9691.2607422, Time: 38.6835s, Throughput: 129.047245 im/s
Epoch 42, Iter 39, Loss: 9272.8867188, Throughput: 128.174953 im/s
Epoch 42, Train Loss: 9501.7851562, Time: 38.9469s, Throughput: 128.174452 im/s
Epoch 43, Iter 39, Loss: 9460.9716797, Throughput: 124.368900 im/s
Epoch 43, Train Loss: 9444.0019531, Time: 40.1388s, Throughput: 124.368442 im/s
Epoch 44, Iter 39, Loss: 9162.3291016, Throughput: 128.020782 im/s
Epoch 44, Train Loss: 9430.1718750, Time: 38.9938s, Throughput: 128.020276 im/s
Epoch 45, Iter 39, Loss: 9401.9560547, Throughput: 128.881267 im/s
Epoch 45, Train Loss: 9441.3828125, Time: 38.7335s, Throughput: 128.880717 im/s
Epoch 46, Iter 39, Loss: 9186.8027344, Throughput: 128.589249 im/s
Epoch 46, Train Loss: 9407.2734375, Time: 38.8214s, Throughput: 128.588748 im/s
Epoch 47, Iter 39, Loss: 9415.8417969, Throughput: 126.201556 im/s
Epoch 47, Train Loss: 9402.4208984, Time: 39.5559s, Throughput: 126.200989 im/s
Epoch 48, Iter 39, Loss: 9218.3457031, Throughput: 124.955970 im/s
Epoch 48, Train Loss: 9294.5742188, Time: 39.9503s, Throughput: 124.955335 im/s
Epoch 49, Iter 39, Loss: 9466.5244141, Throughput: 127.328398 im/s
Epoch 49, Train Loss: 9351.0224609, Time: 39.2059s, Throughput: 127.327914 im/s
Epoch 50, Iter 39, Loss: 9475.5917969, Throughput: 128.658225 im/s
Epoch 50, Train Loss: 9302.8398438, Time: 38.8006s, Throughput: 128.657722 im/s

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
  CPU: 4 × AMD EPYC 9V45 96-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, znver5)
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
