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
Epoch 1, Iter 39, Loss: 23504.5468750, Throughput: 20.328268 im/s
Epoch 1, Train Loss: 39746.5429688, Time: 245.7930s, Throughput: 20.309770 im/s
Epoch 2, Iter 39, Loss: 17457.8398438, Throughput: 121.725812 im/s
Epoch 2, Train Loss: 20217.5156250, Time: 41.0104s, Throughput: 121.725173 im/s
Epoch 3, Iter 39, Loss: 16353.9941406, Throughput: 121.762406 im/s
Epoch 3, Train Loss: 16783.9511719, Time: 40.9981s, Throughput: 121.761717 im/s
Epoch 4, Iter 39, Loss: 15172.5927734, Throughput: 121.684339 im/s
Epoch 4, Train Loss: 15183.7500000, Time: 41.0244s, Throughput: 121.683805 im/s
Epoch 5, Iter 39, Loss: 14789.0527344, Throughput: 121.605565 im/s
Epoch 5, Train Loss: 14223.9580078, Time: 41.0510s, Throughput: 121.604934 im/s
Epoch 6, Iter 39, Loss: 12920.7128906, Throughput: 122.025521 im/s
Epoch 6, Train Loss: 13560.6943359, Time: 40.9097s, Throughput: 122.024948 im/s
Epoch 7, Iter 39, Loss: 12216.6464844, Throughput: 121.471551 im/s
Epoch 7, Train Loss: 13107.8349609, Time: 41.0962s, Throughput: 121.470984 im/s
Epoch 8, Iter 39, Loss: 13368.5703125, Throughput: 122.285804 im/s
Epoch 8, Train Loss: 12629.0019531, Time: 40.8226s, Throughput: 122.285303 im/s
Epoch 9, Iter 39, Loss: 11962.1591797, Throughput: 122.585875 im/s
Epoch 9, Train Loss: 12229.4980469, Time: 40.7227s, Throughput: 122.585289 im/s
Epoch 10, Iter 39, Loss: 12228.5791016, Throughput: 122.415181 im/s
Epoch 10, Train Loss: 11970.7597656, Time: 40.7794s, Throughput: 122.414659 im/s
Epoch 11, Iter 39, Loss: 12621.9013672, Throughput: 121.965572 im/s
Epoch 11, Train Loss: 11812.0576172, Time: 40.9298s, Throughput: 121.965053 im/s
Epoch 12, Iter 39, Loss: 12025.3349609, Throughput: 121.852197 im/s
Epoch 12, Train Loss: 11605.1582031, Time: 40.9679s, Throughput: 121.851606 im/s
Epoch 13, Iter 39, Loss: 11836.9306641, Throughput: 121.817068 im/s
Epoch 13, Train Loss: 11421.9755859, Time: 40.9796s, Throughput: 121.816578 im/s
Epoch 14, Iter 39, Loss: 10678.9960938, Throughput: 122.104186 im/s
Epoch 14, Train Loss: 11294.0117188, Time: 40.8833s, Throughput: 122.103706 im/s
Epoch 15, Iter 39, Loss: 10858.6679688, Throughput: 122.121933 im/s
Epoch 15, Train Loss: 11204.5781250, Time: 40.8774s, Throughput: 122.121413 im/s
Epoch 16, Iter 39, Loss: 10519.3281250, Throughput: 122.579265 im/s
Epoch 16, Train Loss: 10964.4501953, Time: 40.7248s, Throughput: 122.578790 im/s
Epoch 17, Iter 39, Loss: 10559.6914062, Throughput: 122.622187 im/s
Epoch 17, Train Loss: 10868.4882812, Time: 40.7106s, Throughput: 122.621715 im/s
Epoch 18, Iter 39, Loss: 10888.6445312, Throughput: 122.400251 im/s
Epoch 18, Train Loss: 10863.5205078, Time: 40.7844s, Throughput: 122.399780 im/s
Epoch 19, Iter 39, Loss: 10399.3164062, Throughput: 122.463573 im/s
Epoch 19, Train Loss: 10704.3603516, Time: 40.7633s, Throughput: 122.463092 im/s
Epoch 20, Iter 39, Loss: 10727.6777344, Throughput: 122.902381 im/s
Epoch 20, Train Loss: 10634.2978516, Time: 40.6178s, Throughput: 122.901884 im/s
Epoch 21, Iter 39, Loss: 10577.1777344, Throughput: 122.435979 im/s
Epoch 21, Train Loss: 10504.2343750, Time: 40.7725s, Throughput: 122.435387 im/s
Epoch 22, Iter 39, Loss: 10743.0722656, Throughput: 121.373826 im/s
Epoch 22, Train Loss: 10545.8251953, Time: 41.1293s, Throughput: 121.373312 im/s
Epoch 23, Iter 39, Loss: 10782.6533203, Throughput: 122.772891 im/s
Epoch 23, Train Loss: 10488.8544922, Time: 40.6606s, Throughput: 122.772293 im/s
Epoch 24, Iter 39, Loss: 10235.5546875, Throughput: 122.902584 im/s
Epoch 24, Train Loss: 10323.8007812, Time: 40.6178s, Throughput: 122.901809 im/s
Epoch 25, Iter 39, Loss: 10138.1679688, Throughput: 122.566544 im/s
Epoch 25, Train Loss: 10333.2197266, Time: 40.7291s, Throughput: 122.566026 im/s
Epoch 26, Iter 39, Loss: 10414.9375000, Throughput: 122.517291 im/s
Epoch 26, Train Loss: 10178.8886719, Time: 40.7456s, Throughput: 122.516311 im/s
Epoch 27, Iter 39, Loss: 10739.6455078, Throughput: 121.881877 im/s
Epoch 27, Train Loss: 10140.2470703, Time: 40.9579s, Throughput: 121.881332 im/s
Epoch 28, Iter 39, Loss: 9852.0283203, Throughput: 122.997169 im/s
Epoch 28, Train Loss: 10084.4707031, Time: 40.5865s, Throughput: 122.996588 im/s
Epoch 29, Iter 39, Loss: 9924.1445312, Throughput: 122.922328 im/s
Epoch 29, Train Loss: 10026.4765625, Time: 40.6112s, Throughput: 122.921883 im/s
Epoch 30, Iter 39, Loss: 9757.7031250, Throughput: 123.004149 im/s
Epoch 30, Train Loss: 10012.9121094, Time: 40.5842s, Throughput: 123.003594 im/s
Epoch 31, Iter 39, Loss: 10530.6181641, Throughput: 122.630410 im/s
Epoch 31, Train Loss: 10006.0791016, Time: 40.7078s, Throughput: 122.629982 im/s
Epoch 32, Iter 39, Loss: 9895.8876953, Throughput: 122.523492 im/s
Epoch 32, Train Loss: 9883.2548828, Time: 40.7434s, Throughput: 122.522896 im/s
Epoch 33, Iter 39, Loss: 10159.0000000, Throughput: 122.146394 im/s
Epoch 33, Train Loss: 9863.5351562, Time: 40.8692s, Throughput: 122.145760 im/s
Epoch 34, Iter 39, Loss: 10039.1025391, Throughput: 122.951601 im/s
Epoch 34, Train Loss: 9814.5673828, Time: 40.6015s, Throughput: 122.951150 im/s
Epoch 35, Iter 39, Loss: 10457.7285156, Throughput: 123.181901 im/s
Epoch 35, Train Loss: 9781.3173828, Time: 40.5256s, Throughput: 123.181357 im/s
Epoch 36, Iter 39, Loss: 10022.7548828, Throughput: 123.157024 im/s
Epoch 36, Train Loss: 9772.2011719, Time: 40.5573s, Throughput: 123.085123 im/s
Epoch 37, Iter 39, Loss: 9707.7519531, Throughput: 122.921690 im/s
Epoch 37, Train Loss: 9689.1318359, Time: 40.6114s, Throughput: 122.921196 im/s
Epoch 38, Iter 39, Loss: 9712.0439453, Throughput: 123.614950 im/s
Epoch 38, Train Loss: 9726.8906250, Time: 40.3836s, Throughput: 123.614491 im/s
Epoch 39, Iter 39, Loss: 10170.0429688, Throughput: 123.394393 im/s
Epoch 39, Train Loss: 9680.1044922, Time: 40.4558s, Throughput: 123.393929 im/s
Epoch 40, Iter 39, Loss: 10003.6328125, Throughput: 123.246691 im/s
Epoch 40, Train Loss: 9614.9541016, Time: 40.5043s, Throughput: 123.246131 im/s
Epoch 41, Iter 39, Loss: 9333.2324219, Throughput: 122.513009 im/s
Epoch 41, Train Loss: 9618.2001953, Time: 40.7469s, Throughput: 122.512495 im/s
Epoch 42, Iter 39, Loss: 9275.1835938, Throughput: 122.152700 im/s
Epoch 42, Train Loss: 9527.4931641, Time: 40.8671s, Throughput: 122.152112 im/s
Epoch 43, Iter 39, Loss: 9561.7109375, Throughput: 121.640397 im/s
Epoch 43, Train Loss: 9518.6171875, Time: 41.0392s, Throughput: 121.639852 im/s
Epoch 44, Iter 39, Loss: 9495.0468750, Throughput: 123.134061 im/s
Epoch 44, Train Loss: 9455.1582031, Time: 40.5413s, Throughput: 123.133542 im/s
Epoch 45, Iter 39, Loss: 9908.9208984, Throughput: 123.184849 im/s
Epoch 45, Train Loss: 9573.1064453, Time: 40.5246s, Throughput: 123.184299 im/s
Epoch 46, Iter 39, Loss: 9286.3164062, Throughput: 123.016446 im/s
Epoch 46, Train Loss: 9504.8027344, Time: 40.5801s, Throughput: 123.015861 im/s
Epoch 47, Iter 39, Loss: 9280.5517578, Throughput: 123.149528 im/s
Epoch 47, Train Loss: 9457.9394531, Time: 40.5363s, Throughput: 123.148957 im/s
Epoch 48, Iter 39, Loss: 8940.7500000, Throughput: 123.022297 im/s
Epoch 48, Train Loss: 9389.6855469, Time: 40.5783s, Throughput: 123.021545 im/s
Epoch 49, Iter 39, Loss: 9706.6533203, Throughput: 122.804286 im/s
Epoch 49, Train Loss: 9333.6933594, Time: 40.6503s, Throughput: 122.803606 im/s
Epoch 50, Iter 39, Loss: 9604.3056641, Throughput: 122.716404 im/s
Epoch 50, Train Loss: 9325.5185547, Time: 40.6793s, Throughput: 122.715882 im/s

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
