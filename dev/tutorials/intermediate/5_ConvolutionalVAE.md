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
Epoch 1, Iter 39, Loss: 23578.7539062, Throughput: 6.321473 im/s
Epoch 1, Train Loss: 39511.1289062, Time: 789.9834s, Throughput: 6.319120 im/s
Epoch 2, Iter 39, Loss: 18018.2968750, Throughput: 75.257919 im/s
Epoch 2, Train Loss: 20022.5507812, Time: 66.3321s, Throughput: 75.257732 im/s
Epoch 3, Iter 39, Loss: 14985.8837891, Throughput: 75.707483 im/s
Epoch 3, Train Loss: 16516.4257812, Time: 65.9382s, Throughput: 75.707278 im/s
Epoch 4, Iter 39, Loss: 14159.0107422, Throughput: 75.687858 im/s
Epoch 4, Train Loss: 14999.8720703, Time: 65.9553s, Throughput: 75.687663 im/s
Epoch 5, Iter 39, Loss: 12819.0039062, Throughput: 74.950907 im/s
Epoch 5, Train Loss: 13930.2031250, Time: 66.6038s, Throughput: 74.950709 im/s
Epoch 6, Iter 39, Loss: 12287.7998047, Throughput: 75.455882 im/s
Epoch 6, Train Loss: 13276.6679688, Time: 66.1580s, Throughput: 75.455687 im/s
Epoch 7, Iter 39, Loss: 13094.9785156, Throughput: 75.072256 im/s
Epoch 7, Train Loss: 12938.1953125, Time: 66.4961s, Throughput: 75.072071 im/s
Epoch 8, Iter 39, Loss: 12404.4501953, Throughput: 75.601309 im/s
Epoch 8, Train Loss: 12477.4843750, Time: 66.0308s, Throughput: 75.601115 im/s
Epoch 9, Iter 39, Loss: 12114.5800781, Throughput: 74.926232 im/s
Epoch 9, Train Loss: 12102.5019531, Time: 66.6257s, Throughput: 74.926050 im/s
Epoch 10, Iter 39, Loss: 12063.8398438, Throughput: 75.220032 im/s
Epoch 10, Train Loss: 11863.2949219, Time: 66.3655s, Throughput: 75.219855 im/s
Epoch 11, Iter 39, Loss: 10626.1279297, Throughput: 75.659462 im/s
Epoch 11, Train Loss: 11696.7783203, Time: 65.9800s, Throughput: 75.659283 im/s
Epoch 12, Iter 39, Loss: 11444.1972656, Throughput: 75.540617 im/s
Epoch 12, Train Loss: 11521.2050781, Time: 66.0838s, Throughput: 75.540444 im/s
Epoch 13, Iter 39, Loss: 10897.4140625, Throughput: 75.881911 im/s
Epoch 13, Train Loss: 11305.5205078, Time: 65.7866s, Throughput: 75.881738 im/s
Epoch 14, Iter 39, Loss: 11131.1865234, Throughput: 76.141922 im/s
Epoch 14, Train Loss: 11219.4755859, Time: 65.5619s, Throughput: 76.141745 im/s
Epoch 15, Iter 39, Loss: 10431.1386719, Throughput: 75.961578 im/s
Epoch 15, Train Loss: 11108.7539062, Time: 65.7176s, Throughput: 75.961406 im/s
Epoch 16, Iter 39, Loss: 11066.9111328, Throughput: 75.377858 im/s
Epoch 16, Train Loss: 11016.4873047, Time: 66.2265s, Throughput: 75.377690 im/s
Epoch 17, Iter 39, Loss: 11232.6523438, Throughput: 75.765574 im/s
Epoch 17, Train Loss: 10825.6669922, Time: 65.8876s, Throughput: 75.765384 im/s
Epoch 18, Iter 39, Loss: 10836.1191406, Throughput: 75.376567 im/s
Epoch 18, Train Loss: 10790.6640625, Time: 66.2276s, Throughput: 75.376389 im/s
Epoch 19, Iter 39, Loss: 10539.2460938, Throughput: 74.931638 im/s
Epoch 19, Train Loss: 10620.7539062, Time: 66.6209s, Throughput: 74.931482 im/s
Epoch 20, Iter 39, Loss: 10982.8740234, Throughput: 75.172283 im/s
Epoch 20, Train Loss: 10613.5322266, Time: 66.4076s, Throughput: 75.172119 im/s
Epoch 21, Iter 39, Loss: 11035.0195312, Throughput: 75.590497 im/s
Epoch 21, Train Loss: 10498.9150391, Time: 66.0402s, Throughput: 75.590309 im/s
Epoch 22, Iter 39, Loss: 10626.9033203, Throughput: 74.896863 im/s
Epoch 22, Train Loss: 10423.3642578, Time: 66.6518s, Throughput: 74.896701 im/s
Epoch 23, Iter 39, Loss: 10787.4716797, Throughput: 75.287611 im/s
Epoch 23, Train Loss: 10344.2421875, Time: 66.3059s, Throughput: 75.287441 im/s
Epoch 24, Iter 39, Loss: 9559.4003906, Throughput: 74.931565 im/s
Epoch 24, Train Loss: 10299.2167969, Time: 66.6209s, Throughput: 74.931394 im/s
Epoch 25, Iter 39, Loss: 9572.0654297, Throughput: 74.964187 im/s
Epoch 25, Train Loss: 10208.9472656, Time: 66.5920s, Throughput: 74.964010 im/s
Epoch 26, Iter 39, Loss: 10817.8964844, Throughput: 75.114970 im/s
Epoch 26, Train Loss: 10132.8935547, Time: 66.4583s, Throughput: 75.114794 im/s
Epoch 27, Iter 39, Loss: 10799.8164062, Throughput: 75.054162 im/s
Epoch 27, Train Loss: 10156.8505859, Time: 66.5121s, Throughput: 75.053982 im/s
Epoch 28, Iter 39, Loss: 10184.5019531, Throughput: 75.338154 im/s
Epoch 28, Train Loss: 10084.0859375, Time: 66.2614s, Throughput: 75.337976 im/s
Epoch 29, Iter 39, Loss: 10159.1230469, Throughput: 75.628003 im/s
Epoch 29, Train Loss: 9940.9492188, Time: 66.0074s, Throughput: 75.627842 im/s
Epoch 30, Iter 39, Loss: 10717.8398438, Throughput: 75.188513 im/s
Epoch 30, Train Loss: 9975.7929688, Time: 66.3933s, Throughput: 75.188341 im/s
Epoch 31, Iter 39, Loss: 10046.5898438, Throughput: 75.092292 im/s
Epoch 31, Train Loss: 9935.5888672, Time: 66.4784s, Throughput: 75.092115 im/s
Epoch 32, Iter 39, Loss: 9170.1083984, Throughput: 75.836427 im/s
Epoch 32, Train Loss: 9854.2402344, Time: 65.8260s, Throughput: 75.836269 im/s
Epoch 33, Iter 39, Loss: 9297.5927734, Throughput: 75.303911 im/s
Epoch 33, Train Loss: 9832.6406250, Time: 66.2915s, Throughput: 75.303741 im/s
Epoch 34, Iter 39, Loss: 10102.5839844, Throughput: 75.405304 im/s
Epoch 34, Train Loss: 9742.7197266, Time: 66.2024s, Throughput: 75.405131 im/s
Epoch 35, Iter 39, Loss: 9819.1679688, Throughput: 75.392417 im/s
Epoch 35, Train Loss: 9720.2099609, Time: 66.2137s, Throughput: 75.392248 im/s
Epoch 36, Iter 39, Loss: 9530.1064453, Throughput: 75.159584 im/s
Epoch 36, Train Loss: 9712.4287109, Time: 66.4188s, Throughput: 75.159411 im/s
Epoch 37, Iter 39, Loss: 10131.9785156, Throughput: 75.070271 im/s
Epoch 37, Train Loss: 9616.6894531, Time: 66.4979s, Throughput: 75.070066 im/s
Epoch 38, Iter 39, Loss: 10212.1484375, Throughput: 75.248124 im/s
Epoch 38, Train Loss: 9664.5371094, Time: 66.3407s, Throughput: 75.247952 im/s
Epoch 39, Iter 39, Loss: 10004.9765625, Throughput: 74.650833 im/s
Epoch 39, Train Loss: 9606.3789062, Time: 66.8715s, Throughput: 74.650666 im/s
Epoch 40, Iter 39, Loss: 9687.2734375, Throughput: 75.059318 im/s
Epoch 40, Train Loss: 9552.1396484, Time: 66.5075s, Throughput: 75.059151 im/s
Epoch 41, Iter 39, Loss: 10629.8476562, Throughput: 75.339682 im/s
Epoch 41, Train Loss: 9544.5419922, Time: 66.2601s, Throughput: 75.339498 im/s
Epoch 42, Iter 39, Loss: 9420.9453125, Throughput: 74.627477 im/s
Epoch 42, Train Loss: 9510.8867188, Time: 66.8924s, Throughput: 74.627321 im/s
Epoch 43, Iter 39, Loss: 9581.8613281, Throughput: 75.218630 im/s
Epoch 43, Train Loss: 9454.7109375, Time: 66.3667s, Throughput: 75.218437 im/s
Epoch 44, Iter 39, Loss: 9731.1044922, Throughput: 75.406397 im/s
Epoch 44, Train Loss: 9367.7490234, Time: 66.2014s, Throughput: 75.406230 im/s
Epoch 45, Iter 39, Loss: 9325.5136719, Throughput: 74.985971 im/s
Epoch 45, Train Loss: 9457.7451172, Time: 66.5726s, Throughput: 74.985809 im/s
Epoch 46, Iter 39, Loss: 9251.8359375, Throughput: 74.696818 im/s
Epoch 46, Train Loss: 9409.5917969, Time: 66.8303s, Throughput: 74.696649 im/s
Epoch 47, Iter 39, Loss: 9629.6191406, Throughput: 74.970446 im/s
Epoch 47, Train Loss: 9338.2070312, Time: 66.5864s, Throughput: 74.970281 im/s
Epoch 48, Iter 39, Loss: 9283.7460938, Throughput: 75.009481 im/s
Epoch 48, Train Loss: 9394.3427734, Time: 66.5517s, Throughput: 75.009324 im/s
Epoch 49, Iter 39, Loss: 9500.5996094, Throughput: 75.097187 im/s
Epoch 49, Train Loss: 9313.5947266, Time: 66.4740s, Throughput: 75.096997 im/s
Epoch 50, Iter 39, Loss: 9605.0644531, Throughput: 75.518974 im/s
Epoch 50, Train Loss: 9411.7988281, Time: 66.1027s, Throughput: 75.518816 im/s

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
Julia Version 1.12.4
Commit 01a2eadb047 (2026-01-06 16:56 UTC)
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
