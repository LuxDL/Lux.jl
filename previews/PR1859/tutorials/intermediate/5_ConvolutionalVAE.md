---
url: /previews/PR1859/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23614.8437500, Throughput: 16.475138 im/s
Epoch 1, Train Loss: 39607.2773438, Time: 303.2778s, Throughput: 16.460156 im/s
Epoch 2, Iter 39, Loss: 17494.0683594, Throughput: 172.854785 im/s
Epoch 2, Train Loss: 20320.8164062, Time: 28.8799s, Throughput: 172.853760 im/s
Epoch 3, Iter 39, Loss: 15854.3193359, Throughput: 173.282166 im/s
Epoch 3, Train Loss: 16851.1152344, Time: 28.8087s, Throughput: 173.281204 im/s
Epoch 4, Iter 39, Loss: 15792.3115234, Throughput: 173.876994 im/s
Epoch 4, Train Loss: 15176.4179688, Time: 28.7101s, Throughput: 173.875898 im/s
Epoch 5, Iter 39, Loss: 13943.5361328, Throughput: 172.817324 im/s
Epoch 5, Train Loss: 14201.7822266, Time: 28.8862s, Throughput: 172.816217 im/s
Epoch 6, Iter 39, Loss: 13135.4140625, Throughput: 173.707123 im/s
Epoch 6, Train Loss: 13458.0253906, Time: 28.7382s, Throughput: 173.706271 im/s
Epoch 7, Iter 39, Loss: 13387.1865234, Throughput: 173.067920 im/s
Epoch 7, Train Loss: 13003.0244141, Time: 28.8443s, Throughput: 173.067002 im/s
Epoch 8, Iter 39, Loss: 12558.6591797, Throughput: 173.305238 im/s
Epoch 8, Train Loss: 12615.1582031, Time: 28.8048s, Throughput: 173.304395 im/s
Epoch 9, Iter 39, Loss: 11220.7363281, Throughput: 173.068533 im/s
Epoch 9, Train Loss: 12213.0351562, Time: 28.8442s, Throughput: 173.067609 im/s
Epoch 10, Iter 39, Loss: 12479.0146484, Throughput: 173.812598 im/s
Epoch 10, Train Loss: 12069.9384766, Time: 28.7208s, Throughput: 173.811593 im/s
Epoch 11, Iter 39, Loss: 12456.1337891, Throughput: 173.147579 im/s
Epoch 11, Train Loss: 11809.3906250, Time: 28.8310s, Throughput: 173.146679 im/s
Epoch 12, Iter 39, Loss: 11190.9814453, Throughput: 173.493345 im/s
Epoch 12, Train Loss: 11582.7119141, Time: 28.7736s, Throughput: 173.492356 im/s
Epoch 13, Iter 39, Loss: 11660.8017578, Throughput: 172.694149 im/s
Epoch 13, Train Loss: 11393.9775391, Time: 28.9067s, Throughput: 172.693265 im/s
Epoch 14, Iter 39, Loss: 11246.4433594, Throughput: 172.707551 im/s
Epoch 14, Train Loss: 11241.1357422, Time: 28.9045s, Throughput: 172.706696 im/s
Epoch 15, Iter 39, Loss: 10936.9687500, Throughput: 173.205943 im/s
Epoch 15, Train Loss: 11109.4179688, Time: 28.8213s, Throughput: 173.205047 im/s
Epoch 16, Iter 39, Loss: 11459.5615234, Throughput: 173.151729 im/s
Epoch 16, Train Loss: 10914.7187500, Time: 28.8304s, Throughput: 173.150744 im/s
Epoch 17, Iter 39, Loss: 10586.6318359, Throughput: 173.027849 im/s
Epoch 17, Train Loss: 10927.4394531, Time: 28.8510s, Throughput: 173.026865 im/s
Epoch 18, Iter 39, Loss: 11532.0273438, Throughput: 173.582793 im/s
Epoch 18, Train Loss: 10807.3271484, Time: 28.7588s, Throughput: 173.581827 im/s
Epoch 19, Iter 39, Loss: 10212.0234375, Throughput: 171.328434 im/s
Epoch 19, Train Loss: 10750.0781250, Time: 29.1372s, Throughput: 171.327599 im/s
Epoch 20, Iter 39, Loss: 10350.8183594, Throughput: 173.066042 im/s
Epoch 20, Train Loss: 10595.3535156, Time: 28.8446s, Throughput: 173.065028 im/s
Epoch 21, Iter 39, Loss: 10387.4814453, Throughput: 173.073915 im/s
Epoch 21, Train Loss: 10517.2226562, Time: 28.8433s, Throughput: 173.072996 im/s
Epoch 22, Iter 39, Loss: 10401.2646484, Throughput: 173.387759 im/s
Epoch 22, Train Loss: 10485.7841797, Time: 28.7911s, Throughput: 173.386752 im/s
Epoch 23, Iter 39, Loss: 10660.8339844, Throughput: 173.728868 im/s
Epoch 23, Train Loss: 10384.8408203, Time: 28.7346s, Throughput: 173.727798 im/s
Epoch 24, Iter 39, Loss: 10654.8222656, Throughput: 172.916808 im/s
Epoch 24, Train Loss: 10350.9462891, Time: 28.8696s, Throughput: 172.915645 im/s
Epoch 25, Iter 39, Loss: 10348.8496094, Throughput: 173.471859 im/s
Epoch 25, Train Loss: 10321.1875000, Time: 28.7772s, Throughput: 173.470810 im/s
Epoch 26, Iter 39, Loss: 11095.6992188, Throughput: 173.193841 im/s
Epoch 26, Train Loss: 10188.3457031, Time: 28.8233s, Throughput: 173.192962 im/s
Epoch 27, Iter 39, Loss: 9578.5976562, Throughput: 173.595981 im/s
Epoch 27, Train Loss: 10129.9404297, Time: 28.7566s, Throughput: 173.595027 im/s
Epoch 28, Iter 39, Loss: 10174.8828125, Throughput: 172.802673 im/s
Epoch 28, Train Loss: 10072.7363281, Time: 28.8886s, Throughput: 172.801757 im/s
Epoch 29, Iter 39, Loss: 9987.2324219, Throughput: 173.466770 im/s
Epoch 29, Train Loss: 10011.0898438, Time: 28.7780s, Throughput: 173.465630 im/s
Epoch 30, Iter 39, Loss: 10448.9589844, Throughput: 173.394281 im/s
Epoch 30, Train Loss: 10034.2441406, Time: 28.7901s, Throughput: 173.393153 im/s
Epoch 31, Iter 39, Loss: 9914.0078125, Throughput: 173.763633 im/s
Epoch 31, Train Loss: 9974.5292969, Time: 28.7288s, Throughput: 173.762629 im/s
Epoch 32, Iter 39, Loss: 10101.6562500, Throughput: 172.709332 im/s
Epoch 32, Train Loss: 9944.2451172, Time: 28.9042s, Throughput: 172.708406 im/s
Epoch 33, Iter 39, Loss: 10565.4531250, Throughput: 173.261514 im/s
Epoch 33, Train Loss: 9886.6826172, Time: 28.8121s, Throughput: 173.260595 im/s
Epoch 34, Iter 39, Loss: 9931.8613281, Throughput: 173.007970 im/s
Epoch 34, Train Loss: 9883.4072266, Time: 28.8543s, Throughput: 173.007005 im/s
Epoch 35, Iter 39, Loss: 10304.8457031, Throughput: 173.172289 im/s
Epoch 35, Train Loss: 9780.2216797, Time: 28.8270s, Throughput: 173.171232 im/s
Epoch 36, Iter 39, Loss: 9338.1562500, Throughput: 172.726261 im/s
Epoch 36, Train Loss: 9796.5468750, Time: 28.9014s, Throughput: 172.725353 im/s
Epoch 37, Iter 39, Loss: 9557.4648438, Throughput: 171.414392 im/s
Epoch 37, Train Loss: 9724.3750000, Time: 29.1226s, Throughput: 171.413491 im/s
Epoch 38, Iter 39, Loss: 9367.3359375, Throughput: 172.718115 im/s
Epoch 38, Train Loss: 9661.0908203, Time: 28.9027s, Throughput: 172.717153 im/s
Epoch 39, Iter 39, Loss: 9716.4199219, Throughput: 173.741311 im/s
Epoch 39, Train Loss: 9607.1826172, Time: 28.7325s, Throughput: 173.740374 im/s
Epoch 40, Iter 39, Loss: 9341.0332031, Throughput: 173.168301 im/s
Epoch 40, Train Loss: 9653.3457031, Time: 28.8276s, Throughput: 173.167345 im/s
Epoch 41, Iter 39, Loss: 10124.3056641, Throughput: 173.506569 im/s
Epoch 41, Train Loss: 9581.4013672, Time: 28.7714s, Throughput: 173.505508 im/s
Epoch 42, Iter 39, Loss: 10105.4941406, Throughput: 172.951296 im/s
Epoch 42, Train Loss: 9586.8876953, Time: 28.8638s, Throughput: 172.950386 im/s
Epoch 43, Iter 39, Loss: 9590.8037109, Throughput: 173.768013 im/s
Epoch 43, Train Loss: 9487.7529297, Time: 28.7281s, Throughput: 173.766954 im/s
Epoch 44, Iter 39, Loss: 9169.6416016, Throughput: 173.456922 im/s
Epoch 44, Train Loss: 9446.2441406, Time: 28.7796s, Throughput: 173.456071 im/s
Epoch 45, Iter 39, Loss: 10004.6777344, Throughput: 173.735174 im/s
Epoch 45, Train Loss: 9422.5097656, Time: 28.7335s, Throughput: 173.734273 im/s
Epoch 46, Iter 39, Loss: 9535.2851562, Throughput: 173.640249 im/s
Epoch 46, Train Loss: 9390.5253906, Time: 28.7493s, Throughput: 173.639246 im/s
Epoch 47, Iter 39, Loss: 10357.9101562, Throughput: 173.153224 im/s
Epoch 47, Train Loss: 9399.9140625, Time: 28.8301s, Throughput: 173.152348 im/s
Epoch 48, Iter 39, Loss: 9043.8046875, Throughput: 173.344794 im/s
Epoch 48, Train Loss: 9370.6416016, Time: 28.7983s, Throughput: 173.343680 im/s
Epoch 49, Iter 39, Loss: 9562.0966797, Throughput: 172.587025 im/s
Epoch 49, Train Loss: 9379.7880859, Time: 28.9247s, Throughput: 172.586161 im/s
Epoch 50, Iter 39, Loss: 9442.8476562, Throughput: 173.081919 im/s
Epoch 50, Train Loss: 9283.2939453, Time: 28.8420s, Throughput: 173.081067 im/s

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
