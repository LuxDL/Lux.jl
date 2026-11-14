---
url: /previews/PR1702/tutorials/intermediate/5_ConvolutionalVAE.md
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
Epoch 1, Iter 39, Loss: 23409.3750000, Throughput: 3.883900 im/s
Epoch 1, Train Loss: 39814.0820312, Time: 1285.6094s, Throughput: 3.882983 im/s
Epoch 2, Iter 39, Loss: 19377.0722656, Throughput: 69.737008 im/s
Epoch 2, Train Loss: 20425.9472656, Time: 71.5834s, Throughput: 69.736794 im/s
Epoch 3, Iter 39, Loss: 15271.5878906, Throughput: 70.013868 im/s
Epoch 3, Train Loss: 16778.8066406, Time: 71.3004s, Throughput: 70.013666 im/s
Epoch 4, Iter 39, Loss: 13945.8857422, Throughput: 69.853780 im/s
Epoch 4, Train Loss: 15254.3623047, Time: 71.4638s, Throughput: 69.853583 im/s
Epoch 5, Iter 39, Loss: 14110.8427734, Throughput: 70.185910 im/s
Epoch 5, Train Loss: 14293.4267578, Time: 71.1256s, Throughput: 70.185704 im/s
Epoch 6, Iter 39, Loss: 12726.2050781, Throughput: 68.747983 im/s
Epoch 6, Train Loss: 13679.2421875, Time: 72.6133s, Throughput: 68.747781 im/s
Epoch 7, Iter 39, Loss: 13140.7597656, Throughput: 69.356106 im/s
Epoch 7, Train Loss: 13145.4980469, Time: 71.9766s, Throughput: 69.355922 im/s
Epoch 8, Iter 39, Loss: 12538.6816406, Throughput: 70.137447 im/s
Epoch 8, Train Loss: 12699.7753906, Time: 71.1747s, Throughput: 70.137235 im/s
Epoch 9, Iter 39, Loss: 12697.7773438, Throughput: 70.076379 im/s
Epoch 9, Train Loss: 12495.8349609, Time: 71.2368s, Throughput: 70.076175 im/s
Epoch 10, Iter 39, Loss: 12584.1533203, Throughput: 70.090679 im/s
Epoch 10, Train Loss: 12255.9521484, Time: 71.2222s, Throughput: 70.090496 im/s
Epoch 11, Iter 39, Loss: 12190.8984375, Throughput: 69.858379 im/s
Epoch 11, Train Loss: 11967.0996094, Time: 71.4590s, Throughput: 69.858195 im/s
Epoch 12, Iter 39, Loss: 11272.5371094, Throughput: 70.005764 im/s
Epoch 12, Train Loss: 11705.6640625, Time: 71.3086s, Throughput: 70.005608 im/s
Epoch 13, Iter 39, Loss: 11749.7128906, Throughput: 70.123351 im/s
Epoch 13, Train Loss: 11502.2558594, Time: 71.1890s, Throughput: 70.123158 im/s
Epoch 14, Iter 39, Loss: 11386.3740234, Throughput: 69.911175 im/s
Epoch 14, Train Loss: 11430.8505859, Time: 71.4051s, Throughput: 69.911007 im/s
Epoch 15, Iter 39, Loss: 10843.2197266, Throughput: 69.889983 im/s
Epoch 15, Train Loss: 11259.6328125, Time: 71.4267s, Throughput: 69.889807 im/s
Epoch 16, Iter 39, Loss: 11548.4902344, Throughput: 70.006726 im/s
Epoch 16, Train Loss: 11151.2373047, Time: 71.3076s, Throughput: 70.006534 im/s
Epoch 17, Iter 39, Loss: 10763.9843750, Throughput: 70.330539 im/s
Epoch 17, Train Loss: 11047.7607422, Time: 70.9793s, Throughput: 70.330345 im/s
Epoch 18, Iter 39, Loss: 11671.4326172, Throughput: 69.976371 im/s
Epoch 18, Train Loss: 10970.4882812, Time: 71.3386s, Throughput: 69.976189 im/s
Epoch 19, Iter 39, Loss: 10838.8710938, Throughput: 70.130583 im/s
Epoch 19, Train Loss: 10840.7998047, Time: 71.1817s, Throughput: 70.130402 im/s
Epoch 20, Iter 39, Loss: 10662.0292969, Throughput: 69.726666 im/s
Epoch 20, Train Loss: 10764.9833984, Time: 71.5940s, Throughput: 69.726491 im/s
Epoch 21, Iter 39, Loss: 10097.3476562, Throughput: 69.839768 im/s
Epoch 21, Train Loss: 10649.4970703, Time: 71.4781s, Throughput: 69.839594 im/s
Epoch 22, Iter 39, Loss: 10743.1015625, Throughput: 70.099924 im/s
Epoch 22, Train Loss: 10551.9169922, Time: 71.2128s, Throughput: 70.099740 im/s
Epoch 23, Iter 39, Loss: 10145.4492188, Throughput: 69.855055 im/s
Epoch 23, Train Loss: 10449.0957031, Time: 71.4625s, Throughput: 69.854827 im/s
Epoch 24, Iter 39, Loss: 9378.3144531, Throughput: 70.080692 im/s
Epoch 24, Train Loss: 10382.7705078, Time: 71.2324s, Throughput: 70.080504 im/s
Epoch 25, Iter 39, Loss: 10719.3085938, Throughput: 69.438624 im/s
Epoch 25, Train Loss: 10250.5341797, Time: 71.8910s, Throughput: 69.438440 im/s
Epoch 26, Iter 39, Loss: 9525.2226562, Throughput: 69.772737 im/s
Epoch 26, Train Loss: 10247.2070312, Time: 71.5468s, Throughput: 69.772554 im/s
Epoch 27, Iter 39, Loss: 11098.2519531, Throughput: 69.353249 im/s
Epoch 27, Train Loss: 10220.2675781, Time: 71.9795s, Throughput: 69.353070 im/s
Epoch 28, Iter 39, Loss: 10439.1044922, Throughput: 69.818894 im/s
Epoch 28, Train Loss: 10138.2197266, Time: 71.4995s, Throughput: 69.818710 im/s
Epoch 29, Iter 39, Loss: 9931.7626953, Throughput: 69.726367 im/s
Epoch 29, Train Loss: 10065.4736328, Time: 71.5943s, Throughput: 69.726183 im/s
Epoch 30, Iter 39, Loss: 9851.3798828, Throughput: 70.102039 im/s
Epoch 30, Train Loss: 10037.2792969, Time: 71.2107s, Throughput: 70.101855 im/s
Epoch 31, Iter 39, Loss: 10147.2246094, Throughput: 69.715351 im/s
Epoch 31, Train Loss: 10035.3222656, Time: 71.6056s, Throughput: 69.715184 im/s
Epoch 32, Iter 39, Loss: 10019.9277344, Throughput: 70.017288 im/s
Epoch 32, Train Loss: 9970.3056641, Time: 71.2969s, Throughput: 70.017118 im/s
Epoch 33, Iter 39, Loss: 10533.9609375, Throughput: 69.869582 im/s
Epoch 33, Train Loss: 9855.1621094, Time: 71.4476s, Throughput: 69.869411 im/s
Epoch 34, Iter 39, Loss: 9900.8281250, Throughput: 70.041425 im/s
Epoch 34, Train Loss: 9844.6748047, Time: 71.2723s, Throughput: 70.041230 im/s
Epoch 35, Iter 39, Loss: 10324.3115234, Throughput: 69.788180 im/s
Epoch 35, Train Loss: 9865.9707031, Time: 71.5309s, Throughput: 69.788013 im/s
Epoch 36, Iter 39, Loss: 9240.6347656, Throughput: 69.655700 im/s
Epoch 36, Train Loss: 9769.2656250, Time: 71.6670s, Throughput: 69.655489 im/s
Epoch 37, Iter 39, Loss: 9735.0449219, Throughput: 69.989752 im/s
Epoch 37, Train Loss: 9732.3320312, Time: 71.3249s, Throughput: 69.989577 im/s
Epoch 38, Iter 39, Loss: 9995.8994141, Throughput: 69.719522 im/s
Epoch 38, Train Loss: 9748.3017578, Time: 71.6014s, Throughput: 69.719322 im/s
Epoch 39, Iter 39, Loss: 9506.3173828, Throughput: 69.734606 im/s
Epoch 39, Train Loss: 9571.4755859, Time: 71.5859s, Throughput: 69.734412 im/s
Epoch 40, Iter 39, Loss: 10033.8916016, Throughput: 70.419780 im/s
Epoch 40, Train Loss: 9678.9804688, Time: 70.8894s, Throughput: 70.419603 im/s
Epoch 41, Iter 39, Loss: 9494.7441406, Throughput: 69.735602 im/s
Epoch 41, Train Loss: 9584.2724609, Time: 71.5848s, Throughput: 69.735427 im/s
Epoch 42, Iter 39, Loss: 9300.0605469, Throughput: 70.231180 im/s
Epoch 42, Train Loss: 9504.1796875, Time: 71.0797s, Throughput: 70.230997 im/s
Epoch 43, Iter 39, Loss: 9646.0351562, Throughput: 70.092144 im/s
Epoch 43, Train Loss: 9493.1835938, Time: 71.2207s, Throughput: 70.091975 im/s
Epoch 44, Iter 39, Loss: 9348.9863281, Throughput: 69.818742 im/s
Epoch 44, Train Loss: 9505.4443359, Time: 71.4996s, Throughput: 69.818549 im/s
Epoch 45, Iter 39, Loss: 9402.1162109, Throughput: 70.063615 im/s
Epoch 45, Train Loss: 9443.1347656, Time: 71.2497s, Throughput: 70.063464 im/s
Epoch 46, Iter 39, Loss: 9355.3652344, Throughput: 70.085918 im/s
Epoch 46, Train Loss: 9416.6142578, Time: 71.2270s, Throughput: 70.085761 im/s
Epoch 47, Iter 39, Loss: 8925.4531250, Throughput: 69.942794 im/s
Epoch 47, Train Loss: 9419.3027344, Time: 71.3728s, Throughput: 69.942599 im/s
Epoch 48, Iter 39, Loss: 9757.4726562, Throughput: 69.827949 im/s
Epoch 48, Train Loss: 9350.2968750, Time: 71.4902s, Throughput: 69.827778 im/s
Epoch 49, Iter 39, Loss: 9465.2929688, Throughput: 69.844431 im/s
Epoch 49, Train Loss: 9335.9150391, Time: 71.4733s, Throughput: 69.844243 im/s
Epoch 50, Iter 39, Loss: 9083.6464844, Throughput: 69.956672 im/s
Epoch 50, Train Loss: 9283.0722656, Time: 71.3587s, Throughput: 69.956478 im/s

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
Julia Version 1.12.6
Commit 15346901f00 (2026-04-09 19:20 UTC)
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
