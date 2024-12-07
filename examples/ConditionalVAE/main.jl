# # [Conditional VAE for MNIST using Reactant](@id Conditional-VAE-Tutorial)

# Convolutional variational autoencoder (CVAE) implementation in MLX using MNIST. This is
# based on the [CVAE implementation in MLX](https://github.com/ml-explore/mlx-examples/blob/main/cvae/).

using Lux, Reactant, MLDatasets, Random, Statistics, Enzyme, MLUtils, DataAugmentation,
      ConcreteStructs, OneHotArrays, ImageShow, Images, Printf, Optimisers

const xdev = reactant_device()
const cdev = cpu_device()

# ## Model Definition

# First we will define the encoder.It maps the input to a normal distribution in latent
# space and sample a latent vector from that distribution.

function cvae_encoder(
        rng=Random.default_rng(); num_latent_dims::Int,
        image_shape::Dims{3}, max_num_filters::Int
)
    flattened_dim = prod(image_shape[1:2] .÷ 8) * max_num_filters
    return @compact(;
        embed=Chain(
            Chain(
                Conv((3, 3), image_shape[3] => max_num_filters ÷ 4; stride=2, pad=1),
                BatchNorm(max_num_filters ÷ 4, leakyrelu)
            ),
            Chain(
                Conv((3, 3), max_num_filters ÷ 4 => max_num_filters ÷ 2; stride=2, pad=1),
                BatchNorm(max_num_filters ÷ 2, leakyrelu)
            ),
            Chain(
                Conv((3, 3), max_num_filters ÷ 2 => max_num_filters; stride=2, pad=1),
                BatchNorm(max_num_filters, leakyrelu)
            ),
            FlattenLayer()
        ),
        proj_mu=Dense(flattened_dim, num_latent_dims),
        proj_log_var=Dense(flattened_dim, num_latent_dims),
        rng) do x
        y = embed(x)

        μ = proj_mu(y)
        logσ² = proj_log_var(y)
        σ² = exp.(logσ² .* eltype(logσ²)(0.5))

        ## Generate a tensor of random values from a normal distribution
        rng = Lux.replicate(rng)
        ϵ = randn_like(rng, σ²)

        ## Reparametrization trick to brackpropagate through sampling
        z = ϵ .* σ² .+ μ

        @return z, μ, logσ²
    end
end

# Similarly we define the decoder.

function cvae_decoder(; num_latent_dims::Int, image_shape::Dims{3}, max_num_filters::Int)
    flattened_dim = prod(image_shape[1:2] .÷ 8) * max_num_filters
    return @compact(;
        linear=Dense(num_latent_dims, flattened_dim),
        upchain=Chain(
            Chain(
                Upsample(2),
                Conv((3, 3), max_num_filters => max_num_filters ÷ 2; stride=1, pad=1),
                BatchNorm(max_num_filters ÷ 2, leakyrelu)
            ),
            Chain(
                Upsample(2),
                Conv((3, 3), max_num_filters ÷ 2 => max_num_filters ÷ 4; stride=1, pad=1),
                BatchNorm(max_num_filters ÷ 4, leakyrelu)
            ),
            Chain(
                Upsample(2),
                Conv((3, 3), max_num_filters ÷ 4 => image_shape[3]; stride=1, pad=1)
            )
        ),
        max_num_filters) do x
        y = linear(x)
        img = reshape(y, image_shape[1] ÷ 8, image_shape[2] ÷ 8, max_num_filters, :)
        @return upchain(img)
    end
end

@concrete struct CVAE <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder <: Lux.AbstractLuxLayer
    decoder <: Lux.AbstractLuxLayer
end

function CVAE(; num_latent_dims::Int, image_shape::Dims{3}, max_num_filters::Int)
    decoder = cvae_decoder(; num_latent_dims, image_shape, max_num_filters)
    encoder = cvae_encoder(; num_latent_dims, image_shape, max_num_filters)
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

# ## Loading MNIST

@concrete struct TensorDataset
    dataset
    transform
end

Base.length(ds::TensorDataset) = length(ds.dataset)

function Base.getindex(ds::TensorDataset, idxs::Union{Vector{<:Integer}, AbstractRange})
    img = Image.(eachslice(convert2image(ds.dataset, idxs); dims=3))
    return stack(parent ∘ itemdata ∘ Base.Fix1(apply, ds.transform), img)
end

function loadmnist(batchsize, image_size::Dims{2})
    ## Load MNIST: Only 1500 for demonstration purposes
    N = parse(Bool, get(ENV, "CI", "false")) ? 1500 : nothing
    train_dataset = MNIST(; split=:train)
    test_dataset = MNIST(; split=:test)
    if N !== nothing
        train_dataset = train_dataset[1:N]
        test_dataset = test_dataset[1:N]
    end

    train_transform = ScaleKeepAspect(image_size) |> Maybe(FlipX{2}()) |> ImageToTensor()
    test_transform = ScaleKeepAspect(image_size) |> ImageToTensor()

    trainset = TensorDataset(train_dataset, train_transform)
    trainloader = DataLoader(
        trainset; batchsize, shuffle=true, parallel=true, partial=false)

    testset = TensorDataset(test_dataset, test_transform)
    testloader = DataLoader(testset; batchsize, shuffle=false, parallel=true, partial=false)

    return trainloader, testloader
end

# ## Helper Functions

# Generate an Image Grid from a list of images

function create_image_grid(imgs::AbstractArray, grid_rows::Int, grid_cols::Int)
    total_images = grid_rows * grid_cols
    imgs = map(eachslice(imgs[:, :, :, 1:total_images]; dims=4)) do img
        cimg = size(img, 3) == 1 ? colorview(Gray, view(img, :, :, 1)) : colorview(RGB, img)
        return cimg'
    end
    return create_image_grid(imgs, grid_rows, grid_cols)
end

function create_image_grid(images::Vector, grid_rows::Int, grid_cols::Int)
    ## Check if the number of images matches the grid
    total_images = grid_rows * grid_cols
    @assert length(images) == total_images

    ## Get the size of a single image (assuming all images are the same size)
    img_height, img_width = size(images[1])

    ## Create a blank grid canvas
    grid_height = img_height * grid_rows
    grid_width = img_width * grid_cols
    grid_canvas = similar(images[1], grid_height, grid_width)

    ## Place each image in the correct position on the canvas
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
    kldiv_loss = -0.5f0 * sum(1 .+ logσ² .- μ .^ 2 .- exp.(logσ²))
    loss = reconstruction_loss + kldiv_loss
    return loss, st, (; y, μ, logσ², reconstruction_loss, kldiv_loss)
end

function generate_images(model, ps, st; num_samples::Int=128, num_latent_dims::Int)
    z = randn(Float32, num_latent_dims, num_samples) |> get_device((ps, st))
    images, _ = decode(model, z, ps, Lux.testmode(st))
    return create_image_grid(images, 8, num_samples ÷ 8)
end

# ## Training the Model

Comonicon.@main function main(; batchsize=128, image_size=(64, 64), num_latent_dims=32,
        max_num_filters=64, seed=0, epochs=100, weight_decay=1e-3, learning_rate=1e-3)
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    cvae = CVAE(; num_latent_dims, image_shape=(image_size..., 1), max_num_filters)
    ps, st = Lux.setup(rng, cvae) |> xdev

    train_dataloader, test_dataloader = loadmnist(batchsize, image_size) |> xdev

    opt = AdamW(; eta=learning_rate, lambda=weight_decay)

    train_state = Training.TrainState(cvae, ps, st, opt)

    for epoch in 1:epochs
        loss_total = 0.0f0
        total_samples = 0

        stime = time()
        for (i, X) in enumerate(train_dataloader)
            throughput_tic = time()
            (_, loss, _, train_state) = Training.single_train_step!(
                AutoEnzyme(), loss_function, X, train_state)
            throughput_toc = time()

            loss_total += loss
            total_samples += size(X, ndims(X))

            if i % 10 == 0 || i == length(train_dataloader)
                @printf "Epoch %d, Iter %d, Loss: %.4f, Throughput: %.6f it/s\n" epoch i loss ((throughput_toc -
                                                                                                throughput_tic)/size(
                    X, ndims(X)))
            end
        end
        ttime = time() - stime

        train_loss = loss_total / total_samples
        @printf "Epoch %d, Train Loss: %.4f, Time: %.4fs\n" epoch train_loss ttime
    end
end

# XXX: Move into a proper function

rng = Random.default_rng()
Random.seed!(rng, 0)

cvae = CVAE(; num_latent_dims=32, image_shape=(64, 64, 1), max_num_filters=64)
ps, st = Lux.setup(rng, cvae) |> xdev;

train_dataloader, test_dataloader = loadmnist(128, (64, 64)) |> xdev

opt = AdamW(; eta=1e-3, lambda=1e-3)

epochs = 100

train_state = Training.TrainState(cvae, ps, st, opt)

for epoch in 1:epochs
    loss_total = 0.0f0
    total_samples = 0

    stime = time()
    for (i, X) in enumerate(train_dataloader)
        throughput_tic = time()
        (_, loss, _, train_state) = Training.single_train_step!(
            AutoEnzyme(), loss_function, X, train_state)
        throughput_toc = time()

        loss_total += loss
        total_samples += size(X, ndims(X))

        if i % 10 == 0 || i == length(train_dataloader)
            @printf "Epoch %d, Iter %d, Loss: %.4f, Throughput: %.6f it/s\n" epoch i loss ((throughput_toc -
                                                                                            throughput_tic)/size(
                X, ndims(X)))
        end
    end
    ttime = time() - stime

    train_loss = loss_total / total_samples
    @printf "Epoch %d, Train Loss: %.4f, Time: %.4fs\n" epoch train_loss ttime

    # XXX: Generate images conditionally
    display(generate_images(cvae, ps, st; num_samples=128, num_latent_dims=32))
end
