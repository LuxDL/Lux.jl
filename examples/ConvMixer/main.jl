using Comonicon, ConcreteStructs, DataAugmentation, ImageShow, Interpolations, Lux, LuxCUDA,
      MLDatasets, MLUtils, OneHotArrays, Optimisers, Printf, ProgressBars, Random,
      StableRNGs, Statistics, Zygote
using Reactant, Enzyme

CUDA.allowscalar(false)

@concrete struct TensorDataset
    dataset
    transform
end

Base.length(ds::TensorDataset) = length(ds.dataset)

function Base.getindex(ds::TensorDataset, idxs::Union{Vector{<:Integer}, AbstractRange})
    img = Image.(eachslice(convert2image(ds.dataset, idxs); dims=3))
    y = onehotbatch(ds.dataset.targets[idxs], 0:9)
    return stack(parent ∘ itemdata ∘ Base.Fix1(apply, ds.transform), img), y
end

function get_dataloaders(batchsize; kwargs...)
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    train_transform = RandomResizeCrop((32, 32)) |>
                      Maybe(FlipX{2}()) |>
                      ImageToTensor() |>
                      Normalize(cifar10_mean, cifar10_std)

    test_transform = ImageToTensor() |> Normalize(cifar10_mean, cifar10_std)

    trainset = TensorDataset(CIFAR10(:train), train_transform)
    trainloader = DataLoader(trainset; batchsize, shuffle=true, parallel=true, kwargs...)

    testset = TensorDataset(CIFAR10(:test), test_transform)
    testloader = DataLoader(testset; batchsize, shuffle=false, parallel=true, kwargs...)

    return trainloader, testloader
end

function ConvMixer(; dim, depth, kernel_size=5, patch_size=2)
    #! format: off
    return Chain(
        Conv((patch_size, patch_size), 3 => dim, gelu; stride=patch_size),
        BatchNorm(dim),
        [Chain(
            SkipConnection(
                Chain(
                    Conv((kernel_size, kernel_size), dim => dim, gelu; groups=dim, pad=SamePad()),
                    BatchNorm(dim)
                ),
                +
            ),
            Conv((1, 1), dim => dim, gelu), BatchNorm(dim))
        for _ in 1:depth]...,
        GlobalMeanPool(),
        FlattenLayer(),
        Dense(dim => 10)
    )
    #! format: on
end

function accuracy(model, ps, st, dataloader)
    total_correct, total = 0, 0
    cdev = cpu_device()
    st = Lux.testmode(st)
    for (x, y) in dataloader
        target_class = onecold(cdev(y))
        predicted_class = onecold(cdev(first(model(x, ps, st))))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

Comonicon.@main function main(; batchsize::Int=512, hidden_dim::Int=256, depth::Int=8,
        patch_size::Int=2, kernel_size::Int=5, weight_decay::Float64=1e-5,
        clip_norm::Bool=false, seed::Int=42, epochs::Int=25, lr_max::Float64=0.01,
        backend::String="gpu_if_available")
    rng = StableRNG(seed)

    if backend == "gpu_if_available"
        accelerator_device = gpu_device()
    elseif backend == "gpu"
        accelerator_device = gpu_device(; force=true)
    elseif backend == "reactant"
        accelerator_device = reactant_device(; force=true)
    elseif backend == "cpu"
        accelerator_device = cpu_device()
    else
        error("Invalid backend: $(backend). Valid Options are: `gpu_if_available`, `gpu`, \
               `reactant`, and `cpu`.")
    end

    kwargs = accelerator_device isa ReactantDevice ? (; partial=false) : ()
    trainloader, testloader = get_dataloaders(batchsize; kwargs...) |> accelerator_device

    model = ConvMixer(; dim=hidden_dim, depth, kernel_size, patch_size)
    ps, st = Lux.setup(rng, model) |> accelerator_device

    opt = AdamW(; eta=lr_max, lambda=weight_decay)
    clip_norm && (opt = OptimiserChain(ClipNorm(), opt))

    train_state = Training.TrainState(model, ps, st, opt)

    lr_schedule = linear_interpolation(
        [0, epochs * 2 ÷ 5, epochs * 4 ÷ 5, epochs + 1], [0, lr_max, lr_max / 20, 0]
    )

    adtype = backend == "reactant" ? AutoEnzyme() : AutoZygote()

    if backend == "reactant"
        x_ra = rand(rng, Float32, size(first(trainloader)[1])) |> accelerator_device
        @printf "[Info] Compiling model with Reactant.jl\n"
        model_compiled = @compile model(x_ra, ps, Lux.testmode(st))
        @printf "[Info] Model compiled!\n"
    else
        model_compiled = model
    end

    loss = CrossEntropyLoss(; logits=Val(true))

    @printf "[Info] Training model\n"
    for epoch in 1:epochs
        stime = time()
        lr = 0
        for (i, (x, y)) in enumerate(trainloader)
            lr = lr_schedule((epoch - 1) + (i + 1) / length(trainloader))
            train_state = Optimisers.adjust!(train_state, lr)
            (_, _, _, train_state) = Training.single_train_step!(
                adtype, loss, (x, y), train_state
            )
            @show i, time() - stime
        end
        ttime = time() - stime

        train_acc = accuracy(
            model_compiled, train_state.parameters, train_state.states, trainloader
        ) * 100
        test_acc = accuracy(
            model_compiled, train_state.parameters, train_state.states, testloader
        ) * 100

        @printf "[Train] Epoch %2d: Learning Rate %.2e, Train Acc: %.2f%%, Test Acc: \
                 %.2f%%, Time: %.2f\n" epoch lr train_acc test_acc ttime
    end
    @printf "[Info] Finished training\n"
end
