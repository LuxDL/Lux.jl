using Comonicon, ConcreteStructs, DataAugmentation, ImageShow, Interpolations, Lux, LuxCUDA,
      MLDatasets, MLUtils, OneHotArrays, Optimisers, Printf, ProgressBars, Random,
      StableRNGs, Statistics, Zygote

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

function get_dataloaders(batchsize)
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    train_transform = RandomResizeCrop((32, 32)) |>
                      Maybe(FlipX{2}()) |>
                      ImageToTensor() |>
                      Normalize(cifar10_mean, cifar10_std)

    test_transform = ImageToTensor() |> Normalize(cifar10_mean, cifar10_std)

    trainset = TensorDataset(CIFAR10(:train), train_transform)
    trainloader = DataLoader(trainset; batchsize, shuffle=true, parallel=true)

    testset = TensorDataset(CIFAR10(:test), test_transform)
    testloader = DataLoader(testset; batchsize, shuffle=false, parallel=true)

    return trainloader, testloader
end

function ConvMixer(; dim, depth, kernel_size=5, patch_size=2)
    #! format: off
    return Chain(
        Conv((patch_size, patch_size), 3 => dim, gelu; stride=patch_size),
        BatchNorm(dim),
        [Chain(
             SkipConnection(
                 Chain(Conv((kernel_size, kernel_size), dim => dim, gelu; groups=dim,
                    pad=SamePad()), BatchNorm(dim)), +),
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
    st = Lux.testmode(st)
    for (x, y) in dataloader
        target_class = onecold(y)
        predicted_class = onecold(first(model(x, ps, st)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

Comonicon.@main function main(; batchsize::Int=512, hidden_dim::Int=256, depth::Int=8,
        patch_size::Int=2, kernel_size::Int=5, weight_decay::Float64=1e-5,
        clip_norm::Bool=false, seed::Int=42, epochs::Int=25, lr_max::Float64=0.01)
    rng = StableRNG(seed)

    gdev = gpu_device()
    trainloader, testloader = get_dataloaders(batchsize) |> gdev

    model = ConvMixer(; dim=hidden_dim, depth, kernel_size, patch_size)
    ps, st = Lux.setup(rng, model) |> gdev

    opt = AdamW(; eta=lr_max, lambda=weight_decay)
    clip_norm && (opt = OptimiserChain(ClipNorm(), opt))

    train_state = Training.TrainState(model, ps, st, opt)

    lr_schedule = linear_interpolation(
        [0, epochs * 2 ÷ 5, epochs * 4 ÷ 5, epochs + 1], [0, lr_max, lr_max / 20, 0])

    loss = CrossEntropyLoss(; logits=Val(true))

    for epoch in 1:epochs
        stime = time()
        lr = 0
        for (i, (x, y)) in enumerate(trainloader)
            lr = lr_schedule((epoch - 1) + (i + 1) / length(trainloader))
            train_state = Optimisers.adjust!(train_state, lr)
            (_, _, _, train_state) = Training.single_train_step!(
                AutoZygote(), loss, (x, y), train_state)
        end
        ttime = time() - stime

        train_acc = accuracy(
            model, train_state.parameters, train_state.states, trainloader) * 100
        test_acc = accuracy(model, train_state.parameters, train_state.states, testloader) *
                   100

        @printf "Epoch %2d: Learning Rate %.2e, Train Acc: %.2f%%, Test Acc: %.2f%%, \
                 Time: %.2f\n" epoch lr train_acc test_acc ttime
    end
end
