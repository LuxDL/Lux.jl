using ConcreteStructs, DataAugmentation, ImageShow, Lux, MLDatasets, MLUtils, OneHotArrays,
      Printf, ProgressTables, Random
using LuxCUDA, Reactant

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

function get_cifar10_dataloaders(batchsize; kwargs...)
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    train_transform = RandomResizeCrop((32, 32)) |>
                      Maybe(FlipX{2}()) |>
                      ImageToTensor() |>
                      Normalize(cifar10_mean, cifar10_std)

    test_transform = ImageToTensor() |> Normalize(cifar10_mean, cifar10_std)

    trainset = TensorDataset(CIFAR10(:train), train_transform)
    trainloader = DataLoader(trainset; batchsize, shuffle=true, kwargs...)

    testset = TensorDataset(CIFAR10(:test), test_transform)
    testloader = DataLoader(testset; batchsize, shuffle=false, kwargs...)

    return trainloader, testloader
end

function accuracy(model, ps, st, dataloader)
    total_correct, total = 0, 0
    cdev = cpu_device()
    for (x, y) in dataloader
        target_class = onecold(cdev(y))
        predicted_class = onecold(cdev(first(model(x, ps, st))))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

function get_accelerator_device(backend::String)
    if backend == "gpu_if_available"
        return gpu_device()
    elseif backend == "gpu"
        return gpu_device(; force=true)
    elseif backend == "reactant"
        return reactant_device(; force=true)
    elseif backend == "cpu"
        return cpu_device()
    else
        error("Invalid backend: $(backend). Valid Options are: `gpu_if_available`, `gpu`, \
               `reactant`, and `cpu`.")
    end
end

function train_model(
        model, opt, scheduler=nothing;
        backend::String, batchsize::Int=512, seed::Int=1234, epochs::Int=25
)
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    accelerator_device = get_accelerator_device(backend)
    kwargs = accelerator_device isa ReactantDevice ? (; partial=false) : ()
    trainloader, testloader = get_cifar10_dataloaders(batchsize; kwargs...) |>
                              accelerator_device

    ps, st = Lux.setup(rng, model) |> accelerator_device

    train_state = Training.TrainState(model, ps, st, opt)

    adtype = backend == "reactant" ? AutoEnzyme() : AutoZygote()

    if backend == "reactant"
        x_ra = rand(rng, Float32, size(first(trainloader)[1])) |> accelerator_device
        @printf "[Info] Compiling model with Reactant.jl\n"
        st_test = Lux.testmode(st)
        model_compiled = @compile model(x_ra, ps, st_test)
        @printf "[Info] Model compiled!\n"
    else
        model_compiled = model
    end

    loss_fn = CrossEntropyLoss(; logits=Val(true))

    pt = ProgressTable(;
        header=[
            "Epoch", "Learning Rate", "Train Accuracy (%)", "Test Accuracy (%)", "Time (s)"
        ],
        widths=[24, 24, 24, 24, 24],
        format=["%3d", "%.6f", "%.6f", "%.6f", "%.6f"],
        color=[:normal, :normal, :blue, :blue, :normal],
        border=true,
        alignment=[:center, :center, :center, :center, :center]
    )

    @printf "[Info] Training model\n"
    initialize(pt)

    for epoch in 1:epochs
        stime = time()
        lr = 0
        for (i, (x, y)) in enumerate(trainloader)
            if scheduler !== nothing
                lr = scheduler((epoch - 1) + (i + 1) / length(trainloader))
                train_state = Optimisers.adjust!(train_state, lr)
            end
            (_, loss, _, train_state) = Training.single_train_step!(
                adtype, loss_fn, (x, y), train_state
            )
            isnan(loss) && error("NaN loss encountered!")
        end
        ttime = time() - stime

        train_acc = accuracy(
            model_compiled, train_state.parameters,
            Lux.testmode(train_state.states), trainloader
        ) * 100
        test_acc = accuracy(
            model_compiled, train_state.parameters,
            Lux.testmode(train_state.states), testloader
        ) * 100

        scheduler === nothing && (lr = NaN32)
        next(pt, [epoch, lr, train_acc, test_acc, ttime])
    end

    finalize(pt)
    @printf "[Info] Finished training\n"
end
