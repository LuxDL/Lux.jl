# Set some global flags that will improve performance
XLA_FLAGS = get(ENV, "XLA_FLAGS", "")
ENV["XLA_FLAGS"] = "$(XLA_FLAGS) --xla_gpu_enable_cublaslt=true"

# ## Load Common Packages

using ConcreteStructs,
    DataAugmentation,
    ImageShow,
    Lux,
    MLDatasets,
    MLUtils,
    OneHotArrays,
    Printf,
    ProgressTables,
    Random,
    BFloat16s
using Reactant

# ## Data Loading Functionality

@concrete struct TensorDataset
    dataset
    transform
end

Base.length(ds::TensorDataset) = length(ds.dataset)

function Base.getindex(ds::TensorDataset, idxs::Union{Vector{<:Integer},AbstractRange})
    img = Image.(eachslice(convert2image(ds.dataset, idxs); dims=3))
    y = onehotbatch(ds.dataset.targets[idxs], 0:9)
    return stack(parent ∘ itemdata ∘ Base.Fix1(apply, ds.transform), img), y
end

function get_cifar10_dataloaders(::Type{T}, batchsize; kwargs...) where {T}
    cifar10_mean = T.((0.4914, 0.4822, 0.4465))
    cifar10_std = T.((0.2471, 0.2435, 0.2616))

    train_transform =
        RandomResizeCrop((32, 32)) |>
        Maybe(FlipX{2}()) |>
        ImageToTensor() |>
        Normalize(cifar10_mean, cifar10_std) |>
        ToEltype(T)

    test_transform = ImageToTensor() |> Normalize(cifar10_mean, cifar10_std) |> ToEltype(T)

    trainset = TensorDataset(CIFAR10(; Tx=T, split=:train), train_transform)
    trainloader = DataLoader(trainset; batchsize, shuffle=true, kwargs...)

    testset = TensorDataset(CIFAR10(; Tx=T, split=:test), test_transform)
    testloader = DataLoader(testset; batchsize, shuffle=false, kwargs...)

    return trainloader, testloader
end

# ## Utility Functions

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

# ## Training Loop

function train_model(
    model,
    opt,
    scheduler=nothing;
    batchsize::Int=512,
    seed::Int=1234,
    epochs::Int=25,
    bfloat16::Bool=false,
)
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    prec = bfloat16 ? bf16 : f32
    prec_jl = bfloat16 ? BFloat16 : Float32
    prec_str = bfloat16 ? "BFloat16" : "Float32"
    @printf "[Info] Using %s precision\n" prec_str

    dev = reactant_device(; force=true)

    trainloader, testloader =
        get_cifar10_dataloaders(prec_jl, batchsize; partial=false) |> dev

    ps, st = prec(Lux.setup(rng, model)) |> dev

    train_state = Training.TrainState(model, ps, st, opt)

    x_ra = rand(rng, prec_jl, size(first(trainloader)[1])) |> dev
    @printf "[Info] Compiling model with Reactant.jl\n"
    model_compiled = @compile model(x_ra, ps, Lux.testmode(st))
    @printf "[Info] Model compiled!\n"

    loss_fn = CrossEntropyLoss(; logits=Val(true))

    pt = ProgressTable(;
        header=[
            "Epoch", "Learning Rate", "Train Accuracy (%)", "Test Accuracy (%)", "Time (s)"
        ],
        widths=[24, 24, 24, 24, 24],
        format=["%3d", "%.6f", "%.6f", "%.6f", "%.6f"],
        color=[:normal, :normal, :blue, :blue, :normal],
        border=true,
        alignment=[:center, :center, :center, :center, :center],
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
                AutoEnzyme(), loss_fn, (x, y), train_state; return_gradients=Val(false)
            )
            isnan(loss) && error("NaN loss encountered!")
        end
        ttime = time() - stime

        train_acc =
            accuracy(
                model_compiled,
                train_state.parameters,
                Lux.testmode(train_state.states),
                trainloader,
            ) * 100
        test_acc =
            accuracy(
                model_compiled,
                train_state.parameters,
                Lux.testmode(train_state.states),
                testloader,
            ) * 100

        scheduler === nothing && (lr = NaN32)
        next(pt, [epoch, lr, train_acc, test_acc, ttime])
    end

    finalize(pt)
    return @printf "[Info] Finished training\n"
end
