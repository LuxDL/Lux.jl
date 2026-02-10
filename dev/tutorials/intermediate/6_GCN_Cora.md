---
url: /dev/tutorials/intermediate/6_GCN_Cora.md
---
# Graph Convolutional Networks on Cora {#GCN-Tutorial-Cora}

This example is based on [GCN MLX tutorial](https://github.com/ml-explore/mlx-examples/blob/main/gcn/). While we are doing this manually, we recommend directly using [GNNLux.jl](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/).

```julia
using Lux,
    Reactant,
    MLDatasets,
    Random,
    Statistics,
    GNNGraphs,
    ConcreteStructs,
    Printf,
    OneHotArrays,
    Optimisers

const xdev = reactant_device(; force=true)
const cdev = cpu_device()
```

## Loading Cora Dataset {#Loading-Cora-Dataset}

```julia
function loadcora()
    data = Cora()
    gph = data.graphs[1]
    gnngraph = GNNGraph(
        gph.edge_index; ndata=gph.node_data, edata=gph.edge_data, gph.num_nodes
    )
    return (
        gph.node_data.features,
        onehotbatch(gph.node_data.targets, data.metadata["classes"]),
        # We use a dense matrix here to avoid incompatibility with Reactant
        Matrix{Int32}(adjacency_matrix(gnngraph)),
        # We use this since Reactant doesn't yet support gather adjoint
        (1:140, 141:640, 1709:2708),
    )
end
```

## Model Definition {#Model-Definition}

```julia
function GCNLayer(args...; kwargs...)
    return @compact(; dense=Dense(args...; kwargs...)) do (x, adj)
        @return dense(x) * adj
    end
end

function GCN(x_dim, h_dim, out_dim; nb_layers=2, dropout=0.5, kwargs...)
    layer_sizes = vcat(x_dim, [h_dim for _ in 1:nb_layers])
    gcn_layers = [
        GCNLayer(in_dim => out_dim; kwargs...) for
        (in_dim, out_dim) in zip(layer_sizes[1:(end - 1)], layer_sizes[2:end])
    ]
    last_layer = GCNLayer(layer_sizes[end] => out_dim; kwargs...)
    dropout = Dropout(dropout)

    return @compact(; gcn_layers, dropout, last_layer) do (x, adj, mask)
        for layer in gcn_layers
            x = relu.(layer((x, adj)))
            x = dropout(x)
        end
        @return last_layer((x, adj))[:, mask]
    end
end
```

## Helper Functions {#Helper-Functions}

```julia
function loss_function(model, ps, st, (x, y, adj, mask))
    y_pred, st = model((x, adj, mask), ps, st)
    loss = CrossEntropyLoss(; agg=mean, logits=Val(true))(y_pred, y[:, mask])
    return loss, st, (; y_pred)
end

accuracy(y_pred, y) = mean(onecold(y_pred) .== onecold(y)) * 100
```

## Training the Model {#Training-the-Model}

```julia
function main(;
    hidden_dim::Int=64,
    dropout::Float64=0.1,
    nb_layers::Int=2,
    use_bias::Bool=true,
    lr::Float64=0.001,
    weight_decay::Float64=0.0,
    patience::Int=20,
    epochs::Int=200,
)
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    features, targets, adj, (train_idx, val_idx, test_idx) = loadcora() |> xdev

    gcn = GCN(size(features, 1), hidden_dim, size(targets, 1); nb_layers, dropout, use_bias)
    ps, st = Lux.setup(rng, gcn) |> xdev
    opt = iszero(weight_decay) ? Adam(lr) : AdamW(; eta=lr, lambda=weight_decay)

    train_state = Training.TrainState(gcn, ps, st, opt)

    @printf "Total Trainable Parameters: %0.4f M\n" (Lux.parameterlength(ps) / 1.0e6)

    val_loss_compiled = @compile loss_function(
        gcn, ps, Lux.testmode(st), (features, targets, adj, val_idx)
    )

    train_model_compiled = @compile gcn((features, adj, train_idx), ps, Lux.testmode(st))
    val_model_compiled = @compile gcn((features, adj, val_idx), ps, Lux.testmode(st))

    best_loss_val = Inf
    cnt = 0

    for epoch in 1:epochs
        (_, loss, _, train_state) = Lux.Training.single_train_step!(
            AutoEnzyme(),
            loss_function,
            (features, targets, adj, train_idx),
            train_state;
            return_gradients=Val(false),
        )
        train_acc = accuracy(
            Array(
                train_model_compiled(
                    (features, adj, train_idx),
                    train_state.parameters,
                    Lux.testmode(train_state.states),
                )[1],
            ),
            Array(targets)[:, train_idx],
        )

        val_loss = first(
            val_loss_compiled(
                gcn,
                train_state.parameters,
                Lux.testmode(train_state.states),
                (features, targets, adj, val_idx),
            ),
        )
        val_acc = accuracy(
            Array(
                val_model_compiled(
                    (features, adj, val_idx),
                    train_state.parameters,
                    Lux.testmode(train_state.states),
                )[1],
            ),
            Array(targets)[:, val_idx],
        )

        @printf "Epoch %3d\tTrain Loss: %.6f\tTrain Acc: %.4f%%\tVal Loss: %.6f\t\
                 Val Acc: %.4f%%\n" epoch loss train_acc val_loss val_acc

        if val_loss < best_loss_val
            best_loss_val = val_loss
            cnt = 0
        else
            cnt += 1
            if cnt == patience
                @printf "Early Stopping at Epoch %d\n" epoch
                break
            end
        end
    end

    test_loss = @jit(
        loss_function(
            gcn,
            train_state.parameters,
            Lux.testmode(train_state.states),
            (features, targets, adj, test_idx),
        )
    )[1]
    test_acc = accuracy(
        Array(
            @jit(
                gcn(
                    (features, adj, test_idx),
                    train_state.parameters,
                    Lux.testmode(train_state.states),
                )
            )[1],
        ),
        Array(targets)[:, test_idx],
    )

    @printf "Test Loss: %.6f\tTest Acc: %.4f%%\n" test_loss test_acc
    return nothing
end

main()
```

```
┌ Warning: `replicate` doesn't work for `TaskLocalRNG`. Returning the same `TaskLocalRNG`.
└ @ LuxCore ~/work/Lux.jl/Lux.jl/lib/LuxCore/src/LuxCore.jl:18
Total Trainable Parameters: 0.0964 M
Epoch   1	Train Loss: 15.483308	Train Acc: 22.1429%	Val Loss: 7.571783	Val Acc: 25.8000%
Epoch   2	Train Loss: 10.125030	Train Acc: 22.1429%	Val Loss: 3.797886	Val Acc: 29.4000%
Epoch   3	Train Loss: 4.467243	Train Acc: 37.8571%	Val Loss: 2.431701	Val Acc: 32.0000%
Epoch   4	Train Loss: 2.424877	Train Acc: 51.4286%	Val Loss: 2.113642	Val Acc: 37.8000%
Epoch   5	Train Loss: 1.761382	Train Acc: 58.5714%	Val Loss: 1.889250	Val Acc: 45.0000%
Epoch   6	Train Loss: 1.484980	Train Acc: 67.8571%	Val Loss: 1.611183	Val Acc: 51.6000%
Epoch   7	Train Loss: 1.267712	Train Acc: 71.4286%	Val Loss: 1.504884	Val Acc: 58.4000%
Epoch   8	Train Loss: 1.319321	Train Acc: 72.1429%	Val Loss: 1.505576	Val Acc: 59.8000%
Epoch   9	Train Loss: 1.617086	Train Acc: 73.5714%	Val Loss: 1.520861	Val Acc: 61.2000%
Epoch  10	Train Loss: 1.249781	Train Acc: 74.2857%	Val Loss: 1.519172	Val Acc: 62.0000%
Epoch  11	Train Loss: 1.187690	Train Acc: 78.5714%	Val Loss: 1.504537	Val Acc: 62.0000%
Epoch  12	Train Loss: 1.179360	Train Acc: 78.5714%	Val Loss: 1.547555	Val Acc: 61.8000%
Epoch  13	Train Loss: 0.898748	Train Acc: 80.0000%	Val Loss: 1.608347	Val Acc: 62.0000%
Epoch  14	Train Loss: 0.946830	Train Acc: 80.0000%	Val Loss: 1.649865	Val Acc: 61.8000%
Epoch  15	Train Loss: 1.425961	Train Acc: 80.7143%	Val Loss: 1.633293	Val Acc: 64.4000%
Epoch  16	Train Loss: 0.875585	Train Acc: 82.1429%	Val Loss: 1.616587	Val Acc: 66.6000%
Epoch  17	Train Loss: 0.810615	Train Acc: 81.4286%	Val Loss: 1.592887	Val Acc: 67.0000%
Epoch  18	Train Loss: 0.763063	Train Acc: 80.7143%	Val Loss: 1.569996	Val Acc: 67.4000%
Epoch  19	Train Loss: 0.881349	Train Acc: 82.1429%	Val Loss: 1.543069	Val Acc: 67.2000%
Epoch  20	Train Loss: 0.750949	Train Acc: 82.8571%	Val Loss: 1.520200	Val Acc: 66.8000%
Epoch  21	Train Loss: 0.685395	Train Acc: 83.5714%	Val Loss: 1.504100	Val Acc: 66.6000%
Epoch  22	Train Loss: 0.611383	Train Acc: 85.0000%	Val Loss: 1.500499	Val Acc: 66.0000%
Epoch  23	Train Loss: 0.603166	Train Acc: 84.2857%	Val Loss: 1.511355	Val Acc: 66.2000%
Epoch  24	Train Loss: 1.565988	Train Acc: 85.7143%	Val Loss: 1.550028	Val Acc: 66.0000%
Epoch  25	Train Loss: 0.564262	Train Acc: 88.5714%	Val Loss: 1.616222	Val Acc: 64.6000%
Epoch  26	Train Loss: 0.524013	Train Acc: 87.8571%	Val Loss: 1.695767	Val Acc: 64.0000%
Epoch  27	Train Loss: 0.508034	Train Acc: 88.5714%	Val Loss: 1.788846	Val Acc: 64.0000%
Epoch  28	Train Loss: 0.621814	Train Acc: 87.8571%	Val Loss: 1.853111	Val Acc: 63.0000%
Epoch  29	Train Loss: 0.579144	Train Acc: 88.5714%	Val Loss: 1.872775	Val Acc: 63.2000%
Epoch  30	Train Loss: 0.491464	Train Acc: 88.5714%	Val Loss: 1.874164	Val Acc: 63.8000%
Epoch  31	Train Loss: 0.493937	Train Acc: 89.2857%	Val Loss: 1.847677	Val Acc: 64.6000%
Epoch  32	Train Loss: 0.562605	Train Acc: 90.0000%	Val Loss: 1.800509	Val Acc: 66.0000%
Epoch  33	Train Loss: 0.490371	Train Acc: 91.4286%	Val Loss: 1.742706	Val Acc: 66.0000%
Epoch  34	Train Loss: 0.623589	Train Acc: 91.4286%	Val Loss: 1.702445	Val Acc: 65.8000%
Epoch  35	Train Loss: 0.441532	Train Acc: 92.8571%	Val Loss: 1.669238	Val Acc: 66.2000%
Epoch  36	Train Loss: 0.414883	Train Acc: 92.1429%	Val Loss: 1.649799	Val Acc: 67.4000%
Epoch  37	Train Loss: 0.396852	Train Acc: 93.5714%	Val Loss: 1.642260	Val Acc: 68.0000%
Epoch  38	Train Loss: 0.370066	Train Acc: 93.5714%	Val Loss: 1.644972	Val Acc: 68.2000%
Epoch  39	Train Loss: 0.402366	Train Acc: 93.5714%	Val Loss: 1.657054	Val Acc: 68.6000%
Epoch  40	Train Loss: 0.802922	Train Acc: 95.7143%	Val Loss: 1.677369	Val Acc: 67.8000%
Epoch  41	Train Loss: 0.378652	Train Acc: 95.7143%	Val Loss: 1.707681	Val Acc: 68.0000%
Epoch  42	Train Loss: 0.366849	Train Acc: 95.0000%	Val Loss: 1.735516	Val Acc: 68.2000%
Early Stopping at Epoch 42
Test Loss: 1.518861	Test Acc: 68.8000%

```

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
