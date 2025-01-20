# # [Graph Convolutional Networks on Cora](@id GCN-Tutorial-Cora)

# This example is based on [GCN MLX tutorial](https://github.com/ml-explore/mlx-examples/blob/main/gcn/). While we are doing this manually, we recommend directly using
# [GNNLux.jl](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/).

using Lux, Reactant, MLDatasets, Random, Statistics, Enzyme, GNNGraphs, ConcreteStructs,
      Printf, OneHotArrays, Optimisers

const xdev = reactant_device(; force=true)
const cdev = cpu_device()

# ## Loading Cora Dataset

function loadcora()
    data = Cora()
    gph = data.graphs[1]
    gnngraph = GNNGraph(
        gph.edge_index; ndata=gph.node_data, edata=gph.edge_data, gph.num_nodes
    )
    return (
        gph.node_data.features,
        onehotbatch(gph.node_data.targets, data.metadata["classes"]),
        ## We use a dense matrix here to avoid incompatibility with Reactant
        Matrix(adjacency_matrix(gnngraph)),
        ## We use this since Reactant doesn't yet support gather adjoint
        (1:140, 141:640, 1709:2708)
    )
end

# ## Model Definition

function GCNLayer(args...; kwargs...)
    return @compact(; dense=Dense(args...; kwargs...)) do (x, adj)
        @return dense(x) * adj
    end
end

function GCN(x_dim, h_dim, out_dim; nb_layers=2, dropout=0.5, kwargs...)
    layer_sizes = vcat(x_dim, [h_dim for _ in 1:nb_layers])
    gcn_layers = [GCNLayer(in_dim => out_dim; kwargs...)
                  for (in_dim, out_dim) in zip(layer_sizes[1:(end - 1)], layer_sizes[2:end])]
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

# ## Helper Functions

function loss_function(model, ps, st, (x, y, adj, mask))
    y_pred, st = model((x, adj, mask), ps, st)
    loss = CrossEntropyLoss(; agg=mean, logits=Val(true))(y_pred, y[:, mask])
    return loss, st, (; y_pred)
end

accuracy(y_pred, y) = mean(onecold(y_pred) .== onecold(y)) * 100

# ## Training the Model

function main(;
        hidden_dim::Int=64, dropout::Float64=0.1, nb_layers::Int=2, use_bias::Bool=true,
        lr::Float64=0.001, weight_decay::Float64=0.0, patience::Int=20, epochs::Int=200
)
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    features, targets, adj, (train_idx, val_idx, test_idx) = loadcora() |> xdev

    gcn = GCN(size(features, 1), hidden_dim, size(targets, 1); nb_layers, dropout, use_bias)
    ps, st = Lux.setup(rng, gcn) |> xdev
    opt = iszero(weight_decay) ? Adam(lr) : AdamW(; eta=lr, lambda=weight_decay)

    train_state = Training.TrainState(gcn, ps, st, opt)

    @printf "Total Trainable Parameters: %0.4f M\n" (Lux.parameterlength(ps)/1e6)

    val_loss_compiled = @compile loss_function(
        gcn, ps, Lux.testmode(st), (features, targets, adj, val_idx))

    train_model_compiled = @compile gcn((features, adj, train_idx), ps, Lux.testmode(st))
    val_model_compiled = @compile gcn((features, adj, val_idx), ps, Lux.testmode(st))

    best_loss_val = Inf
    cnt = 0

    for epoch in 1:epochs
        (_, loss, _, train_state) = Lux.Training.single_train_step!(
            AutoEnzyme(), loss_function, (features, targets, adj, train_idx), train_state;
            return_gradients=Val(false)
        )
        train_acc = accuracy(
            Array(train_model_compiled((features, adj, train_idx),
                train_state.parameters, Lux.testmode(train_state.states))[1]),
            Array(targets)[:, train_idx]
        )

        val_loss = first(val_loss_compiled(
            gcn, train_state.parameters, Lux.testmode(train_state.states),
            (features, targets, adj, val_idx)))
        val_acc = accuracy(
            Array(val_model_compiled((features, adj, val_idx),
                train_state.parameters, Lux.testmode(train_state.states))[1]),
            Array(targets)[:, val_idx]
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

    test_loss = @jit(loss_function(
        gcn, train_state.parameters, Lux.testmode(train_state.states),
        (features, targets, adj, test_idx)))[1]
    test_acc = accuracy(
        Array(@jit(gcn((features, adj, test_idx),
            train_state.parameters, Lux.testmode(train_state.states)))[1]),
        Array(targets)[:, test_idx]
    )

    @printf "Test Loss: %.6f\tTest Acc: %.4f%%\n" test_loss test_acc
    return
end

main()
nothing #hide
