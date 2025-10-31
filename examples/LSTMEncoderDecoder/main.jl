# # Building a LSTM Encoder-Decoder model using Lux.jl

# This examples is based on [LSTM_encoder_decoder](https://github.com/lkulowski/LSTM_encoder_decoder)
# by [Laura Kulowski](https://github.com/lkulowski).

using Lux, Reactant, Random, Optimisers, Statistics, Enzyme, Printf, CairoMakie, MLUtils

const xdev = reactant_device(; force=true)
const cdev = cpu_device()
nothing #hide

# ## Generate synthetic data

function synthetic_data(Nt=2000, tf=80 * Float32(π))
    t = range(0.0f0, tf; length=Nt)
    y = sin.(2.0f0 * t) .+ 0.5f0 * cos.(t) .+ randn(Float32, Nt) * 0.2f0
    return t, y
end

function train_test_split(t, y, split=0.8)
    indx_split = ceil(Int, length(t) * split)
    indx_train = 1:indx_split
    indx_test = (indx_split + 1):length(t)

    t_train = t[indx_train]
    y_train = reshape(y[indx_train], 1, :)

    t_test = t[indx_test]
    y_test = reshape(y[indx_test], 1, :)

    return t_train, y_train, t_test, y_test
end

function windowed_dataset(y; input_window=5, output_window=1, stride=1, num_features=1)
    L = size(y, ndims(y))
    num_samples = (L - input_window - output_window) ÷ stride + 1

    X = zeros(Float32, num_features, input_window, num_samples)
    Y = zeros(Float32, num_features, output_window, num_samples)

    for ii in 1:num_samples, ff in 1:num_features
        start_x = stride * (ii - 1) + 1
        end_x = start_x + input_window - 1
        X[ff, :, ii] .= y[start_x:end_x]

        start_y = stride * (ii - 1) + input_window + 1
        end_y = start_y + output_window - 1
        Y[ff, :, ii] .= y[start_y:end_y]
    end

    return X, Y
end

t, y = synthetic_data()

begin
    fig = Figure(; size=(1000, 400))
    ax = Axis(fig[1, 1]; title="Synthetic Time Series", xlabel="t", ylabel="y")

    lines!(ax, t, y; label="y", color=:black, linewidth=2)

    fig
end

# ---

t_train, y_train, t_test, y_test = train_test_split(t, y)

begin
    fig = Figure(; size=(1000, 400))
    ax = Axis(
        fig[1, 1];
        title="Time Series Split into Train and Test Sets",
        xlabel="t",
        ylabel="y",
    )

    lines!(ax, t_train, y_train[1, :]; label="Train", color=:black, linewidth=2)
    lines!(ax, t_test, y_test[1, :]; label="Test", color=:red, linewidth=2)

    fig[1, 2] = Legend(fig, ax)

    fig
end

# ---

X_train, Y_train = windowed_dataset(y_train; input_window=80, output_window=20, stride=5)
X_test, Y_test = windowed_dataset(y_test; input_window=80, output_window=20, stride=5)

begin
    fig = Figure(; size=(1000, 400))
    ax = Axis(fig[1, 1]; title="Example of Windowed Training Data", xlabel="t", ylabel="y")

    linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]

    for b in 1:4:16
        lines!(
            ax,
            0:79,
            X_train[1, :, b];
            label="Input",
            color=:black,
            linewidth=2,
            linestyle=linestyles[mod1(b, 5)],
        )
        lines!(
            ax,
            79:99,
            vcat(X_train[1, end, b], Y_train[1, :, b]);
            label="Target",
            color=:red,
            linewidth=2,
            linestyle=linestyles[mod1(b, 5)],
        )
    end

    fig
end

# ## Define the model

struct RNNEncoder{C} <: AbstractLuxWrapperLayer{:cell}
    cell::C
end

function (rnn::RNNEncoder)(x::AbstractArray{T,3}, ps, st) where {T}
    (y, carry), st = Lux.apply(rnn.cell, x[:, 1, :], ps, st)
    @trace for i in 2:size(x, 2)
        (y, carry), st = Lux.apply(rnn.cell, (x[:, i, :], carry), ps, st)
    end
    return (y, carry), st
end

struct RNNDecoder{C,L} <: AbstractLuxContainerLayer{(:cell, :linear)}
    cell::C
    linear::L
    training_mode::Symbol
    teacher_forcing_ratio::Float32

    function RNNDecoder(
        cell::C,
        linear::L;
        training_mode::Symbol=:recursive,
        teacher_forcing_ratio::Float32=0.5f0,
    ) where {C,L}
        @assert training_mode in (:recursive, :teacher_forcing, :mixed_teacher_forcing)
        return new{C,L}(cell, linear, training_mode, Float32(teacher_forcing_ratio))
    end
end

function LuxCore.initialstates(rng::AbstractRNG, d::RNNDecoder)
    return (;
        cell=LuxCore.initialstates(rng, d.cell),
        linear=LuxCore.initialstates(rng, d.linear),
        training=Val(true),
        rng,
    )
end

function _teacher_forcing_condition(::Val{false}, x, mode, rng, ratio, target_len)
    res = similar(x, Bool, target_len)
    fill!(res, true)
    return res
end
function _teacher_forcing_condition(::Val{true}, x, mode, rng, ratio, target_len)
    mode === :recursive &&
        return _teacher_forcing_condition(Val(false), x, mode, rng, ratio, target_len)
    mode === :teacher_forcing && fill(rand(rng, Float32) < ratio, target_len)
    return rand(rng, Float32, target_len) .< ratio
end

function (rnn::RNNDecoder)((decoder_input, carry, target_len, target), ps, st)
    @assert ndims(decoder_input) == 2
    rng = Lux.replicate(st.rng)

    if target === nothing
        ### This will be optimized out by Reactant
        target = similar(
            decoder_input, size(decoder_input, 1), target_len, size(decoder_input, 3)
        )
        fill!(target, 0)
    else
        @assert size(target, 2) ≤ target_len
    end

    (y_latent, carry), st_cell = Lux.apply(
        rnn.cell, (decoder_input, carry), ps.cell, st.cell
    )
    y_pred, st_linear = Lux.apply(rnn.linear, y_latent, ps.linear, st.linear)

    y_full = similar(y_pred, size(y_pred, 1), target_len, size(y_pred, 2))
    y_full[:, 1, :] = y_pred

    conditions = _teacher_forcing_condition(
        st.training,
        decoder_input,
        rnn.training_mode,
        rng,
        rnn.teacher_forcing_ratio,
        target_len,
    )
    decoder_input = ifelse.(@allowscalar(conditions[1]), target[:, 1, :], y_pred)

    @trace for i in 2:target_len
        (y_latent, carry), st_cell = Lux.apply(
            rnn.cell, (decoder_input, carry), ps.cell, st_cell
        )

        y_pred, st_linear = Lux.apply(rnn.linear, y_latent, ps.linear, st_linear)
        y_full[:, i, :] = y_pred

        decoder_input = ifelse.(@allowscalar(conditions[i]), target[:, i, :], y_pred)
    end

    return y_full, merge(st, (; cell=st_cell, linear=st_linear, rng))
end

struct RNNEncoderDecoder{C<:RNNEncoder,L<:RNNDecoder} <:
       AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder::C
    decoder::L
end

function (rnn::RNNEncoderDecoder)((x, target_len, target), ps, st)
    (y, carry), st_encoder = Lux.apply(rnn.encoder, x, ps.encoder, st.encoder)
    pred, st_decoder = Lux.apply(
        rnn.decoder, (x[:, end, :], carry, target_len, target), ps.decoder, st.decoder
    )
    return pred, (; encoder=st_encoder, decoder=st_decoder)
end

# ## Training

function train(
    train_dataset,
    validation_dataset;
    nepochs=50,
    batchsize=32,
    hidden_dims=32,
    training_mode=:mixed_teacher_forcing,
    teacher_forcing_ratio=0.5f0,
    learning_rate=1e-3,
)
    (X_train, Y_train), (X_test, Y_test) = train_dataset, validation_dataset
    in_dims = size(X_train, 1)
    @assert size(Y_train, 2) == size(Y_test, 2)
    target_len = size(Y_train, 2)

    train_dataloader =
        DataLoader(
            (X_train, Y_train);
            batchsize=min(batchsize, size(X_train, 4)),
            shuffle=true,
            partial=false,
        ) |> xdev
    X_test, Y_test = (X_test, Y_test) |> xdev

    model = RNNEncoderDecoder(
        RNNEncoder(LSTMCell(in_dims => hidden_dims)),
        RNNDecoder(
            LSTMCell(in_dims => hidden_dims),
            Dense(hidden_dims => in_dims);
            training_mode,
            teacher_forcing_ratio,
        ),
    )
    ps, st = Lux.setup(Random.default_rng(), model) |> xdev

    train_state = Training.TrainState(model, ps, st, Optimisers.Adam(learning_rate))

    stime = time()
    model_compiled = @compile model((X_test, target_len, nothing), ps, Lux.testmode(st))
    ttime = time() - stime
    @printf "Compilation time: %.4f seconds\n\n" ttime

    for epoch in 1:nepochs
        stime = time()
        for (x, y) in train_dataloader
            (_, _, _, train_state) = Training.single_train_step!(
                AutoEnzyme(),
                MSELoss(),
                ((x, target_len, y), y),
                train_state;
                return_gradients=Val(false),
            )
        end
        ttime = time() - stime

        y_pred, _ = model_compiled(
            (X_test, target_len, nothing),
            train_state.parameters,
            Lux.testmode(train_state.states),
        )
        pred_loss = Float32(@jit(MSELoss()(y_pred, Y_test)))

        @printf(
            "[%3d/%3d]\tTime per epoch: %3.5fs\tValidation Loss: %.4f\n",
            epoch,
            nepochs,
            ttime,
            pred_loss,
        )
    end

    return StatefulLuxLayer(
        model, train_state.parameters |> cdev, train_state.states |> cdev
    )
end

trained_model = train(
    (X_train, Y_train),
    (X_test, Y_test);
    nepochs=50,
    batchsize=4,
    hidden_dims=32,
    training_mode=:mixed_teacher_forcing,
    teacher_forcing_ratio=0.5f0,
    learning_rate=3e-4,
)

# ## Making Predictions

Y_pred = trained_model((X_test, 20, nothing))

begin
    fig = Figure(; size=(1200, 800))

    for i in 1:4, j in 1:2
        b = i + j * 4
        ax = Axis(fig[i, j]; xlabel="t", ylabel="y")
        i != 4 && hidexdecorations!(ax; grid=false)
        j != 1 && hideydecorations!(ax; grid=false)

        lines!(ax, 0:79, X_test[1, :, b]; label="Input", color=:black, linewidth=2)
        lines!(
            ax,
            79:99,
            vcat(X_test[1, end, b], Y_test[1, :, b]);
            label="Ground Truth\n(Noisy)",
            color=:red,
            linewidth=2,
        )
        lines!(
            ax,
            79:99,
            vcat(X_test[1, end, b], Y_pred[1, :, b]);
            label="Prediction",
            color=:blue,
            linewidth=2,
        )

        i == 4 && j == 2 && axislegend(ax; position=:lb)
    end

    fig[0, :] = Label(fig, "Predictions from Trained Model"; fontsize=20)

    fig
end
