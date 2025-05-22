# # Building a LSTM Encoder-Decoder model using Lux.jl

# This examples is based on [LSTM_encoder_decoder](https://github.com/lkulowski/LSTM_encoder_decoder)
# by [Laura Kulowski](https://github.com/lkulowski).

using Lux, Reactant, Random, Optimisers, Statistics, Enzyme, Printf, CairoMakie

const xdev = reactant_device(; force=true)
const cdev = cpu_device()

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

function (rnn::RNNEncoder)(x::AbstractArray{T, 3}, ps, st) where {T}
    (y, carry), st = Lux.apply(rnn.cell, view(x, :, 1, :), ps.cell, st.cell)
    @trace for i in 2:size(x, 2)
        (y, carry), st = Lux.apply(rnn.cell, (view(x, :, i, :), carry), ps.cell, st.cell)
    end
    return (y, carry), st
end
