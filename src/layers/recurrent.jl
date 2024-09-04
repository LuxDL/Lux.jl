abstract type AbstractRecurrentCell <: AbstractExplicitLayer end

const AbstractDebugRecurrentCell = Experimental.DebugLayer{
    <:Any, <:Any, <:AbstractRecurrentCell}

# Fallback for vector inputs
function (rnn::AbstractRecurrentCell)(x::AbstractVector, ps, st::NamedTuple)
    (y, carry), stₙ = rnn(reshape(x, :, 1), ps, st)
    return (vec(y), vec.(carry)), stₙ
end

function (rnn::AbstractRecurrentCell)((x, carry), ps, st::NamedTuple)
    xₙ = reshape(x, :, 1)
    carryₙ = map(Base.Fix2(reshape, (:, 1)), carry)
    (y, carry₂), stₙ = rnn((xₙ, carryₙ), ps, st)
    return (vec(y), vec.(carry₂)), stₙ
end

abstract type AbstractTimeSeriesDataBatchOrdering end

struct TimeLastIndex <: AbstractTimeSeriesDataBatchOrdering end
struct BatchLastIndex <: AbstractTimeSeriesDataBatchOrdering end

LuxOps.eachslice(x::AbstractArray, ::TimeLastIndex) = LuxOps.eachslice(x, Val(ndims(x)))
function LuxOps.eachslice(x::AbstractArray, ::BatchLastIndex)
    return LuxOps.eachslice(x, Val(ndims(x) - 1))
end
LuxOps.eachslice(x::AbstractMatrix, ::BatchLastIndex) = LuxOps.eachslice(x, Val(ndims(x)))

"""
    Recurrence(cell;
        ordering::AbstractTimeSeriesDataBatchOrdering=BatchLastIndex(),
        return_sequence::Bool=false)

Wraps a recurrent cell (like [`RNNCell`](@ref), [`LSTMCell`](@ref), [`GRUCell`](@ref)) to
automatically operate over a sequence of inputs.

!!! warning "Relation to `Flux.Recur`"

    This is completely distinct from `Flux.Recur`. It doesn't make the `cell` stateful,
    rather allows operating on an entire sequence of inputs at once. See
    [`StatefulRecurrentCell`](@ref) for functionality similar to `Flux.Recur`.

## Arguments

  - `cell`: A recurrent cell. See [`RNNCell`](@ref), [`LSTMCell`](@ref), [`GRUCell`](@ref),
    for how the inputs/outputs of a recurrent cell must be structured.

## Keyword Arguments

  - `return_sequence`: If `true` returns the entire sequence of outputs, else returns only
    the last output. Defaults to `false`.
  - `ordering`: The ordering of the batch and time dimensions in the input. Defaults to
    `BatchLastIndex()`. Alternatively can be set to `TimeLastIndex()`.

# Extended Help

## Inputs

  - If `x` is a

      + Tuple or Vector: Each element is fed to the `cell` sequentially.

      + Array (except a Vector): It is spliced along the penultimate dimension and each
        slice is fed to the `cell` sequentially.

## Returns

  - Output of the `cell` for the entire sequence.
  - Update state of the `cell`.

!!! tip

    Frameworks like Tensorflow have special implementation of
    [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/MultiRNNCell)
    to handle sequentially composed RNN Cells. In Lux, one can simple stack multiple
    `Recurrence` blocks in a `Chain` to achieve the same.

        Chain(
            Recurrence(RNNCell(inputsize => latentsize); return_sequence=true),
            Recurrence(RNNCell(latentsize => latentsize); return_sequence=true),
            :
            x -> stack(x; dims=2)
        )

    For some discussion on this topic, see https://github.com/LuxDL/Lux.jl/issues/472.
"""
@concrete struct Recurrence{R <: StaticBool} <: AbstractExplicitContainerLayer{(:cell,)}
    cell <: Union{<:AbstractRecurrentCell, <:AbstractDebugRecurrentCell}
    ordering <: AbstractTimeSeriesDataBatchOrdering
    return_sequence::R
end

function Recurrence(cell; ordering::AbstractTimeSeriesDataBatchOrdering=BatchLastIndex(),
        return_sequence::Bool=false)
    return Recurrence(cell, ordering, static(return_sequence))
end

function (r::Recurrence)(x::AbstractArray, ps, st::NamedTuple)
    return apply(r, safe_eachslice(x, r.ordering), ps, st)
end

function (r::Recurrence{False})(x::Union{AbstractVector, NTuple}, ps, st::NamedTuple)
    (out, carry), st = apply(r.cell, first(x), ps, st)
    for xᵢ in x[(begin + 1):end]
        (out, carry), st = apply(r.cell, (xᵢ, carry), ps, st)
    end
    return out, st
end

function (r::Recurrence{True})(x::Union{AbstractVector, NTuple}, ps, st::NamedTuple)
    function recur_op(::Nothing, input)
        (out, carry), state = apply(r.cell, input, ps, st)
        return [out], carry, state
    end
    function recur_op((outputs, carry, state), input)
        (out, carry), state = apply(r.cell, (input, carry), ps, state)
        return vcat(outputs, [out]), carry, state
    end
    results = private_foldl_init(recur_op, x)
    return first(results), last(results)
end

"""
    StatefulRecurrentCell(cell)

Wraps a recurrent cell (like [`RNNCell`](@ref), [`LSTMCell`](@ref), [`GRUCell`](@ref)) and
makes it stateful.

To avoid undefined behavior, once the processing of a single sequence of data is complete,
update the state with `Lux.update_state(st, :carry, nothing)`.

## Arguments

  - `cell`: A recurrent cell. See [`RNNCell`](@ref), [`LSTMCell`](@ref), [`GRUCell`](@ref),
    for how the inputs/outputs of a recurrent cell must be structured.

## Inputs

  - Input to the `cell`.

## Returns

  - Output of the `cell` for the entire sequence.
  - Update state of the `cell` and updated `carry`.

## States

  - NamedTuple containing:

      + `cell`: Same as `cell`.
      + `carry`: The carry state of the `cell`.
"""
@concrete struct StatefulRecurrentCell <: AbstractExplicitContainerLayer{(:cell,)}
    cell <: Union{<:AbstractRecurrentCell, <:AbstractDebugRecurrentCell}
end

function initialstates(rng::AbstractRNG, r::StatefulRecurrentCell)
    return (cell=initialstates(rng, r.cell), carry=nothing)
end

function (r::StatefulRecurrentCell)(x, ps, st::NamedTuple)
    (out, carry), stₙ = applyrecurrentcell(r.cell, x, ps, st.cell, st.carry)
    return out, (; cell=stₙ, carry)
end

function applyrecurrentcell(l::AbstractRecurrentCell, x, ps, st, carry)
    return apply(l, (x, carry), ps, st)
end
applyrecurrentcell(l::AbstractRecurrentCell, x, ps, st, ::Nothing) = apply(l, x, ps, st)

@doc doc"""
    RNNCell(in_dims => out_dims, activation=tanh; use_bias=True(), train_state=False(),
        init_bias=zeros32, init_weight=glorot_uniform, init_state=ones32)

An Elman RNNCell cell with `activation` (typically set to `tanh` or `relu`).

``h_{new} = activation(weight_{ih} \times x + weight_{hh} \times h_{prev} + bias)``

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State) Dimension
  - `activation`: Activation function
  - `use_bias`: Set to false to deactivate bias
  - `train_state`: Trainable initial hidden state can be activated by setting this to `true`
  - `init_bias`: Initializer for bias
  - `init_weight`: Initializer for weight
  - `init_state`: Initializer for hidden state

## Inputs

  - Case 1a: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `false` - Creates a hidden state using `init_state` and proceeds to Case 2.
  - Case 1b: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `true` - Repeats `hidden_state` from parameters to match the shape of `x`
             and proceeds to Case 2.
  - Case 2: Tuple `(x, (h, ))` is provided, then the output and a tuple containing the
            updated hidden state is returned.

## Returns
  - Tuple containing

      + Output ``h_{new}`` of shape `(out_dims, batch_size)`
      + Tuple containing new hidden state ``h_{new}``

  - Updated model state

## Parameters

  - `weight_ih`: Maps the input to the hidden state.
  - `weight_hh`: Maps the hidden state to the hidden state.
  - `bias`: Bias vector (not present if `use_bias=false`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation
"""
@concrete struct RNNCell <: AbstractRecurrentCell
    train_state <: StaticBool
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_weight
    init_state
    use_bias <: StaticBool
end

function RNNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh;
        use_bias::BoolType=True(), train_state::BoolType=False(),
        init_bias=zeros32, init_weight=glorot_uniform, init_state=ones32)
    return RNNCell(static(train_state), activation, in_dims, out_dims,
        init_bias, init_weight, init_state, static(use_bias))
end

function initialparameters(rng::AbstractRNG, rnn::RNNCell)
    ps = (weight_ih=rnn.init_weight(rng, rnn.out_dims, rnn.in_dims),
        weight_hh=rnn.init_weight(rng, rnn.out_dims, rnn.out_dims))
    has_bias(rnn) && (ps = merge(ps, (bias=rnn.init_bias(rng, rnn.out_dims),)))
    has_train_state(rnn) &&
        (ps = merge(ps, (hidden_state=rnn.init_state(rng, rnn.out_dims),)))
    return ps
end

initialstates(rng::AbstractRNG, ::RNNCell) = (rng=Utils.sample_replicate(rng),)

function (rnn::RNNCell{False})(x::AbstractMatrix, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = Utils.init_hidden_state(rng, rnn, x)
    return rnn((x, (hidden_state,)), ps, merge(st, (; rng)))
end

function (rnn::RNNCell{True})(x::AbstractMatrix, ps, st::NamedTuple)
    hidden_state = Utils.init_trainable_hidden_state(ps.hidden_state, x)
    return rnn((x, (hidden_state,)), ps, st)
end

function (rnn::RNNCell)(
        (x, (hidden_state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    y, hidden_stateₙ = match_eltype(rnn, ps, st, x, hidden_state)
    bias = safe_getproperty(ps, Val(:bias))
    z = fused_dense_bias_activation(identity, ps.weight_hh, hidden_stateₙ, bias)
    hₙ = fast_activation!!(rnn.activation, LuxLib.Impl.matmul(ps.weight_ih, y) .+ z)
    return (hₙ, (hₙ,)), st
end

function Base.show(io::IO, r::RNNCell)
    print(io, "RNNCell($(r.in_dims) => $(r.out_dims)")
    (r.activation == identity) || print(io, ", $(r.activation)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end

@doc doc"""
    LSTMCell(in_dims => out_dims; use_bias::Bool=true, train_state::Bool=false,
             train_memory::Bool=false,
             init_weight=(glorot_uniform, glorot_uniform, glorot_uniform, glorot_uniform),
             init_bias=(zeros32, zeros32, ones32, zeros32), init_state=zeros32,
             init_memory=zeros32)

Long Short-Term (LSTM) Cell

```math
\begin{align}
  i &= \sigma(W_{ii} \times x + W_{hi} \times h_{prev} + b_{i})\\
  f &= \sigma(W_{if} \times x + W_{hf} \times h_{prev} + b_{f})\\
  g &= tanh(W_{ig} \times x + W_{hg} \times h_{prev} + b_{g})\\
  o &= \sigma(W_{io} \times x + W_{ho} \times h_{prev} + b_{o})\\
  c_{new} &= f \cdot c_{prev} + i \cdot g\\
  h_{new} &= o \cdot tanh(c_{new})
\end{align}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension
  - `use_bias`: Set to false to deactivate bias
  - `train_state`: Trainable initial hidden state can be activated by setting this to `true`
  - `train_memory`: Trainable initial memory can be activated by setting this to `true`
  - `init_bias`: Initializer for bias. Must be a tuple containing 4 functions
  - `init_weight`: Initializer for weight. Must be a tuple containing 4 functions
  - `init_state`: Initializer for hidden state
  - `init_memory`: Initializer for memory

## Inputs

  - Case 1a: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `false`, `train_memory` is set to `false` - Creates a hidden state using
             `init_state`, hidden memory using `init_memory` and proceeds to Case 2.
  - Case 1b: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `true`, `train_memory` is set to `false` - Repeats `hidden_state` vector
             from the parameters to match the shape of `x`, creates hidden memory using
             `init_memory` and proceeds to Case 2.
  - Case 1c: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `false`, `train_memory` is set to `true` - Creates a hidden state using
             `init_state`, repeats the memory vector from parameters to match the shape of
             `x` and proceeds to Case 2.
  - Case 1d: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `true`, `train_memory` is set to `true` - Repeats the hidden state and
             memory vectors from the parameters to match the shape of  `x` and proceeds to
             Case 2.
  - Case 2: Tuple `(x, (h, c))` is provided, then the output and a tuple containing the 
            updated hidden state and memory is returned.

## Returns

  - Tuple Containing

      + Output ``h_{new}`` of shape `(out_dims, batch_size)`
      + Tuple containing new hidden state ``h_{new}`` and new memory ``c_{new}``

  - Updated model state

## Parameters

  - `weight_i`: Concatenated Weights to map from input space
                ``\{ W_{ii}, W_{if}, W_{ig}, W_{io} \}``.
  - `weight_h`: Concatenated Weights to map from hidden space
                ``\{ W_{hi}, W_{hf}, W_{hg}, W_{ho} \}``
  - `bias`: Bias vector (not present if `use_bias=false`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  - `memory`: Initial memory vector (not present if `train_memory=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation
"""
@concrete struct LSTMCell <: AbstractRecurrentCell
    train_state <: StaticBool
    train_memory <: StaticBool
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_weight
    init_state
    init_memory
    use_bias <: StaticBool
end

function LSTMCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(),
        train_memory::BoolType=False(), init_weight=glorot_uniform,
        init_bias=zeros32, init_state=zeros32, init_memory=zeros32)
    init_weight isa NTuple{4} || (init_weight = ntuple(Returns(init_weight), 4))
    init_bias isa NTuple{4} || (init_bias = ntuple(Returns(init_bias), 4))
    return LSTMCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_weight, init_state, init_memory, static(use_bias))
end

function initialparameters(rng::AbstractRNG, lstm::LSTMCell)
    weight_i = vcat([init_weight(rng, lstm.out_dims, lstm.in_dims)
                     for init_weight in lstm.init_weight]...)
    weight_h = vcat([init_weight(rng, lstm.out_dims, lstm.out_dims)
                     for init_weight in lstm.init_weight]...)
    ps = (; weight_i, weight_h)
    if has_bias(lstm)
        # TODO: in v1 we make this a flat vector
        bias = vcat([init_bias(rng, lstm.out_dims, 1) for init_bias in lstm.init_bias]...)
        ps = merge(ps, (bias=bias,))
    end
    has_train_state(lstm) &&
        (ps = merge(ps, (hidden_state=lstm.init_state(rng, lstm.out_dims),)))
    known(lstm.train_memory) &&
        (ps = merge(ps, (memory=lstm.init_memory(rng, lstm.out_dims),)))
    return ps
end

initialstates(rng::AbstractRNG, ::LSTMCell) = (rng=Utils.sample_replicate(rng),)

function (lstm::LSTMCell{False, False})(x::AbstractMatrix, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = Utils.init_hidden_state(rng, lstm, x)
    memory = Utils.init_hidden_state(rng, lstm, x)
    return lstm((x, (hidden_state, memory)), ps, merge(st, (; rng)))
end

function (lstm::LSTMCell{True, False})(x::AbstractMatrix, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = Utils.init_trainable_hidden_state(ps.hidden_state, x)
    memory = Utils.init_hidden_state(rng, lstm, x)
    return lstm((x, (hidden_state, memory)), ps, merge(st, (; rng)))
end

function (lstm::LSTMCell{False, True})(x::AbstractMatrix, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = Utils.init_hidden_state(rng, lstm, x)
    memory = Utils.init_trainable_hidden_state(ps.memory, x)
    return lstm((x, (hidden_state, memory)), ps, merge(st, (; rng)))
end

function (lstm::LSTMCell{True, True})(x::AbstractMatrix, ps, st::NamedTuple)
    hidden_state = Utils.init_trainable_hidden_state(ps.hidden_state, x)
    memory = Utils.init_trainable_hidden_state(ps.memory, x)
    return lstm((x, (hidden_state, memory)), ps, st)
end

const _LSTMCellInputType = Tuple{
    <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}}

function (lstm::LSTMCell)(
        (x, (hidden_state, memory))::_LSTMCellInputType, ps, st::NamedTuple)
    y, hidden_stateₙ, memoryₙ = match_eltype(lstm, ps, st, x, hidden_state, memory)
    bias = safe_vec(safe_getproperty(ps, Val(:bias)))
    z = fused_dense_bias_activation(identity, ps.weight_h, hidden_stateₙ, bias)
    g = LuxLib.Impl.matmul(ps.weight_i, y) .+ z

    input, forget, cell, output = multigate(g, Val(4))
    memory₂ = @. sigmoid_fast(forget) * memoryₙ + sigmoid_fast(input) * tanh_fast(cell)
    hidden_state₂ = @. sigmoid_fast(output) * tanh_fast(memory₂)
    return (hidden_state₂, (hidden_state₂, memory₂)), st
end

function Base.show(io::IO, lstm::LSTMCell)
    print(io, "LSTMCell($(lstm.in_dims) => $(lstm.out_dims)")
    has_bias(lstm) || print(io, ", use_bias=false")
    has_train_state(lstm) && print(io, ", train_state=true")
    known(lstm.train_memory) && print(io, ", train_memory=true")
    print(io, ")")
end

@doc doc"""
    GRUCell((in_dims, out_dims)::Pair{<:Int,<:Int}; use_bias=true, train_state::Bool=false,
            init_weight::Tuple{Function,Function,Function}=(glorot_uniform, glorot_uniform,
                                                            glorot_uniform),
            init_bias::Tuple{Function,Function,Function}=(zeros32, zeros32, zeros32),
            init_state::Function=zeros32)

Gated Recurrent Unit (GRU) Cell

```math
\begin{align}
  r &= \sigma(W_{ir} \times x + W_{hr} \times h_{prev} + b_{hr})\\
  z &= \sigma(W_{iz} \times x + W_{hz} \times h_{prev} + b_{hz})\\
  n &= \tanh(W_{in} \times x + b_{in} + r \cdot (W_{hn} \times h_{prev} + b_{hn}))\\
  h_{new} &= (1 - z) \cdot n + z \cdot h_{prev}
\end{align}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State) Dimension
  - `use_bias`: Set to false to deactivate bias
  - `train_state`: Trainable initial hidden state can be activated by setting this to `true`
  - `init_bias`: Initializer for bias. Must be a tuple containing 3 functions
  - `init_weight`: Initializer for weight. Must be a tuple containing 3 functions
  - `init_state`: Initializer for hidden state

## Inputs

  - Case 1a: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `false` - Creates a hidden state using `init_state` and proceeds to Case 2.
  - Case 1b: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `true` - Repeats `hidden_state` from parameters to match the shape of `x`
             and proceeds to Case 2.
  - Case 2: Tuple `(x, (h, ))` is provided, then the output and a tuple containing the 
            updated hidden state is returned.

## Returns

  - Tuple containing

      + Output ``h_{new}`` of shape `(out_dims, batch_size)`
      + Tuple containing new hidden state ``h_{new}``

  - Updated model state

## Parameters

  - `weight_i`: Concatenated Weights to map from input space
                ``\{ W_{ir}, W_{iz}, W_{in} \}``.
  - `weight_h`: Concatenated Weights to map from hidden space
                ``\{ W_{hr}, W_{hz}, W_{hn} \}``.
  - `bias_i`: Bias vector (``b_{in}``; not present if `use_bias=false`).
  - `bias_h`: Concatenated Bias vector for the hidden space
              ``\{ b_{hr}, b_{hz}, b_{hn} \}`` (not present if `use_bias=false`).
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
              ``\{ b_{hr}, b_{hz}, b_{hn} \}``.

## States

  - `rng`: Controls the randomness (if any) in the initial state generation
"""
@concrete struct GRUCell <: AbstractRecurrentCell
    train_state <: StaticBool
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_weight
    init_state
    use_bias <: StaticBool
end

function GRUCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(),
        init_weight=glorot_uniform, init_bias=zeros32, init_state=zeros32)
    init_weight isa NTuple{3} || (init_weight = ntuple(Returns(init_weight), 3))
    init_bias isa NTuple{3} || (init_bias = ntuple(Returns(init_bias), 3))
    return GRUCell(static(train_state), in_dims, out_dims, init_bias,
        init_weight, init_state, static(use_bias))
end

function initialparameters(rng::AbstractRNG, gru::GRUCell)
    weight_i = vcat([init_weight(rng, gru.out_dims, gru.in_dims)
                     for init_weight in gru.init_weight]...)
    weight_h = vcat([init_weight(rng, gru.out_dims, gru.out_dims)
                     for init_weight in gru.init_weight]...)
    ps = (; weight_i, weight_h)
    if has_bias(gru)
        bias_i = gru.init_bias[1](rng, gru.out_dims, 1)
        # TODO: in v1 we make this a flat vector
        bias_h = vcat([init_bias(rng, gru.out_dims, 1) for init_bias in gru.init_bias]...)
        ps = merge(ps, (bias_i=bias_i, bias_h=bias_h))
    end
    has_train_state(gru) &&
        (ps = merge(ps, (hidden_state=gru.init_state(rng, gru.out_dims),)))
    return ps
end

initialstates(rng::AbstractRNG, ::GRUCell) = (rng=Utils.sample_replicate(rng),)

function (gru::GRUCell{True})(x::AbstractMatrix, ps, st::NamedTuple)
    hidden_state = Utils.init_trainable_hidden_state(ps.hidden_state, x)
    return gru((x, (hidden_state,)), ps, st)
end

function (gru::GRUCell{False})(x::AbstractMatrix, ps, st::NamedTuple)
    rng = replicate(st.rng)
    st = merge(st, (; rng))
    hidden_state = Utils.init_hidden_state(rng, gru, x)
    return gru((x, (hidden_state,)), ps, st)
end

const _GRUCellInputType = Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}}

function (gru::GRUCell)((x, (hidden_state,))::_GRUCellInputType, ps, st::NamedTuple)
    y, hidden_stateₙ = match_eltype(gru, ps, st, x, hidden_state)
    gxs = multigate(ps.weight_i * y, Val(3))
    bias_h = safe_vec(safe_getproperty(ps, Val(:bias_h)))
    ghbs = multigate(
        fused_dense_bias_activation(identity, ps.weight_h, hidden_stateₙ, bias_h), Val(3))

    r = @. sigmoid_fast(gxs[1] + ghbs[1])
    z = @. sigmoid_fast(gxs[2] + ghbs[2])
    n = gru_cell_compute(gxs[3], r, ghbs[3], safe_getproperty(ps, Val(:bias_i)))
    hidden_state₂ = @. (1 - z) * n + z * hidden_stateₙ

    return (hidden_state₂, (hidden_state₂,)), st
end

gru_cell_compute(x, r, y, ::Nothing) = @. tanh_fast(x + r * y)
gru_cell_compute(x, r, y, bias) = @. tanh_fast(x + r * y + bias)

function Base.show(io::IO, g::GRUCell)
    print(io, "GRUCell($(g.in_dims) => $(g.out_dims)")
    has_bias(g) || print(io, ", use_bias=false")
    has_train_state(g) && print(io, ", train_state=true")
    print(io, ")")
end

"""
    BidirectionalRNN(cell::AbstractRecurrentCell,
        backward_cell::Union{AbstractRecurrentCell, Nothing}=nothing;
        merge_mode::Union{Function, Nothing}=vcat,
        ordering::AbstractTimeSeriesDataBatchOrdering=BatchLastIndex())

Bidirectional RNN wrapper.

## Arguments

  - `cell`: A recurrent cell. See [`RNNCell`](@ref), [`LSTMCell`](@ref), [`GRUCell`](@ref),
    for how the inputs/outputs of a recurrent cell must be structured.
  - `backward_cell`: A optional backward recurrent cell. If `backward_cell` is `nothing`,
    the rnn layer instance passed as the `cell` argument will be used to generate the
    backward layer automatically. `in_dims` of `backward_cell` should be consistent with
    `in_dims` of `cell`

## Keyword Arguments

  - `merge_mode`: Function by which outputs of the forward and backward RNNs will be combined.
    default value is `vcat`. If `nothing`, the outputs will not be combined.
  - `ordering`: The ordering of the batch and time dimensions in the input. Defaults to
    `BatchLastIndex()`. Alternatively can be set to `TimeLastIndex()`.

# Extended Help

## Inputs

  - If `x` is a

      + Tuple or Vector: Each element is fed to the `cell` sequentially.

      + Array (except a Vector): It is spliced along the penultimate dimension and each
        slice is fed to the `cell` sequentially.

## Returns

  - Merged output of the `cell` and `backward_cell` for the entire sequence.
  - Update state of the `cell` and `backward_cell`.

## Parameters

  - `NamedTuple` with `cell` and `backward_cell`.

## States

  - Same as `cell` and `backward_cell`.
"""
@concrete struct BidirectionalRNN <: AbstractExplicitContainerLayer{(:model,)}
    model <: Parallel
end

function PrettyPrinting.printable_children(l::BidirectionalRNN)
    merge_mode = l.model.connection isa Broadcast.BroadcastFunction ? l.model.connection.f :
                 nothing
    return (; merge_mode, forward_cell=l.model.layers.forward_rnn.cell,
        backward_cell=l.model.layers.backward_rnn.rnn.cell)
end

(rnn::BidirectionalRNN)(x, ps, st::NamedTuple) = rnn.model(x, ps, st)

function BidirectionalRNN(cell::AbstractRecurrentCell,
        backward_cell::Union{AbstractRecurrentCell, Nothing}=nothing;
        merge_mode::Union{Function, Nothing}=vcat,
        ordering::AbstractTimeSeriesDataBatchOrdering=BatchLastIndex())
    layer = Recurrence(cell; return_sequence=true, ordering)
    backward_rnn_layer = backward_cell === nothing ? layer :
                         Recurrence(backward_cell; return_sequence=true, ordering)
    fuse_op = merge_mode === nothing ? nothing : Broadcast.BroadcastFunction(merge_mode)
    return BidirectionalRNN(Parallel(fuse_op;
        forward_rnn=layer,
        backward_rnn=Chain(;
            rev1=ReverseSequence(), rnn=backward_rnn_layer, rev2=ReverseSequence())))
end
