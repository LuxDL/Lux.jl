abstract type AbstractRecurrentCell{use_bias, train_state} <: AbstractExplicitLayer end

# Fallback for vector inputs
function (rnn::AbstractRecurrentCell)(x::AbstractVector, ps, st::NamedTuple)
    (y, carry), st_ = rnn(reshape(x, :, 1), ps, st)
    return (vec(y), vec.(carry)), st_
end

function (rnn::AbstractRecurrentCell)((x, carry), ps, st::NamedTuple)
    x_ = reshape(x, :, 1)
    carry_ = map(Base.Fix2(reshape, (:, 1)), carry)
    (y, carry_new), st_ = rnn((x_, carry_), ps, st)
    return (vec(y), vec.(carry_new)), st_
end

abstract type AbstractTimeSeriesDataBatchOrdering end

struct TimeLastIndex <: AbstractTimeSeriesDataBatchOrdering end
struct BatchLastIndex <: AbstractTimeSeriesDataBatchOrdering end

"""
    Recurrence(cell;
        ordering::AbstractTimeSeriesDataBatchOrdering=BatchLastIndex(),
        return_sequence::Bool=false)

Wraps a recurrent cell (like [`RNNCell`](@ref), [`LSTMCell`](@ref), [`GRUCell`](@ref)) to
automatically operate over a sequence of inputs.

!!! warning

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

## Inputs

  - If `x` is a

      + Tuple or Vector: Each element is fed to the `cell` sequentially.

      + Array (except a Vector): It is spliced along the penultimate dimension and each
        slice is fed to the `cell` sequentially.

## Returns

  - Output of the `cell` for the entire sequence.
  - Update state of the `cell`.

## Parameters

  - Same as `cell`.

## States

  - Same as `cell`.

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
struct Recurrence{
    R, C <: AbstractRecurrentCell, O <: AbstractTimeSeriesDataBatchOrdering} <:
       AbstractExplicitContainerLayer{(:cell,)}
    cell::C
    ordering::O
end

function Recurrence(cell; ordering::AbstractTimeSeriesDataBatchOrdering=BatchLastIndex(),
        return_sequence::Bool=false)
    return Recurrence{return_sequence, typeof(cell), typeof(ordering)}(cell, ordering)
end

_eachslice(x::AbstractArray, ::TimeLastIndex) = _eachslice(x, Val(ndims(x)))
_eachslice(x::AbstractArray, ::BatchLastIndex) = _eachslice(x, Val(ndims(x) - 1))
function _eachslice(x::AbstractMatrix, ::BatchLastIndex)
    error("`BatchLastIndex` not supported for AbstractMatrix. You probably want to use \
           `TimeLastIndex`.")
    return
end

@inline function (r::Recurrence)(x::AbstractArray, ps, st::NamedTuple)
    return Lux.apply(r, _eachslice(x, r.ordering), ps, st)
end

function (r::Recurrence{false})(x::Union{AbstractVector, NTuple}, ps, st::NamedTuple)
    (out, carry), st = Lux.apply(r.cell, first(x), ps, st)
    for x_ in x[(begin + 1):end]
        (out, carry), st = Lux.apply(r.cell, (x_, carry), ps, st)
    end
    return out, st
end

@views function (r::Recurrence{true})(x::Union{AbstractVector, NTuple}, ps, st::NamedTuple)
    function __recurrence_op(::Nothing, input)
        (out, carry), state = Lux.apply(r.cell, input, ps, st)
        return [out], carry, state
    end
    function __recurrence_op((outputs, carry, state), input)
        (out, carry), state = Lux.apply(r.cell, (input, carry), ps, state)
        return vcat(outputs, [out]), carry, state
    end
    results = foldl_init(__recurrence_op, x)
    return first(results), last(results)
end

"""
    StatefulRecurrentCell(cell)

Wraps a recurrent cell (like [`RNNCell`](@ref), [`LSTMCell`](@ref), [`GRUCell`](@ref)) and
makes it stateful.

!!! tip

    This is very similar to `Flux.Recur`

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

## Parameters

  - Same as `cell`.

## States

  - NamedTuple containing:

      + `cell`: Same as `cell`.
      + `carry`: The carry state of the `cell`.
"""
struct StatefulRecurrentCell{C <: AbstractRecurrentCell} <:
       AbstractExplicitContainerLayer{(:cell,)}
    cell::C
end

function initialstates(rng::AbstractRNG, r::StatefulRecurrentCell)
    return (cell=initialstates(rng, r.cell), carry=nothing)
end

function (r::StatefulRecurrentCell)(x, ps, st::NamedTuple)
    (out, carry), st_ = applyrecurrentcell(r.cell, x, ps, st.cell, st.carry)
    return out, (; cell=st_, carry)
end

function applyrecurrentcell(l::AbstractRecurrentCell, x, ps, st, carry)
    return Lux.apply(l, (x, carry), ps, st)
end

function applyrecurrentcell(l::AbstractRecurrentCell, x, ps, st, ::Nothing)
    return Lux.apply(l, x, ps, st)
end

@doc doc"""
    RNNCell(in_dims => out_dims, activation=tanh; bias::Bool=true,
            train_state::Bool=false, init_bias=zeros32, init_weight=glorot_uniform,
            init_state=ones32)

An Elman RNNCell cell with `activation` (typically set to `tanh` or `relu`).

``h_{new} = activation(weight_{ih} \times x + weight_{hh} \times h_{prev} + bias)``

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State) Dimension
  - `activation`: Activation function
  - `bias`: Set to false to deactivate bias
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
  - Case 2: Tuple `(x, (h, ))` is provided, then the output and a tuple containing the updated hidden state is returned.

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
@concrete struct RNNCell{use_bias, train_state} <:
                 AbstractRecurrentCell{use_bias, train_state}
    activation
    in_dims::Int
    out_dims::Int
    init_bias
    init_weight
    init_state
end

function RNNCell((in_dims, out_dims)::Pair{<:Int, <:Int}, activation=tanh;
        use_bias::Bool=true, train_state::Bool=false, init_bias=zeros32,
        init_weight=glorot_uniform, init_state=ones32)
    return RNNCell{use_bias, train_state}(
        activation, in_dims, out_dims, init_bias, init_weight, init_state)
end

function initialparameters(
        rng::AbstractRNG, rnn::RNNCell{use_bias, TS}) where {use_bias, TS}
    ps = (weight_ih=rnn.init_weight(rng, rnn.out_dims, rnn.in_dims),
        weight_hh=rnn.init_weight(rng, rnn.out_dims, rnn.out_dims))
    use_bias && (ps = merge(ps, (bias=rnn.init_bias(rng, rnn.out_dims),)))
    TS && (ps = merge(ps, (hidden_state=rnn.init_state(rng, rnn.out_dims),)))
    return ps
end

function initialstates(rng::AbstractRNG, ::RNNCell)
    # FIXME(@avik-pal): Take PRNGs seriously
    randn(rng, 1)
    return (rng=replicate(rng),)
end

function (rnn::RNNCell{use_bias, false})(
        x::AbstractMatrix, ps, st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_hidden_state(rng, rnn, x)
    return rnn((x, (hidden_state,)), ps, st)
end

function (rnn::RNNCell{use_bias, true})(
        x::AbstractMatrix, ps, st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_trainable_hidden_state(ps.hidden_state, x)
    return rnn((x, (hidden_state,)), ps, st)
end

const _RNNCellInputType = Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}}

function (rnn::RNNCell{true})((x, (hidden_state,))::_RNNCellInputType, ps, st::NamedTuple)
    h_new = ps.weight_ih * x .+ ps.weight_hh * hidden_state .+ ps.bias
    h_new = __apply_activation(rnn.activation, h_new)
    return (h_new, (h_new,)), st
end

function (rnn::RNNCell{false})((x, (hidden_state,))::_RNNCellInputType, ps, st::NamedTuple)
    h_new = ps.weight_ih * x .+ ps.weight_hh * hidden_state
    h_new = __apply_activation(rnn.activation, h_new)
    return (h_new, (h_new,)), st
end

function Base.show(io::IO, r::RNNCell{use_bias, TS}) where {use_bias, TS}
    print(io, "RNNCell($(r.in_dims) => $(r.out_dims)")
    (r.activation == identity) || print(io, ", $(r.activation)")
    use_bias || print(io, ", bias=false")
    TS && print(io, ", train_state=true")
    return print(io, ")")
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
@concrete struct LSTMCell{use_bias, train_state, train_memory} <:
                 AbstractRecurrentCell{use_bias, train_state}
    in_dims::Int
    out_dims::Int
    init_bias
    init_weight
    init_state
    init_memory
end

function LSTMCell((in_dims, out_dims)::Pair{<:Int, <:Int};
        use_bias::Bool=true,
        train_state::Bool=false,
        train_memory::Bool=false,
        init_weight::NTuple{4, Function}=(
            glorot_uniform, glorot_uniform, glorot_uniform, glorot_uniform),
        init_bias::NTuple{4, Function}=(zeros32, zeros32, ones32, zeros32),
        init_state::Function=zeros32,
        init_memory::Function=zeros32)
    return LSTMCell{use_bias, train_state, train_memory}(
        in_dims, out_dims, init_bias, init_weight, init_state, init_memory)
end

function initialparameters(rng::AbstractRNG,
        lstm::LSTMCell{use_bias, TS, train_memory}) where {use_bias, TS, train_memory}
    weight_i = vcat([init_weight(rng, lstm.out_dims, lstm.in_dims)
                     for init_weight in lstm.init_weight]...)
    weight_h = vcat([init_weight(rng, lstm.out_dims, lstm.out_dims)
                     for init_weight in lstm.init_weight]...)
    ps = (weight_i=weight_i, weight_h=weight_h)
    if use_bias
        bias = vcat([init_bias(rng, lstm.out_dims, 1) for init_bias in lstm.init_bias]...)
        ps = merge(ps, (bias=bias,))
    end
    TS && (ps = merge(ps, (hidden_state=lstm.init_state(rng, lstm.out_dims),)))
    train_memory && (ps = merge(ps, (memory=lstm.init_memory(rng, lstm.out_dims),)))
    return ps
end

function initialstates(rng::AbstractRNG, ::LSTMCell)
    # FIXME(@avik-pal): Take PRNGs seriously
    randn(rng, 1)
    return (rng=replicate(rng),)
end

function (lstm::LSTMCell{use_bias, false, false})(
        x::AbstractMatrix, ps, st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_hidden_state(rng, lstm, x)
    memory = _init_hidden_state(rng, lstm, x)
    return lstm((x, (hidden_state, memory)), ps, st)
end

function (lstm::LSTMCell{use_bias, true, false})(
        x::AbstractMatrix, ps, st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_trainable_hidden_state(ps.hidden_state, x)
    memory = _init_hidden_state(rng, lstm, x)
    return lstm((x, (hidden_state, memory)), ps, st)
end

function (lstm::LSTMCell{use_bias, false, true})(
        x::AbstractMatrix, ps, st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_hidden_state(rng, lstm, x)
    memory = _init_trainable_hidden_state(ps.memory, x)
    return lstm((x, (hidden_state, memory)), ps, st)
end

function (lstm::LSTMCell{use_bias, true, true})(
        x::AbstractMatrix, ps, st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_trainable_hidden_state(ps.hidden_state, x)
    memory = _init_trainable_hidden_state(ps.memory, x)
    return lstm((x, (hidden_state, memory)), ps, st)
end

const _LSTMCellInputType = Tuple{
    <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}}

function (lstm::LSTMCell{true})(
        (x, (hidden_state, memory))::_LSTMCellInputType, ps, st::NamedTuple)
    g = ps.weight_i * x .+ ps.weight_h * hidden_state .+ ps.bias
    input, forget, cell, output = multigate(g, Val(4))
    memory_new = @. sigmoid_fast(forget) * memory + sigmoid_fast(input) * tanh_fast(cell)
    hidden_state_new = @. sigmoid_fast(output) * tanh_fast(memory_new)
    return (hidden_state_new, (hidden_state_new, memory_new)), st
end

function (lstm::LSTMCell{false})(
        (x, (hidden_state, memory))::_LSTMCellInputType, ps, st::NamedTuple)
    g = ps.weight_i * x .+ ps.weight_h * hidden_state
    input, forget, cell, output = multigate(g, Val(4))
    memory_new = @. sigmoid_fast(forget) * memory + sigmoid_fast(input) * tanh_fast(cell)
    hidden_state_new = @. sigmoid_fast(output) * tanh_fast(memory_new)
    return (hidden_state_new, (hidden_state_new, memory_new)), st
end

function Base.show(io::IO,
        lstm::LSTMCell{use_bias, TS, train_memory}) where {use_bias, TS, train_memory}
    print(io, "LSTMCell($(lstm.in_dims) => $(lstm.out_dims)")
    use_bias || print(io, ", bias=false")
    TS && print(io, ", train_state=true")
    train_memory && print(io, ", train_memory=true")
    return print(io, ")")
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
                ``\\left\\\{ W_{ir}, W_{iz}, W_{in} \\right\\\}``.
  - `weight_h`: Concatenated Weights to map from hidden space
                ``\\left\\\{ W_{hr}, W_{hz}, W_{hn} \\right\\\}``.
  - `bias_i`: Bias vector (``b_{in}``; not present if `use_bias=false`).
  - `bias_h`: Concatenated Bias vector for the hidden space
              ``\\left\\\{ b_{hr}, b_{hz}, b_{hn} \\right\\\}`` (not present if
              `use_bias=false`).
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
              ``\\left\\\{ b_{hr}, b_{hz}, b_{hn} \\right\\\}``.

## States

  - `rng`: Controls the randomness (if any) in the initial state generation
"""
@concrete struct GRUCell{use_bias, train_state} <:
                 AbstractRecurrentCell{use_bias, train_state}
    in_dims::Int
    out_dims::Int
    init_bias
    init_weight
    init_state
end

function GRUCell((in_dims, out_dims)::Pair{<:Int, <:Int};
        use_bias::Bool=true, train_state::Bool=false,
        init_weight::NTuple{3, Function}=(glorot_uniform, glorot_uniform, glorot_uniform),
        init_bias::NTuple{3, Function}=(zeros32, zeros32, zeros32),
        init_state::Function=zeros32)
    return GRUCell{use_bias, train_state}(
        in_dims, out_dims, init_bias, init_weight, init_state)
end

function initialparameters(
        rng::AbstractRNG, gru::GRUCell{use_bias, TS}) where {use_bias, TS}
    weight_i = vcat([init_weight(rng, gru.out_dims, gru.in_dims)
                     for init_weight in gru.init_weight]...)
    weight_h = vcat([init_weight(rng, gru.out_dims, gru.out_dims)
                     for init_weight in gru.init_weight]...)
    ps = (; weight_i, weight_h)
    if use_bias
        bias_i = gru.init_bias[1](rng, gru.out_dims, 1)
        bias_h = vcat([init_bias(rng, gru.out_dims, 1) for init_bias in gru.init_bias]...)
        ps = merge(ps, (bias_i=bias_i, bias_h=bias_h))
    end
    TS && (ps = merge(ps, (hidden_state=gru.init_state(rng, gru.out_dims),)))
    return ps
end

function initialstates(rng::AbstractRNG, ::GRUCell)
    # FIXME(@avik-pal): Take PRNGs seriously
    randn(rng, 1)
    return (rng=replicate(rng),)
end

function (gru::GRUCell{use_bias, true})(
        x::AbstractMatrix, ps, st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_trainable_hidden_state(ps.hidden_state, x)
    return gru((x, (hidden_state,)), ps, st)
end

function (gru::GRUCell{use_bias, false})(
        x::AbstractMatrix, ps, st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_hidden_state(rng, gru, x)
    return gru((x, (hidden_state,)), ps, st)
end

const _GRUCellInputType = Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}}

function (gru::GRUCell{true})((x, (hidden_state,))::_GRUCellInputType, ps, st::NamedTuple)
    gxs = multigate(ps.weight_i * x, Val(3))
    ghbs = multigate(ps.weight_h * hidden_state .+ ps.bias_h, Val(3))

    r = @. sigmoid_fast(gxs[1] + ghbs[1])
    z = @. sigmoid_fast(gxs[2] + ghbs[2])
    n = @. tanh_fast(gxs[3] + ps.bias_i + r * ghbs[3])
    hidden_state_new = @. (1 - z) * n + z * hidden_state

    return (hidden_state_new, (hidden_state_new,)), st
end

function (gru::GRUCell{false})((x, (hidden_state,))::_GRUCellInputType, ps, st::NamedTuple)
    gxs = multigate(ps.weight_i * x, Val(3))
    ghs = multigate(ps.weight_h * hidden_state, Val(3))

    r = @. sigmoid_fast(gxs[1] + ghs[1])
    z = @. sigmoid_fast(gxs[2] + ghs[2])
    n = @. tanh_fast(gxs[3] + r * ghs[3])
    hidden_state_new = @. (1 - z) * n + z * hidden_state

    return (hidden_state_new, (hidden_state_new,)), st
end

function Base.show(io::IO, g::GRUCell{use_bias, TS}) where {use_bias, TS}
    print(io, "GRUCell($(g.in_dims) => $(g.out_dims)")
    use_bias || print(io, ", bias=false")
    TS && print(io, ", train_state=true")
    return print(io, ")")
end
