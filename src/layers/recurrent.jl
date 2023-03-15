# TODO(@avik-pal): We can add another subtype `AbstractRecurrentCell` in the type hierarchy
#                  to make it safer to compose these cells with `Recurrence`
"""
    Recurrence(cell; return_sequence::Bool = false)

Wraps a recurrent cell (like [`RNNCell`](@ref), [`LSTMCell`](@ref), [`GRUCell`](@ref)) to
automatically operate over a sequence of inputs.

!!! warning

    This is completely distinct from `Flux.Recur`. It doesn't make the `cell` stateful,
    rather allows operating on an entire sequence of inputs at once. See
    [`StatefulRecurrentCell`](@ref) for functionality similar to [`Flux.Recur`](@ref).

## Arguments

  - `cell`: A recurrent cell. See [`RNNCell`](@ref), [`LSTMCell`](@ref), [`GRUCell`](@ref),
    for how the inputs/outputs of a recurrent cell must be structured.

## Keyword Arguments

  - `return_sequence`: If `true` returns the entire sequence of outputs, else returns only
    the last output. Defaults to `false`.

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
"""
struct Recurrence{R, C <: AbstractExplicitLayer} <: AbstractExplicitContainerLayer{(:cell,)}
    cell::C
end

function Recurrence(cell; return_sequence::Bool=false)
    return Recurrence{return_sequence, typeof(cell)}(cell)
end

@inline function (r::Recurrence)(x::A, ps, st::NamedTuple) where {A <: AbstractArray}
    return Lux.apply(r, _eachslice(x, Val(ndims(x) - 1)), ps, st)
end

function (r::Recurrence{false})(x::Union{AbstractVector, NTuple}, ps, st::NamedTuple)
    (out, carry), st = Lux.apply(r.cell, first(x), ps, st)
    for x_ in x[(begin + 1):end]
        (out, carry), st = Lux.apply(r.cell, (x_, carry), ps, st)
    end
    return out, st
end

# FIXME: Weird Hack
_generate_init_recurrence(out, carry, st) = (typeof(out)[out], carry, st)
∇_generate_init_recurrence((Δout, Δcarry, Δst)) = (first(Δout), Δcarry, Δst)

function (r::Recurrence{true})(x::Union{AbstractVector, NTuple}, ps, st::NamedTuple)
    (out_, carry), st = Lux.apply(r.cell, first(x), ps, st)

    init = _generate_init_recurrence(out_, carry, st)

    function recurrence_op(input, (outputs, carry, state))
        (out, carry), state = Lux.apply(r.cell, (input, carry), ps, state)
        return vcat(outputs, typeof(out)[out]), carry, state
    end

    results = foldr(recurrence_op, x[(begin + 1):end]; init)
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
struct StatefulRecurrentCell{C <: AbstractExplicitLayer} <:
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

function applyrecurrentcell(l::AbstractExplicitLayer, x, ps, st, carry)
    return Lux.apply(l, (x, carry), ps, st)
end
applyrecurrentcell(l::AbstractExplicitLayer, x, ps, st, ::Nothing) = Lux.apply(l, x, ps, st)

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
  - `bias`: Bias vector (not present if `bias=false`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation
"""
struct RNNCell{use_bias, train_state, A, B, W, S} <: AbstractExplicitLayer
    activation::A
    in_dims::Int
    out_dims::Int
    init_bias::B
    init_weight::W
    init_state::S
end

function RNNCell((in_dims, out_dims)::Pair{<:Int, <:Int}, activation=tanh;
                 use_bias::Bool=true, bias::Union{Missing, Bool}=missing,
                 train_state::Bool=false, init_bias=zeros32, init_weight=glorot_uniform,
                 init_state=ones32)
    # Deprecated Functionality (Remove in v0.5)
    if !ismissing(bias)
        Base.depwarn("`bias` argument to `RNNCell` has been deprecated and will be removed" *
                     " in v0.5. Use `use_bias` kwarg instead.", :RNNCell)
        if !use_bias
            throw(ArgumentError("Both `bias` and `use_bias` are set. Please only use " *
                                "the `use_bias` keyword argument."))
        end
        use_bias = bias
    end

    return RNNCell{use_bias, train_state, typeof(activation), typeof(init_bias),
                   typeof(init_weight), typeof(init_state)}(activation, in_dims, out_dims,
                                                            init_bias, init_weight,
                                                            init_state)
end

function initialparameters(rng::AbstractRNG,
                           rnn::RNNCell{use_bias, TS}) where {use_bias, TS}
    ps = (weight_ih=rnn.init_weight(rng, rnn.out_dims, rnn.in_dims),
          weight_hh=rnn.init_weight(rng, rnn.out_dims, rnn.out_dims))
    if use_bias
        ps = merge(ps, (bias=rnn.init_bias(rng, rnn.out_dims),))
    end
    if TS
        ps = merge(ps, (hidden_state=rnn.init_state(rng, rnn.out_dims),))
    end
    return ps
end

function initialstates(rng::AbstractRNG, ::RNNCell)
    # FIXME(@avik-pal): Take PRNGs seriously
    randn(rng, 1)
    return (rng=replicate(rng),)
end

function (rnn::RNNCell{use_bias, false})(x::AbstractMatrix, ps,
                                         st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_hidden_state(rng, rnn, x)
    return rnn((x, (hidden_state,)), ps, st)
end

function (rnn::RNNCell{use_bias, true})(x::AbstractMatrix, ps,
                                        st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_trainable_hidden_state(ps.hidden_state, x)
    return rnn((x, (hidden_state,)), ps, st)
end

const _RNNCellInputType = Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}}

function (rnn::RNNCell{true})((x, (hidden_state,))::_RNNCellInputType, ps, st::NamedTuple)
    h_new = rnn.activation.(ps.weight_ih * x .+ ps.weight_hh * hidden_state .+ ps.bias)
    return (h_new, (h_new,)), st
end

function (rnn::RNNCell{true, TS, typeof(identity)})((x, (hidden_state,))::_RNNCellInputType,
                                                    ps, st::NamedTuple) where {TS}
    h_new = ps.weight_ih * x .+ ps.weight_hh * hidden_state .+ ps.bias
    return (h_new, (h_new,)), st
end

function (rnn::RNNCell{false})((x, (hidden_state,))::_RNNCellInputType, ps, st::NamedTuple)
    h_new = rnn.activation.(ps.weight_ih * x .+ ps.weight_hh * hidden_state)
    return (h_new, (h_new,)), st
end

function (rnn::RNNCell{false, TS, typeof(identity)})((x, (hs,))::_RNNCellInputType, ps,
                                                     st::NamedTuple) where {TS}
    h_new = ps.weight_ih * x .+ ps.weight_hh * hs
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
                ``\left\{ W_{ii}, W_{if}, W_{ig}, W_{io} \right\}``.
  - `weight_h`: Concatenated Weights to map from hidden space
                ``\left\{ W_{hi}, W_{hf}, W_{hg}, W_{ho} \right\}``
  - `bias`: Bias vector (not present if `use_bias=false`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  - `memory`: Initial memory vector (not present if `train_memory=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation
"""
struct LSTMCell{use_bias, train_state, train_memory, B, W, S, M} <: AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    init_bias::B
    init_weight::W
    init_state::S
    init_memory::M
end

function LSTMCell((in_dims, out_dims)::Pair{<:Int, <:Int}; use_bias::Bool=true,
                  train_state::Bool=false, train_memory::Bool=false,
                  init_weight::NTuple{4, Function}=(glorot_uniform, glorot_uniform,
                                                    glorot_uniform, glorot_uniform),
                  init_bias::NTuple{4, Function}=(zeros32, zeros32, ones32, zeros32),
                  init_state::Function=zeros32, init_memory::Function=zeros32)
    tfields = (use_bias, train_state, train_memory, typeof(init_bias), typeof(init_weight),
               typeof(init_state), typeof(init_memory))
    return LSTMCell{tfields...}(in_dims, out_dims, init_bias, init_weight, init_state,
                                init_memory)
end

function initialparameters(rng::AbstractRNG,
                           lstm::LSTMCell{use_bias, TS, train_memory}) where {use_bias, TS,
                                                                              train_memory}
    weight_i = vcat([init_weight(rng, lstm.out_dims, lstm.in_dims)
                     for init_weight in lstm.init_weight]...)
    weight_h = vcat([init_weight(rng, lstm.out_dims, lstm.out_dims)
                     for init_weight in lstm.init_weight]...)
    ps = (weight_i=weight_i, weight_h=weight_h)
    if use_bias
        bias = vcat([init_bias(rng, lstm.out_dims, 1) for init_bias in lstm.init_bias]...)
        ps = merge(ps, (bias=bias,))
    end
    if TS
        ps = merge(ps, (hidden_state=lstm.init_state(rng, lstm.out_dims),))
    end
    if train_memory
        ps = merge(ps, (memory=lstm.init_memory(rng, lstm.out_dims),))
    end
    return ps
end

function initialstates(rng::AbstractRNG, ::LSTMCell)
    # FIXME(@avik-pal): Take PRNGs seriously
    randn(rng, 1)
    return (rng=replicate(rng),)
end

function (lstm::LSTMCell{use_bias, false, false})(x::AbstractMatrix, ps,
                                                  st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_hidden_state(rng, lstm, x)
    memory = _init_hidden_state(rng, lstm, x)
    return lstm((x, (hidden_state, memory)), ps, st)
end

function (lstm::LSTMCell{use_bias, true, false})(x::AbstractMatrix, ps,
                                                 st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_trainable_hidden_state(ps.hidden_state, x)
    memory = _init_hidden_state(rng, lstm, x)
    return lstm((x, (hidden_state, memory)), ps, st)
end

function (lstm::LSTMCell{use_bias, false, true})(x::AbstractMatrix, ps,
                                                 st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_hidden_state(rng, lstm, x)
    memory = _init_trainable_hidden_state(ps.memory, x)
    return lstm((x, (hidden_state, memory)), ps, st)
end

function (lstm::LSTMCell{use_bias, true, true})(x::AbstractMatrix, ps,
                                                st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_trainable_hidden_state(ps.hidden_state, x)
    memory = _init_trainable_hidden_state(ps.memory, x)
    return lstm((x, (hidden_state, memory)), ps, st)
end

const _LSTMCellInputType = Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix
                                                         }}

function (lstm::LSTMCell{true})((x, (hidden_state, memory))::_LSTMCellInputType, ps,
                                st::NamedTuple)
    g = ps.weight_i * x .+ ps.weight_h * hidden_state .+ ps.bias
    input, forget, cell, output = multigate(g, Val(4))
    memory_new = @. sigmoid_fast(forget) * memory + sigmoid_fast(input) * tanh_fast(cell)
    hidden_state_new = @. sigmoid_fast(output) * tanh_fast(memory_new)
    return (hidden_state_new, (hidden_state_new, memory_new)), st
end

function (lstm::LSTMCell{false})((x, (hidden_state, memory))::_LSTMCellInputType, ps,
                                 st::NamedTuple)
    g = ps.weight_i * x .+ ps.weight_h * hidden_state
    input, forget, cell, output = multigate(g, Val(4))
    memory_new = @. sigmoid_fast(forget) * memory + sigmoid_fast(input) * tanh_fast(cell)
    hidden_state_new = @. sigmoid_fast(output) * tanh_fast(memory_new)
    return (hidden_state_new, (hidden_state_new, memory_new)), st
end

function Base.show(io::IO,
                   lstm::LSTMCell{use_bias, TS, train_memory}) where {use_bias, TS,
                                                                      train_memory}
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
  n &= \sigma(W_{in} \times x + b_{in} + r \cdot (W_{hn} \times h_{prev} + b_{hn}))\\
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
                ``\\left\\{ W_{ir}, W_{iz}, W_{in} \\right\\}``.
  - `weight_h`: Concatenated Weights to map from hidden space
                ``\\left\\{ W_{hr}, W_{hz}, W_{hn} \\right\\}``
  - `bias_i`: Bias vector (``b_{in}``; not present if `use_bias=false`)
  - `bias_h`: Concatenated Bias vector for the hidden space
              ``\\left\\{ b_{hr}, b_{hz}, b_{hn} \\right\\}`` (not present if
              `use_bias=false`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
              ``\\left\\{ b_{hr}, b_{hz}, b_{hn} \\right\\}``

## States

  - `rng`: Controls the randomness (if any) in the initial state generation
"""
struct GRUCell{use_bias, train_state, B, W, S} <: AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    init_bias::B
    init_weight::W
    init_state::S
end

function GRUCell((in_dims, out_dims)::Pair{<:Int, <:Int}; use_bias::Bool=true,
                 train_state::Bool=false,
                 init_weight::NTuple{3, Function}=(glorot_uniform, glorot_uniform,
                                                   glorot_uniform),
                 init_bias::NTuple{3, Function}=(zeros32, zeros32, zeros32),
                 init_state::Function=zeros32)
    return GRUCell{use_bias, train_state, typeof(init_bias), typeof(init_weight),
                   typeof(init_state)}(in_dims, out_dims, init_bias, init_weight,
                                       init_state)
end

function initialparameters(rng::AbstractRNG,
                           gru::GRUCell{use_bias, TS}) where {use_bias, TS}
    weight_i = vcat([init_weight(rng, gru.out_dims, gru.in_dims)
                     for init_weight in gru.init_weight]...)
    weight_h = vcat([init_weight(rng, gru.out_dims, gru.out_dims)
                     for init_weight in gru.init_weight]...)
    ps = (weight_i=weight_i, weight_h=weight_h)
    if use_bias
        bias_i = gru.init_bias[1](rng, gru.out_dims, 1)
        bias_h = vcat([init_bias(rng, gru.out_dims, 1) for init_bias in gru.init_bias]...)
        ps = merge(ps, (bias_i=bias_i, bias_h=bias_h))
    end
    if TS
        ps = merge(ps, (hidden_state=gru.init_state(rng, gru.out_dims),))
    end
    return ps
end

function initialstates(rng::AbstractRNG, ::GRUCell)
    # FIXME(@avik-pal): Take PRNGs seriously
    randn(rng, 1)
    return (rng=replicate(rng),)
end

function (gru::GRUCell{use_bias, true})(x::AbstractMatrix, ps,
                                        st::NamedTuple) where {use_bias}
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = _init_trainable_hidden_state(ps.hidden_state, x)
    return gru((x, (hidden_state,)), ps, st)
end

function (gru::GRUCell{use_bias, false})(x::AbstractMatrix, ps,
                                         st::NamedTuple) where {use_bias}
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
