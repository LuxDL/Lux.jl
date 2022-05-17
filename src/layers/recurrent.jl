"""
    RNNCell(in_dims=>out_dims, activation=tanh; bias::Bool=true, init_bias=zeros32, init_weight=glorot_uniform, init_state=ones32)

An Elman RNNCell cell with `activation` (typically set to `tanh` or `relu`).

``h_{new} = activation.(weight_{ih} \\times x + weight_{hh} \\times h_{prev} + bias)``

## Inputs

* Case 1: Only a single input `x` of shape `(in_dims, batch_size)` - Creates a hidden state using `init_state`.
* Case 2: Tuple (`x`, `h`) is provided, then the updated hidden state is returned.

## Output

* New hidden state ``h_{new}`` of shape `(out_dims, batch_size)`
* Updated model state

## Parameters

* `weight_ih`: Maps the input to the hidden state.
* `weight_hh`: Maps the hidden state to the hidden state.
* `bias`: Bias vector (not present if `bias=false`)

## States

* `rng`: Controls the randomness (if any) in the initial state generation
"""
struct RNNCell{bias,A,B,W,S} <: AbstractExplicitLayer
    activation::A
    in_dims::Int
    out_dims::Int
    init_bias::B
    init_weight::W
    init_state::S
end

function RNNCell(
    (in_dims, out_dims)::Pair{<:Int,<:Int},
    activation=tanh;
    bias::Bool=true,
    init_bias=zeros32,
    init_weight=glorot_uniform,
    init_state=ones32,
)
    return RNNCell{bias,typeof(activation),typeof(init_bias),typeof(init_weight),typeof(init_state)}(
        activation, in_dims, out_dims, init_bias, init_weight, init_state
    )
end

function initialparameters(rng::AbstractRNG, rnn::RNNCell{bias}) where {bias}
    ps = (
        weight_ih=rnn.init_weight(rng, rnn.out_dims, rnn.in_dims),
        weight_hh=rnn.init_weight(rng, rnn.out_dims, rnn.out_dims),
    )
    if bias
        ps = merge(ps, (bias=rnn.init_bias(rng, rnn.out_dims),))
    end
    return ps
end

function initialstates(rng::AbstractRNG, ::RNNCell)
    # FIXME: Take PRNGs seriously
    randn(rng, 1)
    return (rng=replicate(rng),)
end

function (rnn::RNNCell)(x::AbstractMatrix, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple)
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = rnn.init_state(rng, rnn.out_dims, size(x, 2))
    return rnn((x, hidden_state), ps, st)
end

function (rnn::RNNCell{true})(
    (x, hidden_state)::NTuple{2,<:AbstractMatrix}, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
)
    h_new = rnn.activation.(ps.weight_ih * x .+ ps.weight_hh * hidden_state .+ ps.bias)
    return h_new, st
end

function (rnn::RNNCell{true,typeof(identity)})(
    (x, hidden_state)::NTuple{2,<:AbstractMatrix}, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
)
    h_new = ps.weight_ih * x .+ ps.weight_hh * hidden_state .+ ps.bias
    return h_new, st
end

function (rnn::RNNCell{false})(
    (x, hidden_state)::NTuple{2,<:AbstractMatrix}, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
)
    h_new = rnn.activation.(ps.weight_ih * x .+ ps.weight_hh * hidden_state)
    return h_new, st
end

function (rnn::RNNCell{false,typeof(identity)})(
    (x, hidden_state)::NTuple{2,<:AbstractMatrix}, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
)
    h_new = ps.weight_ih * x .+ ps.weight_hh * hidden_state
    return h_new, st
end
