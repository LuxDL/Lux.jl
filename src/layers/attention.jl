@concrete struct MultiHeadAttention <: AbstractLuxContainerLayer{
        (:q_proj, :k_proj, :v_proj, :attention_dropout, :out_proj),
    }
    nheads <: IntegerType
    q_proj <: Dense
    k_proj <: Dense
    v_proj <: Dense
    attention_dropout
    out_proj <: Dense
end

function parse_mha_dims(dims::IntegerType)
    return (;
        q_in = dims, k_in = dims, v_in = dims, qk = dims, v = dims, out = dims,
    )
end

function parse_mha_dims((in_dims, (qkv_dims, out_dims)))
    if in_dims isa IntegerType
        q_in, k_in, v_in = in_dims, in_dims, in_dims
    else
        @assert in_dims isa NTuple{3, <:IntegerType}
        q_in, k_in, v_in = in_dims
    end

    if qkv_dims isa IntegerType
        qk, v = qkv_dims, qkv_dims
    else
        @assert qkv_dims isa NTuple{2, <:IntegerType}
        qk, v = qkv_dims
    end

    return (; q_in, k_in, v_in, qk, v, out = out_dims)
end

function MultiHeadAttention(
        dims; nheads::IntegerType = 1,
        dense_kwargs = (; use_bias = False()), dropout_probability = 0.0f0
    )
    dims = parse_mha_dims(dims)
    @argcheck dims.qk % nheads == 0
    @argcheck dims.v % nheads == 0

    return MultiHeadAttention(
        nheads,
        Dense(dims.q_in, dims.qk; dense_kwargs...),
        Dense(dims.k_in, dims.qk; dense_kwargs...),
        Dense(dims.v_in, dims.v; dense_kwargs...),
        Dropout(dropout_probability),
        Dense(dims.v, dims.out; dense_kwargs...),
    )
end

(mha::MultiHeadAttention)(x, ps, st::NamedTuple) = apply_multiheadattention(mha, ps, st, x)
function (mha::MultiHeadAttention)(x::Tuple, ps, st::NamedTuple)
    return apply_multiheadattention(mha, ps, st, x...)
end

function apply_multiheadattention(mha::MultiHeadAttention, ps, st, qkv)
    return apply_multiheadattention(mha, ps, st, qkv, qkv, qkv, nothing)
end

function apply_multiheadattention(mha::MultiHeadAttention, ps, st, q, kv)
    return apply_multiheadattention(mha, ps, st, q, kv, kv, nothing)
end

function apply_multiheadattention(mha::MultiHeadAttention, ps, st, q, k, v, mask = nothing)
    q, q_st = mha.q_proj(q, ps.q_proj, st.q_proj)
    k, k_st = mha.k_proj(k, ps.k_proj, st.k_proj)
    v, v_st = mha.v_proj(v, ps.v_proj, st.v_proj)

    dropout = StatefulLuxLayer{true}(
        mha.attention_dropout,
        ps.attention_dropout, st.attention_dropout
    )
    x, α = NNlib.dot_product_attention(q, k, v; mha.nheads, fdrop = dropout, mask)

    x, out_st = mha.out_proj(x, ps.out_proj, st.out_proj)

    return (
        x,
        (;
            q_proj = q_st, k_proj = k_st, v_proj = v_st,
            out_proj = out_st, attention_dropout = dropout.st,
        ),
    )
end
