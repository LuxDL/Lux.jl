function Impl.repeat_inner_gqa(x::AnyTracedRArray{T,N}, nrepeat::Int, K::Int) where {T,N}
    x = materialize_traced_array(x)
    sz = size(x)
    x = @opcall broadcast_in_dim(
        x, [1:(K - 1)..., (K + 1):(N + 1)...], [sz[1:(K - 1)]..., nrepeat, sz[K:end]...]
    )
    x = @opcall reshape(x, [sz[1:(K - 1)]..., nrepeat * sz[K], sz[(K + 1):end]...])
    return x
end

function Impl.unsafe_scaled_dot_product_attention(
    q::AnyTracedRArray{TQ,N},
    k::AnyTracedRArray{TK,N},
    v::AnyTracedRArray{TV,N};
    head_dim::Int,
    token_dim::Int,
    num_heads_dim::Int,
    scale,
    mask,
    bias::Union{Nothing,AbstractMatrix},
    fdrop::F,
) where {TQ,TK,TV,F,N}
    q = materialize_traced_array(q)
    k = materialize_traced_array(k)
    v = materialize_traced_array(v)
    mask !== nothing && (mask = materialize_traced_array(mask))
    bias !== nothing && (bias = materialize_traced_array(bias))

    # alpha: [B..., H, S, L]
    scale_q = @opcall fill(scale, size(q))
    q_scaled = @opcall multiply(q, scale_q)
    alpha = @opcall dot_general(
        q_scaled,
        k;
        contracting_dimensions=(Int[head_dim], Int[head_dim]),
        batching_dimensions=(
            collect(Int, (N:-1:4..., num_heads_dim)),
            collect(Int, (N:-1:4..., num_heads_dim)),
        ),
    )

    if bias !== nothing
        bias = @opcall broadcast_in_dim(bias, [N, N - 1], collect(Int, size(alpha)))
        alpha = @opcall add(alpha, bias)
    end

    if mask !== nothing
        mask = @opcall broadcast_in_dim(mask, [N, N - 1], collect(Int, size(alpha)))
        neginf = @opcall fill(typemin(eltype(alpha)), size(alpha))
        alpha = @opcall select(mask, alpha, neginf)
    end

    # softmax
    max_ = @opcall reduce(alpha, typemin(eltype(alpha)), Int64[N], max)
    max_ = @opcall broadcast_in_dim(
        max_, collect(Int, 1:(N - 1)), collect(Int, size(alpha))
    )
    diff = @opcall(exponential(@opcall(subtract(alpha, max_))))
    sum_ = @opcall reduce(diff, eltype(diff)(0), Int64[N], +)
    sum_ = @opcall broadcast_in_dim(
        sum_, collect(Int, 1:(N - 1)), collect(Int, size(alpha))
    )

    scores = @opcall divide(diff, sum_)
    attn_scores = fdrop(scores)

    # x: [B..., H, Ev, L]
    x = @opcall dot_general(
        attn_scores,
        v;
        contracting_dimensions=(Int[N], Int[token_dim]),
        batching_dimensions=(
            collect(Int, 1:(N - 2)), collect(Int, (N:-1:4..., num_heads_dim))
        ),
    )

    attn_scores_to_return = @opcall transpose(attn_scores, collect(Int, N:-1:1))
    x_to_return = @opcall transpose(x, [N, N - 2, N - 1, (N - 3):-1:1...])

    return x_to_return, attn_scores_to_return
end
