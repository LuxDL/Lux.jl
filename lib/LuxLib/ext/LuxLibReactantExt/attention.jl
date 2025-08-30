function Impl.unsafe_scaled_dot_product_attention(
    q::AnyTracedRArray{TQ,N},
    k::AnyTracedRArray{TK,N},
    v::AnyTracedRArray{TV,N};
    head_dim::Int,
    token_dim::Int,
    num_heads_dim::Int,
    scale,
    mask,
    bias,
    fdrop::F,
) where {TQ,TK,TV,F,N}
    q = materialize_traced_array(q)
    k = materialize_traced_array(k)
    v = materialize_traced_array(v)
    mask !== nothing && (mask = materialize_traced_array(mask))
    bias !== nothing && (bias = materialize_traced_array(bias))

    # alpha: [B..., H, S, L]
    scale_q = @opcall fill(scale, size(q))
    q_scaled = @opcall divide(q, scale_q)
    alpha = @opcall dot_general(
        q_scaled,
        k;
        contracting_dimensions=(Int[head_dim], Int[head_dim]),
        batching_dimensions=(
            collect(Int, (ntuple(Base.Fix2(+, 3), Val(N - 3))..., num_heads_dim)),
            collect(Int, (ntuple(Base.Fix2(+, 3), Val(N - 3))..., num_heads_dim)),
        ),
    )

    # TODO: add bias support
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
        v,
        attn_scores;
        contracting_dimensions=(Int[token_dim], Int[N]),
        batching_dimensions=(
            collect(Int, (ntuple(Base.Fix2(+, 3), Val(N - 3))..., num_heads_dim)),
            collect(Int, 1:(N - 2)),
        ),
    )

    attn_scores_to_return = @opcall transpose(attn_scores, [N, N - 2, N - 1, 1:(N - 3)...])
    x_to_return = @opcall transpose(x, [N - 1, N - 2, N, 1:(N - 3)...])

    return x_to_return, attn_scores_to_return
end
