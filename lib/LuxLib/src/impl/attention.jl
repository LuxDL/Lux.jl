function repeat_inner_dim(::AbstractArray{T,N}, nrepeat::Int) where {T,N}
    return ntuple(i -> i == N ? nrepeat : 1, Val(N))
end

function scaled_dot_product_attention(
    q::AbstractArray{TQ,N},
    k::AbstractArray{TK,N},
    v::AbstractArray{TV,N};
    head_dim::Int=1,
    token_dim::Int=3,
    scale=nothing,
    mask=nothing,
    bias=nothing,
    fdrop::F=identity,
) where {TQ,TK,TV,F,N}
    @assert N ≥ 4 "N must be at least 4"
    @assert 1 ≤ head_dim ≤ 3 "head_dim must be between 1 and $N"
    @assert 1 ≤ token_dim ≤ 3 "token_dim must be between 1 and $N"
    @assert head_dim ≠ token_dim "head_dim and token_dim must be different"
    @assert size(q, head_dim) == size(k, head_dim) "`q` and `k` must have the same \
                                                    number of heads. Found $(size(q)) vs \
                                                    $(size(k))."

    for i in 4:N
        if size(q, i) != size(k, i) || size(q, i) != size(v, i)
            throw(DimensionMismatch("All dimensions (except first 3) of q, k, v must be \
                                     the same. Found sizes $(size(q)), $(size(k)), \
                                     $(size(v)) mismatch at dimension $(i)."))
        end
    end

    num_heads_dim = only(setdiff(1:3, (head_dim, token_dim)))
    num_heads = size(q, num_heads_dim)
    num_kv_groups = size(k, num_heads_dim)

    @assert size(v, token_dim) == size(k, token_dim) "`v` and `k` must have the same \
                                                      number of tokens. Found $(size(v)) \
                                                      vs $(size(k))."
    @assert num_kv_groups == size(v, num_heads_dim)
    @assert num_heads % num_kv_groups == 0 "num_heads must be divisible by num_kv_groups"

    if num_kv_groups ≠ num_heads
        group_size = num_heads ÷ num_kv_groups
        k = repeat(k; inner=repeat_inner_dim(k, group_size))
        v = repeat(v; inner=repeat_inner_dim(v, group_size))
    end

    scale = scale === nothing ? sqrt(TQ(size(q, head_dim))) : TQ(scale)

    return unsafe_scaled_dot_product_attention(
        q, k, v; head_dim, token_dim, num_heads_dim, scale, mask, bias, fdrop
    )
end

function unsafe_scaled_dot_product_attention(
    q::AbstractArray{TQ,N},
    k::AbstractArray{TK,N},
    v::AbstractArray{TV,N};
    head_dim::Int,
    token_dim::Int,
    num_heads_dim::Int,
    scale::TQ,
    mask,
    bias,
    fdrop::F,
) where {TQ,TK,TV,F,N}
    # logits: [S, L, H, B...]
    attn_scores = fdrop(
        NNlib.softmax(
            apply_attention_mask_bias(
                batched_matmul(
                    k,
                    q ./ scale;
                    lhs_contracting_dim=head_dim,
                    rhs_contracting_dim=head_dim,
                    lhs_batching_dims=(
                        ntuple(Base.Fix2(+, 3), Val(N - 3))..., num_heads_dim
                    ),
                    rhs_batching_dims=(
                        ntuple(Base.Fix2(+, 3), Val(N - 3))..., num_heads_dim
                    ),
                ),
                mask,
                bias,
            );
            dims=1,
        ),
    )

    # x: [Ev, H, L, B...]
    x = permutedims(
        batched_matmul(
            v,
            attn_scores;
            lhs_contracting_dim=token_dim,
            rhs_contracting_dim=1,
            lhs_batching_dims=(ntuple(Base.Fix2(+, 3), Val(N - 3))..., num_heads_dim),
            rhs_batching_dims=ntuple(Base.Fix2(+, 2), Val(N - 2)),
        ),
        (1, N, 2, 3:(N - 1)...),
    )

    return x, permutedims(attn_scores, (1, 2, N, 3:(N - 1)...))
end

apply_attention_mask_bias(logits, ::Nothing, ::Nothing) = logits
function apply_attention_mask_bias(logits, mask, ::Nothing)
    return ifelse.(mask, logits, typemin(eltype(logits)))
end
apply_attention_mask_bias(logits, ::Nothing, bias) = logits .+ bias
function apply_attention_mask_bias(logits, mask, bias)
    neginf = typemin(eltype(logits))
    return @. ifelse(mask, logits + bias, neginf)
end
