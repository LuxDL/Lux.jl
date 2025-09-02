function repeat_inner_gqa(x::AbstractArray{T,N}, nrepeat::Int, K::Int) where {T,N}
    return repeat(x; inner=ntuple(i -> ifelse(i == K, nrepeat, 1), Val(N)))
end

function scaled_dot_product_attention(
    q::AbstractArray{TQ,N},
    k::AbstractArray{TK,N},
    v::AbstractArray{TV,N};
    head_dim::Int=1,
    token_dim::Int=3,
    scale=nothing,
    mask=nothing,
    is_causal::Union{Bool,Nothing}=nothing,
    bias=nothing,
    fdrop::F=identity,
) where {TQ,TK,TV,F,N}
    @assert N ≥ 4 "N must be at least 4"
    @assert 1 ≤ head_dim ≤ 3 "head_dim must be between 1 and 3"
    @assert 1 ≤ token_dim ≤ 3 "token_dim must be between 1 and 3"
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

    num_heads_dim = get_non_heads_dim(head_dim, token_dim)
    num_heads = size(q, num_heads_dim)
    num_kv_groups = size(k, num_heads_dim)

    @assert size(v, token_dim) == size(k, token_dim) "`v` and `k` must have the same \
                                                      number of tokens. Found $(size(v)) \
                                                      vs $(size(k))."
    @assert num_kv_groups == size(v, num_heads_dim)
    @assert num_heads % num_kv_groups == 0 "num_heads must be divisible by num_kv_groups"

    if num_kv_groups ≠ num_heads
        group_size = num_heads ÷ num_kv_groups
        k = repeat_inner_gqa(k, group_size, num_heads_dim)
        v = repeat_inner_gqa(v, group_size, num_heads_dim)
    end

    scale = scale === nothing ? sqrt(TQ(size(q, head_dim))) : TQ(scale)
    scale = inv(scale)

    if is_causal isa Bool
        @assert mask === nothing "`mask` and `is_causal` cannot both be specified"
        if is_causal
            # generally recommended to use `is_causal=true` for causal attention rather
            # than specifying `mask` manually
            mask = make_causal_mask(q, size(k, token_dim), size(q, token_dim))
        end
    end

    return unsafe_scaled_dot_product_attention(
        q, k, v; head_dim, token_dim, num_heads_dim, scale, mask, bias, fdrop
    )
end

get_non_heads_dim(head_dim::Int, token_dim::Int) = only(setdiff(1:3, (head_dim, token_dim)))

CRC.@non_differentiable get_non_heads_dim(::Any...)

function unsafe_scaled_dot_product_attention(
    q::AbstractArray{TQ,N},
    k::AbstractArray{TK,N},
    v::AbstractArray{TV,N};
    head_dim::Int,
    token_dim::Int,
    num_heads_dim::Int,
    scale::TQ,
    mask,
    bias::Union{Nothing,AbstractMatrix},
    fdrop::F,
) where {TQ,TK,TV,F,N}
    batching_dims = (num_heads_dim, 4:N...)

    # logits: [S, L, H, B...]
    attn_scores = fdrop(
        NNlib.softmax(
            apply_attention_mask_bias(
                batched_matmul(
                    k,
                    q .* scale;
                    lhs_contracting_dim=head_dim,
                    rhs_contracting_dim=head_dim,
                    lhs_batching_dims=batching_dims,
                    rhs_batching_dims=batching_dims,
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
            lhs_batching_dims=batching_dims,
            rhs_batching_dims=ntuple(Base.Fix2(+, 2), Val(N - 2)),
        ),
        (1, 3, 2, 4:N...),
    )

    return x, attn_scores
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

make_causal_mask(x; dims::Int) = make_causal_mask(x, size(x, dims), size(x, dims))

function make_causal_mask(x, x_len, y_len)
    mask = NNlib.trues_like(x, (x_len, y_len))
    triu!(mask)
    return mask
end

CRC.@non_differentiable make_causal_mask(::Any...)
