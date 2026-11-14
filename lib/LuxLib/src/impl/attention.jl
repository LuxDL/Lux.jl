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
    return_attention_weights::Bool=true,
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

    if is_causal isa Bool
        @assert mask === nothing "`mask` and `is_causal` cannot both be specified"
    end
    is_causal = is_causal === true

    # NNlib uses (features, heads, tokens, batch...) and a multiplicative scale.
    perm = (head_dim, num_heads_dim, token_dim, 4:N...)
    if perm != ntuple(identity, Val(N))
        q, k, v = permutedims(q, perm), permutedims(k, perm), permutedims(v, perm)
    end
    scale = scale === nothing ? nothing : inv(TQ(scale))

    if return_attention_weights
        return NNlib.scaled_dot_product_attention_with_weights(
            q, k, v, bias; scale, mask, is_causal, fdrop
        )
    end
    output = NNlib.scaled_dot_product_attention(
        q, k, v, bias; scale, mask, is_causal, fdrop
    )
    return output, nothing
end

get_non_heads_dim(head_dim::Int, token_dim::Int) = only(setdiff(1:3, (head_dim, token_dim)))

CRC.@non_differentiable get_non_heads_dim(::Any...)


function make_causal_mask(x, x_len, y_len)
    mask = NNlib.trues_like(x, (x_len, y_len))
    triu!(mask, min(0, y_len - x_len))
    return mask
end

CRC.@non_differentiable make_causal_mask(::Any...)
