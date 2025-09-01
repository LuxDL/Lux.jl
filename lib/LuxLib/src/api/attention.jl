function scaled_dot_product_attention(
    q::AbstractArray, k::AbstractArray, v::AbstractArray; kwargs...
)
    return scaled_dot_product_attention_impl(q, k, v; kwargs...)
end
