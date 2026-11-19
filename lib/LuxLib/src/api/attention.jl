@doc doc"""
    scaled_dot_product_attention(
        query, key, value; head_dim=1, token_dim=3,
        scale=nothing, mask=nothing, is_causal=nothing, bias=nothing,
        fdrop=identity, return_attention_weights=true
    )

Compute scaled dot-product attention for general tensors.

This function is a thin API wrapper that forwards to the implementation in
`impl/attention.jl` which supports flexible placement of the `head` and
`token` dimensions (via `head_dim` and `token_dim`) and additional features
such as grouped key/value (KV) sharing, optional masking, and dropout on the
attention weights.

## Arguments

- `query`, `key`, `value`: N‑dimensional arrays (N ≥ 4). The first three logical
  dimensions correspond to an embedding/head dimension, a heads dimension,
  and a token (sequence) dimension, but their positions are controlled
  by the `head_dim` and `token_dim` keyword arguments. All non-first-3
  dimensions (batching dims) must match between `q`, `k`, and `v`.

## Keyword arguments

- `head_dim::Int=1`: Which of the first three dimensions indexes the per-head
  embedding dimension.
- `token_dim::Int=3`: Which of the first three dimensions indexes tokens.
  `head_dim` and `token_dim` must be different and must be in `1:3`.
- `scale=nothing`: Scaling factor applied to the dot products. By default
  `scale = sqrt(size(q, head_dim))` and the implementation uses the inverse
  (i.e. divides by `sqrt(d_head)`). You may pass a number to override.
- `mask=nothing`: Optional boolean mask broadcastable to the attention logits.
  When provided it selects which key positions are allowed (true = keep).
- `is_causal::Union{Bool,Nothing}=nothing`: If `true`, causal attention is
  requested from NNlib without materializing a mask on fused backends.
  `is_causal` and `mask` cannot both be specified.
- `bias=nothing`: Optional additive bias applied to attention logits before
  softmax.
- `return_attention_weights::Bool=true`: Return attention weights alongside the output.
  Set to `false` to return `(output, nothing)` and enable fused attention backends.
- `fdrop=identity`: Function applied to the attention probabilities (e.g.
  a dropout function). It is applied after softmax and after mask/bias are
  applied.

# Extended Help

## Mathematical Formulation

```math
\begin{align*}
&\text{Let } q_{d,h,l,b},\ k_{d,h,s,b},\ v_{e,h,s,b} \text{ denote the elements of the query, key, and value tensors,} \\
&\text{where:} \\
&\quad d: \text{embedding dimension (head\_dim)} \\
&\quad h: \text{head index} \\
&\quad l: \text{query token index} \\
&\quad s: \text{key/value token index} \\
&\quad e: \text{value embedding dimension} \\
&\quad b: \text{batch index} \\
\\
&\text{Scaling:} \\
&\quad \text{scale} = \frac{1}{\sqrt{d_{\text{head}}}} \text{ (default, overridable)} \\
\\
&\text{Attention logits:} \\
&\quad \text{logits}_{s,l,h,b} = \text{scale} \cdot \sum_{d=1}^{d_{\text{head}}} k_{d,h,s,b} \, q_{d,h,l,b} + \text{bias}_{s,l,h,b} \\
&\quad \text{If mask is provided: logits}_{s,l,h,b} \rightarrow -\infty \text{ where mask is false} \\
\\
&\text{Attention weights:} \\
&\quad \alpha_{s,l,h,b} = \text{softmax}_s(\text{logits}_{:,l,h,b}) \\
\\
&\text{Output:} \\
&\quad \text{output}_{e,l,h,b} = \sum_{s=1}^{S} v_{e,h,s,b} \, \alpha_{s,l,h,b} \\
\end{align*}
```

## Behavior notes

- The implementation asserts `N ≥ 4` and enforces that `q` and `k` have the
  same size along `head_dim`, and that `k` and `v` have the same token length
  (along `token_dim`).
- Grouped KV: NNlib supports sharing fewer key/value heads across query heads.
- Masking, bias, causal attention, and dropout are handled by NNlib.

## Returns

- A tuple `(output, attn_weights)` where
    - `output` contains the attended values with the same batch/head layout as
      `q`/`v` (see implementation for exact ordering), and
    - `attn_weights` contains the attention probabilities (softmax over the
      key/token dimension), or `nothing` when `return_attention_weights=false`.

## Performance

- Delegates to NNlib's attention implementations. With
  `return_attention_weights=false`, supported backends can use fused attention
  without materializing the attention weights. This includes Reactant's optimized
  attention lowering when available.
"""
function scaled_dot_product_attention(
    q::AbstractArray, k::AbstractArray, v::AbstractArray; kwargs...
)
    return scaled_dot_product_attention_impl(q, k, v; kwargs...)
end
