@doc doc"""
    scaled_dot_product_attention(
        query, key, value; head_dim=1, token_dim=3,
        scale=nothing, mask=nothing, is_causal=nothing, bias=nothing,
        fdrop=identity
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
- `is_causal::Union{Bool,Nothing}=nothing`: If `true` a causal mask is
  constructed automatically with `NNlib.make_causal_mask(...; dims=token_dim)`.
  `is_causal` and `mask` cannot both be specified.
- `bias=nothing`: Optional additive bias applied to attention logits before
  softmax.
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
- Grouped KV: If `k`/`v` provide fewer KV groups than `q`'s number of heads,
  they are repeated along the non-head batching dimension so that the number
  of KV groups divides the number of heads. This supports e.g. shared keys for
  multiple heads.
- Masking and bias are combined using the internal `apply_attention_mask_bias`
  helper which produces numerically stable logits before softmax.

## Returns

- A tuple `(output, attn_weights)` where
    - `output` contains the attended values with the same batch/head layout as
      `q`/`v` (see implementation for exact ordering), and
    - `attn_weights` contains the attention probabilities (softmax over the
      key/token dimension).

## Performance

- CPU: The implementation relies on `batched_matmul` (which may use
  LoopVectorization when available) for good CPU performance.
- Reactant: Lowered to a `dot_general`-style op. It is highly recommended to use this
  function instead of writing out the sdpa implementation directly.
"""
function scaled_dot_product_attention(
    q::AbstractArray, k::AbstractArray, v::AbstractArray; kwargs...
)
    return scaled_dot_product_attention_impl(q, k, v; kwargs...)
end
