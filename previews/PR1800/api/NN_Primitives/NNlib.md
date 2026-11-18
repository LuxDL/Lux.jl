---
url: /previews/PR1800/api/NN_Primitives/NNlib.md
---

# NNlib {#NNlib-API}

Neural Network Primitives with custom bindings for different accelerator backends in Julia.

::: tip Reexport of `NNlib`

Lux doesn't re-export all of `NNlib` for now. Directly loading `NNlib` is the recommended approach for accessing these functions.

:::

## Attention {#Attention}

```julia
scaled_dot_product_attention(query, key, value, [bias]; [fdrop, mask, scale, is_causal, window, q_offset, q_lengths, kv_lengths, q_segments, kv_segments, maxlen])
sdpa(query, key, value, [bias]; kws...)
```

Multihead dot product attention used in transformer architectures, with an explicit head axis.

The input arrays must have the shape `(head_dim, nheads, seq_len, batch_size...)`, that is a feature (head) dimension, a heads dimension, a sequence-length dimension, then an arbitrary number of batch dimensions or none. The number of heads is taken from the second dimension of each input, so no `nheads` keyword is needed.

This is the reverse of the `(batch_size, seq_len, nheads, head_dim)` layout of JAX and Flax. It is not the reverse of PyTorch's `(batch_size, nheads, seq_len, head_dim)`: an array ported from PyTorch by reversing its dimensions has size `(head_dim, seq_len, nheads, batch_size)` and still needs `permutedims(x, (1, 3, 2, 4))`. A missing permutation cannot be detected, since the array is a valid input whose heads are taken as the sequence.

Grouped-query attention (GQA) is supported: if `key` and `value` have fewer heads than `query`, they are shared across query heads. The number of query heads must then be divisible by the number of key/value heads.

Returns the attention output array of size `(v_head_dim, nheads, q_len, batch_size...)`.

See also [`scaled_dot_product_attention_with_weights`](/api/NN_Primitives/NNlib#NNlib.scaled_dot_product_attention_with_weights) if you need the attention weights too.

**Arguments**

* `query`: Query array of size `(qk_head_dim, nheads, q_len, batch_size...)`.

* `key`: Key array of size `(qk_head_dim, nkvheads, kv_len, batch_size...)`.

* `value`: Value array of size `(v_head_dim, nkvheads, kv_len, batch_size...)`.

* `bias`: Either `nothing` or an array broadcastable to size `(kv_len, q_len, nheads, batch_size...)`.         It will be added to the attention logits before applying the softmax. Default `nothing`.

* `fdrop`: A dropout function or layer to be applied on the attention weights right after the softmax.          Default `identity` (no dropout).

* `mask`: Either `nothing` or a boolean array broadcastable to size `(kv_len, q_len, nheads, batch_size...)`.         The mask is applied to the attention logits just before the softmax.         See [`make_causal_mask`](/api/NN_Primitives/NNlib#NNlib.make_causal_mask) for creating causal masks. Default `nothing`.

* `scale`: The multiplier used to scale the `queryᵀ * key` dot products before         the softmax. Default `nothing`, meaning `1 / √(qk_head_dim)`.

* `is_causal`: If `true`, a causal mask is applied so that each query position can         only attend to key positions up to and including its own. Equivalent to         `mask = make_causal_mask(query; dims=3)` for equal query and key lengths, but         a fused implementation does not have to materialize the mask. If         `:bottom_right`, the mask is aligned with the last valid key instead of the         first: the queries are taken as the last positions of the keys, as when         decoding with a KV cache, and `q_offset` is `kv_lengths .- q_lengths`, these         being the sizes of the arrays where not given. This is the causal mask of         FlashAttention and PyTorch's `causal_lower_right`. Default `false`.

* `window`: Either `nothing` or a positive integer, the size of a sliding window: the         query `i` attends only to the keys `j` with `|q_offset + i - j| < window`. With         `is_causal`, these are the last `window` keys up to its own position,         `q_offset + i - window < j ≤ q_offset + i`, as in the sliding-window layers of         Mistral, Gemma or gpt-oss. This is the `sliding_window` of transformers, and the         `window_size` of FlashAttention is `(window - 1, 0)` with `is_causal`,         `(window - 1, window - 1)` without. A fused implementation only goes through         the keys of the windows. Default `nothing`.

* `q_offset`: With `is_causal=true`, the number of keys that precede the first query:         the query `i` attends to the keys `j ≤ q_offset + i`. An integer, or an integer         array with one entry for each batch element. It counts the keys given, which         with a sliding-window cache is not the absolute position. It has no effect         without `is_causal` or `window`, and is set by `is_causal=:bottom_right`.         Default `0`.

* `q_lengths`, `kv_lengths`: Either `nothing` or the number of valid positions of each         query and of each key/value sequence in the batch, the following ones being         padding: an integer array with one entry for each batch element, or an         integer for all of them. The keys past `kv_lengths` are not attended to, and         the output is zero at the queries past `q_lengths`. Each can be given without         the other. Default `nothing`.

* `q_segments`, `kv_segments`: Either `nothing` or the lengths of the sequences packed         one after the other along the sequence dimension of `query` and of `key` and         `value`, as an integer vector. See below. `kv_segments` defaults to `q_segments`.

* `maxlen`: With `q_segments`, either `nothing` or an integer at least as large as every         segment. It saves reading the lengths of the segments on the host where they         are on a GPU, which waits for the device, and stands in for them where they         cannot be read, as the traced ones of Reactant. Default `nothing`.

`mask`, `is_causal`, `window` and `kv_lengths` can be combined: a key is then attended to only where all of them allow it. A query that attends to no key, as a padding query past the window of every valid key, has a `NaN` output, unless `q_lengths` makes it zero.

**Packed sequences**

With `q_segments`, the inputs hold sequences of different lengths one after the other, without padding: they have size `(head_dim, nheads, total_len)`, without batch dimensions, and the output has the size of `query`. The `s`-th segment of `query` attends to the `s`-th segment of `key` and `value` only, as a batch element does in a padded batch: `is_causal` and `window` apply within each segment, and `q_offset`, `q_lengths` and `kv_lengths` take one entry for each segment. The lengths of the segments have to add up to the sequence lengths of the arrays. A `bias` or a `mask` is not supported.

This is how documents packed in a sequence are kept from attending to each other, and how a batch of sequences of different lengths avoids padding: layers other than the attention act on the `(features, total_len)` array of all the positions.

**Fused implementation**

Without an `fdrop`, the output is computed tile by tile, with the softmax taken online over the tiles of keys: the attention weights are never materialized, so that memory grows linearly rather than quadratically with the sequence length. This is what `Array`s and GPU arrays of `Float16`, `Float32` and `Float64` take, and on the GPU those of `BFloat16`, with any of the arguments above. The `bias` and the `mask`, if any, have to be on the device of the inputs. On the GPU, a call with few queries per sequence, as in decoding with a key-value cache, splits the keys of each sequence over several blocks, and reads them once for all the query heads sharing them.

Any other call takes the materialized implementation, which forms the weights. [`NNlib.is_fused_attention`](/api/NN_Primitives/NNlib#NNlib.is_fused_attention) tells which of the two a call takes. With a `window`, it forms them over the band of the window only, for blocks of queries with the keys of their windows, so that memory grows linearly with the sequence length there too. `fdrop` is then given these blocks of weights rather than the whole array.

The fused implementation has a gradient with respect to `query`, `key`, `value` and `scale`, which recomputes the weights tile by tile and does not materialize them either. It is what automatic differentiation based on ChainRules (as that of Zygote) and the reverse mode of Enzyme use. Under them a call with a `bias` takes the materialized implementation, which gives the gradient with respect to the bias too, and so does every call under the other automatic differentiation packages. A query that attends to no key, whose output is `NaN`, takes no part in the fused gradient.

With CUDA.jl and cuDNN.jl loaded, the `Float16` and `BFloat16` `CuArray`s that cuDNN's own fused kernel has an engine for are handed to it: it runs on tensor cores, and has a gradient. It supports `scale`, `is_causal`, `q_lengths`, `kv_lengths` and grouped-query attention. cuDNN.jl 6.5 also supports bottom-right causal masks, sliding windows and packed sequences. It requires:

* no `bias`, `fdrop` or `mask` (that is `nothing`, `identity` and `nothing`);

* with an active causal mask, top-left or bottom-right alignment; without one, a window must be top-left aligned;

* a head dimension that is a multiple of 8 and at most 256;

* a `value` with the same head dimension as `query`.

Earlier cuDNN.jl versions require no window or packed sequences, and a top-left causal mask when it excludes any keys. Unsupported calls use NNlib's own implementation.

**Examples**

```julia
# (head_dim, nheads, seq_len, batch)
q = rand(Float32, 16, 8, 20, 4)
k = v = rand(Float32, 16, 8, 30, 4)
y = scaled_dot_product_attention(q, k, v)
# size(y) == (16, 8, 20, 4)

# grouped-query attention: 8 query heads sharing 2 key/value heads
q = rand(Float32, 16, 8, 20, 4)
k = v = rand(Float32, 16, 2, 30, 4)
y = scaled_dot_product_attention(q, k, v)

# a padded batch, whose first key/value sequence has 12 valid positions out of 30
y = scaled_dot_product_attention(q, k, v; kv_lengths=[12, 30, 30, 30])

# decoding with a KV cache of 64 positions: 3 new queries for each batch element,
# appended to the 10, 20, 30 and 40 keys already cached
q = rand(Float32, 16, 8, 3, 4)
k = v = rand(Float32, 16, 8, 64, 4)
y = scaled_dot_product_attention(q, k, v; is_causal=:bottom_right, kv_lengths=[13, 23, 33, 43])
# the same, with the offsets of the queries
y = scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=[10, 20, 30, 40],
                                 kv_lengths=[13, 23, 33, 43])

# sliding-window attention: each query attends to its last 8 keys, its own included
x = rand(Float32, 16, 8, 100, 4)
y = scaled_dot_product_attention(x, x, x; is_causal=true, window=8)

# three documents of 5, 12 and 3 positions packed in one sequence, each attending
# causally to itself only
x = rand(Float32, 16, 8, 20)
y = scaled_dot_product_attention(x, x, x; is_causal=true, q_segments=[5, 12, 3])
# size(y) == (16, 8, 20)
```

source

```julia
scaled_dot_product_attention_with_weights(query, key, value, [bias]; kws...)
sdpa_with_weights(query, key, value, [bias]; kws...)
```

Return the output of [`scaled_dot_product_attention`](/api/NN_Primitives/NNlib#NNlib.scaled_dot_product_attention) together with the attention weights it is computed from, in a single pass.

The arguments are those of [`scaled_dot_product_attention`](/api/NN_Primitives/NNlib#NNlib.scaled_dot_product_attention). The weights have size `(kv_len, q_len, nheads, batch_size...)`, with `α[j, i, h, b]` the weight of the key `j` for the query `i` in the head `h`: the softmax over the keys of the attention logits, after `fdrop`. They are zero at the queries past `q_lengths`. With `q_segments` they have size `(kv_len, q_len, nheads)`, and are zero but for the keys of the segment of each query.

Returning the weights means materializing them, so this function never takes a fused implementation: prefer [`scaled_dot_product_attention`](/api/NN_Primitives/NNlib#NNlib.scaled_dot_product_attention) when they are not needed.

**Examples**

```julia
q = rand(Float32, 16, 8, 20, 4)
k = v = rand(Float32, 16, 8, 30, 4)
y, α = scaled_dot_product_attention_with_weights(q, k, v)
# size(y) == (16, 8, 20, 4)
# size(α) == (30, 20, 8, 4)
```

source

```julia
dot_product_attention(query, key, value, [bias]; [nheads, fdrop, mask])
```

Deprecated: use [`scaled_dot_product_attention`](/api/NN_Primitives/NNlib#NNlib.scaled_dot_product_attention), which takes inputs with an explicit head axis `(head_dim, nheads, seq_len, batch_size...)` and returns only the attention output, or [`scaled_dot_product_attention_with_weights`](/api/NN_Primitives/NNlib#NNlib.scaled_dot_product_attention_with_weights) for the attention weights too.

Multihead dot product attention on inputs of size `(features, seq_len, batch_size...)`, whose heads are packed into the feature dimension and split according to `nheads`. Returns the attention output of size `(v_dim, q_len, batch_size...)` and the attention weights of size `(kv_len, q_len, nheads, batch_size...)`.

source

```julia
dot_product_attention_scores(query, key, [bias]; [fdrop, mask])
```

Deprecated: use [`scaled_dot_product_attention_with_weights`](/api/NN_Primitives/NNlib#NNlib.scaled_dot_product_attention_with_weights).

source

```julia
make_causal_mask(x; dims)
```

Return a boolean square matrix `m` of the same type as `x` and of side `size(x, dims)`. Its elements are set such that `m[i, j] == i ≤ j`.

Can be used as the `mask` of [`scaled_dot_product_attention`](/api/NN_Primitives/NNlib#NNlib.scaled_dot_product_attention), whose inputs have the sequence length along `dims=3`. For a plain causal mask, prefer the `is_causal=true` keyword there, which lets a fused implementation avoid materializing the mask.

Calling `make_causal_mask(x)` without `dims` is deprecated. It keeps returning the mask for `dims=2`, the sequence dimension of the deprecated `dot_product_attention`.

source

```julia
NNlib.is_fused_attention(query, key, value, [bias]; kws...)
```

Return `true` if [`scaled_dot_product_attention`](/api/NN_Primitives/NNlib#NNlib.scaled_dot_product_attention) called with the same arguments is computed by a fused implementation, which does not materialize the attention weights, and `false` if it takes the materialized one.

This is about a call outside of automatic differentiation: under it, the materialized implementation is differentiated where the fused one has no gradient, as with a `bias`.

source

### Rotary Position Embedding {#Rotary-Position-Embedding}

```julia
NNlib.rotary_embedding_table(head_dim, maxlen, [T=Float32]; base=10_000)
```

Return the tables `(cos, sin)` of the rotary position embedding (RoPE) of [Su et al.](https://arxiv.org/abs/2104.09864) for the positions `1:maxlen`, to be passed to [`NNlib.apply_rotary_embedding`](/api/NN_Primitives/NNlib#NNlib.apply_rotary_embedding).

Each is a `Matrix{T}` of size `(head_dim ÷ 2, maxlen)`: the cosine and the sine of the angle `θ[i] * (p - 1)` by which the `i`-th pair of features is rotated at the position `p`, with the frequencies `θ[i] = base^(-2(i - 1) / head_dim)`. The first position is not rotated, as the position zero of PyTorch and JAX.

The tables are computed in `Float64` on the CPU: move them to the device of the inputs. Other frequencies, as those rescaled by Llama 3 or YaRN, give their own tables `cos.(θ .* (0:maxlen - 1)')` and `sin.(θ .* (0:maxlen - 1)')`.

**Examples**

```julia
julia> cos_table, sin_table = NNlib.rotary_embedding_table(64, 4096);

julia> size(cos_table), size(sin_table)
((32, 4096), (32, 4096))
```

source

```julia
NNlib.apply_rotary_embedding(x, cos, sin; offset=0, positions=nothing, interleaved=false)
```

Apply the rotary position embedding (RoPE) of [Su et al.](https://arxiv.org/abs/2104.09864) to `x`, a query or a key of [`scaled_dot_product_attention`](/api/NN_Primitives/NNlib#NNlib.scaled_dot_product_attention) of size `(head_dim, nheads, seq_len, batch_size...)`: the features of each head are rotated pair by pair, by angles proportional to the position of the token. The dot product of a query and a key so rotated depends on their positions through their difference only. The values are not rotated.

The tables `cos` and `sin` have size `(head_dim ÷ 2, maxlen)`, their column `p` holding the cosines and the sines of the angles of the position `p`, as those of [`NNlib.rotary_embedding_table`](/api/NN_Primitives/NNlib#NNlib.rotary_embedding_table). At the position `p`, the `k`-th pair of features `(a, b)` becomes `(a * cos[k, p] - b * sin[k, p], b * cos[k, p] + a * sin[k, p])`. The pairs are `(x[k], x[k + head_dim ÷ 2])`, from the two halves of the features (as in Llama, Hugging Face Transformers and Flax), or with `interleaved=true` the consecutive features `(x[2k - 1], x[2k])` (as in RoFormer, GPT-J and torchtune).

Returns an array of the size of `x`.

**Keywords**

* `offset`: The token `i` is at the position `offset + i`. An integer, or an integer array         with one entry for each batch element. With a KV cache, the number of tokens         already in the cache, as the `q_offset` of [`scaled_dot_product_attention`](/api/NN_Primitives/NNlib#NNlib.scaled_dot_product_attention).         Default `0`.

* `positions`: Either `nothing` or the positions of the tokens, in place of `offset`, as an         integer array of size `(seq_len, batch_size...)`, or `(seq_len,)` for all the         batch elements. They are the columns of the tables. Default `nothing`.

* `interleaved`: If `true`, the pairs are the consecutive features, else the two halves.         Default `false`.

The positions have to be within the columns of the tables. To rotate part of the features only (as GPT-NeoX and Phi), rotate the slice `x[1:d, :, :, :]` with tables of `d ÷ 2` rows and concatenate the rest.

**Examples**

The causal self-attention of a decoder, in which the queries and the keys are rotated but not the values, here with 8 heads of dimension 64 sharing 2 key/value heads:

```julia
using NNlib: apply_rotary_embedding, rotary_embedding_table

cos_table, sin_table = rotary_embedding_table(64, 4096)
q = rand(Float32, 64, 8, 100, 4)   # (head_dim, nheads, seq_len, batch_size)
k = rand(Float32, 64, 2, 100, 4)
v = rand(Float32, 64, 2, 100, 4)
q = apply_rotary_embedding(q, cos_table, sin_table)
k = apply_rotary_embedding(k, cos_table, sin_table)
y = scaled_dot_product_attention(q, k, v; is_causal=true)
```

Decoding with a KV cache, which holds the keys already rotated: the new tokens are rotated with the number of cached tokens as offset, the `q_offset` of the attention.

```julia
kcache = apply_rotary_embedding(rand(Float32, 64, 2, 10, 4), cos_table, sin_table)
vcache = rand(Float32, 64, 2, 10, 4)
# 3 new tokens at the positions 11 to 13
q = apply_rotary_embedding(rand(Float32, 64, 8, 3, 4), cos_table, sin_table; offset=10)
knew = apply_rotary_embedding(rand(Float32, 64, 2, 3, 4), cos_table, sin_table; offset=10)
k = cat(kcache, knew; dims=3)
v = cat(vcache, rand(Float32, 64, 2, 3, 4); dims=3)
y = scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=10)
```

Documents packed in one sequence (see [`scaled_dot_product_attention`](/api/NN_Primitives/NNlib#NNlib.scaled_dot_product_attention)), whose positions start over at each document:

```julia
segments = [5, 12, 3]
positions = reduce(vcat, [1:n for n in segments])   # [1:5; 1:12; 1:3]
q = apply_rotary_embedding(rand(Float32, 64, 8, 20), cos_table, sin_table; positions)
k = apply_rotary_embedding(rand(Float32, 64, 2, 20), cos_table, sin_table; positions)
v = rand(Float32, 64, 2, 20)
y = scaled_dot_product_attention(q, k, v; is_causal=true, q_segments=segments)
```

source

## State-Space and Linear-Attention Layers {#State-Space-and-Linear-Attention-Layers}

```julia
NNlib.causal_conv1d(x, w; bias=nothing, state=nothing, lengths=nothing) -> (y, new_state)
```

The causal depthwise convolution of the sequences `x`, of size `(channels, seq_len, batch_size)`, with the filters `w`, of size `(kernel_size, channels)`: each channel has its own filter, and the output at a position depends on the inputs up to that position only,

```julia
y[c, t, b] = bias[c] + sum(w[j, c] * x[c, t - kernel_size + j, b] for j in 1:kernel_size)
```

This is the short convolution that Mamba-2, Gated DeltaNet and LFM2 apply before their sequence mixing (see [`NNlib.ssd_scan`](/api/NN_Primitives/NNlib#NNlib.ssd_scan) and [`NNlib.gated_delta_rule`](/api/NN_Primitives/NNlib#NNlib.gated_delta_rule)). As in PyTorch the filter is not flipped: `w[kernel_size, c]` multiplies the current input. The weight of PyTorch's `Conv1d(channels, channels, kernel_size; groups=channels)` has size `(channels, 1, kernel_size)`, which read in Julia's column-major order is `(kernel_size, 1, channels)`: drop the middle dimension.

The `kernel_size - 1` inputs before the first position are the columns of `state`, of size `(channels, kernel_size - 1, batch_size)`, or zeros. The returned `new_state` holds the last `kernel_size - 1` inputs of `state` continued by `x`. Convolving a sequence in pieces, each starting from the `new_state` of the previous one, gives the convolution of the whole sequence: this is how a decoder passes its tokens one at a time. On the GPU backends, a single token outside of differentiation is one kernel.

Returns `y`, of the size of `x`, and `new_state`.

**Keywords**

* `bias`: Either `nothing` or a vector of length `channels`. Default `nothing`.

* `state`: Either `nothing`, meaning zeros, or the inputs before `x`, of size         `(channels, kernel_size - 1, batch_size)`. Default `nothing`.

* `lengths`: Either `nothing` or the number of valid positions of each sequence of a batch         padded on the right: an integer, or an integer array with one entry for each         batch element. `new_state` then holds the last inputs before the end of each         sequence, and the outputs past it are not meaningful. Default `nothing`, meaning         all the positions.

**Examples**

The convolution of a prompt followed by the decoding of one token, with Mamba-2's `kernel_size = 4` and its SiLU:

```julia
w, bias = randn(Float32, 4, 64), randn(Float32, 64)
x = randn(Float32, 64, 10, 2)                  # (channels, seq_len, batch_size)
y, state = NNlib.causal_conv1d(x, w; bias)
y = swish.(y)

x1 = randn(Float32, 64, 1, 2)                  # the next token
y1, state = NNlib.causal_conv1d(x1, w; bias, state)
```

source

```julia
NNlib.ssd_scan(x, dt, A, B, C; D=nothing, initial_state=nothing, lengths=nothing,
               chunk_size=128) -> (y, final_state)
```

The selective scan of Mamba-2, in the state space duality (SSD) form of [Dao and Gu](https://arxiv.org/abs/2405.21060). Each head carries a state `S` of size `(head_dim, state_size)` along the sequence. At the token `t`, with the step `Δ = dt[h, t]` of the head `h`,

```julia
S = exp(Δ * A[h]) * S + Δ * x[:, h, t] * B[:, t]ᵀ
y[:, h, t] = S * C[:, t] + D[h] * x[:, h, t]
```

The arrays have the layout of [`scaled_dot_product_attention`](/api/NN_Primitives/NNlib#NNlib.scaled_dot_product_attention), and the heads are in groups that share `B` and `C`, as the queries share keys and values there:

* `x`: The inputs, of size `(head_dim, nheads, seq_len, batch_size)`.

* `dt`: The positive steps, of size `(nheads, seq_len, batch_size)`. In Mamba-2, the       `softplus` of a projection of the input plus `dt_bias`, clamped.

* `A`: The negative rates of decay of the heads, a vector of length `nheads`. In Mamba-2,       `-exp.(A_log)`.

* `B`, `C`: The projections that write the input into the state and read the state into       the output, of size `(state_size, ngroups, seq_len, batch_size)`, with `nheads` a       multiple of `ngroups`. The heads `(i - 1) * r + 1` to `i * r`, with       `r = nheads ÷ ngroups`, use the group `i`.

Returns `y`, of the size and element type of `x`, and the state after the last token, `final_state`, of size `(head_dim, state_size, nheads, batch_size)`. It is computed in at least `Float32`, as the outputs, and kept in that type.

**Keywords**

* `D`: Either `nothing` or the vector of the skip connections of the heads. Default      `nothing`.

* `initial_state`: Either `nothing`, meaning zeros, or the states before the first token,      of size `(head_dim, state_size, nheads, batch_size)`, as the `final_state` of the      previous part of the sequences. Default `nothing`.

* `lengths`: Either `nothing` or the number of valid positions of each sequence of a batch      padded on the right: an integer, or an integer array with one entry for each batch      element. `final_state` is then the state at the end of each sequence, and the      outputs past it are not meaningful. Default `nothing`, meaning all the positions.

* `chunk_size`: The number of tokens of the chunks of the chunked algorithm. Default `128`.

A sequence of a single token takes one step of the recurrence, as when decoding. Longer sequences are cut in chunks of `chunk_size` tokens, computed in parallel by matrix products, as `mamba_chunk_scan_combined` of [mamba_ssm](https://github.com/state-spaces/mamba) and its PyTorch version in Hugging Face Transformers: the states at the ends of the chunks are combined by a product with a matrix of their decays. The chunk size trades memory, `(chunk_size, chunk_size)` per head and chunk, against the `(head_dim, state_size)` state per head and chunk.

On the GPU backends, the chunked algorithm runs as kernels, as those of mamba_ssm: they form `Cᵀ B` once for each group of heads and apply the decays as they read it, carry the states from chunk to chunk with one pass over them, and have a gradient that keeps only the states at the starts of the chunks. The kernels take chunks of at most 256 tokens. The gradient is that of the kernels under ChainRules (Zygote); other differentiation packages differentiate the generic implementation. Outside of differentiation, a single token is one kernel, which reads and writes the states once, for a `state_size` of at most 256.

The state of Transformers' `ssm_state`, of size `(batch_size, nheads, head_dim, state_size)`, is `permutedims(final_state, (4, 3, 1, 2))`.

**Examples**

The scan of a prompt followed by the decoding of one token, with continuing states:

```julia
P, H, N, G = 64, 8, 128, 1       # head_dim, nheads, state_size, ngroups
x = randn(Float32, P, H, 10, 2)  # (head_dim, nheads, seq_len, batch_size)
dt = softplus.(randn(Float32, H, 10, 2))
A = -exp.(randn(Float32, H))
B, C = randn(Float32, N, G, 10, 2), randn(Float32, N, G, 10, 2)
y, state = NNlib.ssd_scan(x, dt, A, B, C)

x1, dt1 = randn(Float32, P, H, 1, 2), softplus.(randn(Float32, H, 1, 2))
B1, C1 = randn(Float32, N, G, 1, 2), randn(Float32, N, G, 1, 2)
y1, state = NNlib.ssd_scan(x1, dt1, A, B1, C1; initial_state=state)
```

source

```julia
NNlib.gated_delta_rule(q, k, v, g, beta; scale=nothing, initial_state=nothing,
                       lengths=nothing, chunk_size=64) -> (o, final_state)
```

The gated delta rule of Gated DeltaNet ([Yang et al.](https://arxiv.org/abs/2412.06464)), the linear attention of Qwen3-Next and Qwen3.5. Each head carries a state `S` of size `(head_v_dim, head_k_dim)` along the sequence. At the token `t`, it decays by `exp(g)`, then its value at the key `k_t` moves toward `v_t` at the rate `beta`, and is read at `q_t`:

```julia
S = exp(g[h, t]) * S
S = S + beta[h, t] * (v_t - S * k_t) * k_tᵀ
o_t = S * (scale * q_t)
```

The arrays have the layout of [`scaled_dot_product_attention`](/api/NN_Primitives/NNlib#NNlib.scaled_dot_product_attention), and as there the queries and keys can have fewer heads than the values:

* `q`, `k`: The queries and the keys, of size `(head_k_dim, nkheads, seq_len, batch_size)`.       The delta rule expects keys of unit norm: Qwen3-Next and Qwen3.5 normalise the       queries and the keys (Transformers' `use_qk_l2norm_in_kernel`).

* `v`: The values, of size `(head_v_dim, nheads, seq_len, batch_size)`, with `nheads` a       multiple of `nkheads`. The heads `(i - 1) * r + 1` to `i * r`, with       `r = nheads ÷ nkheads`, use the query and key head `i`.

* `g`: The logarithms of the decays, nonpositive, of size `(nheads, seq_len, batch_size)`.       In Gated DeltaNet, `-exp(A_log) * softplus(a + dt_bias)`.

* `beta`: The rates of the writes, in `[0, 1]`, of size `(nheads, seq_len, batch_size)`.

Returns the outputs `o`, of size `(head_v_dim, nheads, seq_len, batch_size)` and the element type of `v`, and the state after the last token, `final_state`, of size `(head_v_dim, head_k_dim, nheads, batch_size)`. It is computed in at least `Float32`, as the outputs, and kept in that type.

**Keywords**

* `scale`: The multiplier of the queries. Default `nothing`, meaning `1 / √head_k_dim`.

* `initial_state`: Either `nothing`, meaning zeros, or the states before the first token,      of size `(head_v_dim, head_k_dim, nheads, batch_size)`, as the `final_state` of the      previous part of the sequences. Default `nothing`.

* `lengths`: Either `nothing` or the number of valid positions of each sequence of a batch      padded on the right: an integer, or an integer array with one entry for each batch      element. `final_state` is then the state at the end of each sequence, and the      outputs past it are not meaningful. Default `nothing`, meaning all the positions.

* `chunk_size`: The number of tokens of the chunks of the chunked algorithm, a power of 2.      Default `64`.

A sequence of a single token takes one step of the recurrence, as when decoding, as `torch_recurrent_gated_delta_rule` of Transformers. Longer sequences are cut in chunks of up to `chunk_size` tokens, as in `torch_chunk_gated_delta_rule`: within a chunk, the writes are found by solving a unit triangular system, here by inverting its matrix by blocks, and the outputs are matrix products. The state is carried from one chunk to the next by a loop over the chunks.

On the GPU backends, the chunked algorithm runs as kernels, as those of flash-linear-attention: the products of the queries and the keys are formed once for each head of keys, the triangular inverse of each chunk in shared memory, and the state is carried through the chunks by blocks of its rows. The gradient keeps the states at the starts of the chunks. The kernels take chunks of at most 64 tokens. The gradient is that of the kernels under ChainRules (Zygote); other differentiation packages differentiate the generic implementation. Outside of differentiation, a single token is one kernel, which reads and writes the states once, for a `head_k_dim` of at most 256.

The state of Transformers' `recurrent_state`, of size `(batch_size, nheads, head_k_dim, head_v_dim)`, is `permutedims(final_state, (4, 3, 2, 1))`.

**Examples**

The gated delta rule of a prompt followed by the decoding of one token, with continuing states, here with 4 heads of values sharing 2 heads of queries and keys:

```julia
l2norm(x) = x ./ sqrt.(sum(abs2, x; dims=1) .+ 1f-6)
q, k = l2norm(randn(Float32, 128, 2, 10, 3)), l2norm(randn(Float32, 128, 2, 10, 3))
v = randn(Float32, 128, 4, 10, 3)           # (head_v_dim, nheads, seq_len, batch_size)
g, beta = -rand(Float32, 4, 10, 3), rand(Float32, 4, 10, 3)
o, state = NNlib.gated_delta_rule(q, k, v, g, beta)

q1, k1 = l2norm(randn(Float32, 128, 2, 1, 3)), l2norm(randn(Float32, 128, 2, 1, 3))
v1, g1, beta1 = randn(Float32, 128, 4, 1, 3), -rand(Float32, 4, 1, 3), rand(Float32, 4, 1, 3)
o1, state = NNlib.gated_delta_rule(q1, k1, v1, g1, beta1; initial_state=state)
```

source

## Softmax {#Softmax}

```julia
softmax(x; dims = 1)
```

[Softmax](https://en.wikipedia.org/wiki/Softmax_function) turns input array `x` into probability distributions that sum to 1 along the dimensions specified by `dims`. It is semantically equivalent to the following:

```julia
softmax(x; dims = 1) = exp.(x) ./ sum(exp.(x), dims = dims)
```

with additional manipulations enhancing numerical stability. For `Float16` and `BFloat16` inputs, the sums are computed in `Float32`.

For a matrix input `x` it will by default (`dims = 1`) treat it as a batch of vectors, with each column independent. Keyword `dims = 2` will instead treat rows independently, and so on.

See also [`logsoftmax`](/api/NN_Primitives/NNlib#NNlib.logsoftmax).

**Examples**

```julia
julia> softmax([1, 2, 3])
3-element Vector{Float64}:
 0.09003057317038046
 0.24472847105479764
 0.6652409557748218

julia> softmax([1 2 3; 2 2 2])  # dims=1
2×3 Matrix{Float64}:
 0.268941  0.5  0.731059
 0.731059  0.5  0.268941

julia> softmax([1 2 3; 2 2 2]; dims=2)
2×3 Matrix{Float64}:
 0.0900306  0.244728  0.665241
 0.333333   0.333333  0.333333
```

Note that, when used with Flux.jl, `softmax` must not be passed to layers like `Dense` which accept an activation function. The activation is broadcasted over the result, thus applies to individual numbers. But `softmax` always needs to see the whole column.

```julia
julia> using Flux

julia> x = randn(Float32, 4, 4, 3, 13);

julia> model = Chain(Conv((4, 4), 3 => 8, tanh), Flux.flatten, Dense(8 => 7), softmax);

julia> model(x) |> size
(7, 13)

julia> Dense(4 => 7, softmax)(x)
ERROR: `softmax(x)` called with a number, but it expects an array. 
```

source

```julia
softmax!(out, x; dims = 1)
softmax!(x; dims = 1)
```

In-place version of [`softmax`](/api/NN_Primitives/NNlib#NNlib.softmax), writing the result into `out` (or `x` itself).

source

```julia
logsoftmax(x; dims = 1)
```

Computes the log of softmax in a more numerically stable way than directly taking `log.(softmax(xs))`. Commonly used in computing cross entropy loss.

It is semantically equivalent to the following:

```julia
logsoftmax(x; dims = 1) = x .- log.(sum(exp.(x), dims = dims))
```

For `Float16` and `BFloat16` inputs, the sum is computed in `Float32`.

See also [`softmax`](/api/NN_Primitives/NNlib#NNlib.softmax).

source

```julia
logsoftmax!(out, x; dims = 1)
logsoftmax!(x; dims = 1)
```

In-place version of [`logsoftmax`](/api/NN_Primitives/NNlib#NNlib.logsoftmax), writing the result into `out` (or `x` itself).

source

```julia
∇softmax(dy, y; dims = 1)
```

Gradient of [`softmax`](/api/NN_Primitives/NNlib#NNlib.softmax) with respect to its input, given the output `y = softmax(x; dims)` and the upstream cotangent `dy`. Does not depend on `x`.

source

```julia
∇softmax!(dx, dy, y; dims = 1)
```

In-place version of [`∇softmax`](/api/NN_Primitives/NNlib#NNlib.∇softmax), writing the gradient into `dx`.

source

```julia
∇logsoftmax(dy, y; dims = 1)
```

Gradient of [`logsoftmax`](/api/NN_Primitives/NNlib#NNlib.logsoftmax) with respect to its input, given the output `y = logsoftmax(x; dims)` and the upstream cotangent `dy`. Does not depend on `x`.

source

```julia
∇logsoftmax!(dx, dy, y; dims = 1)
```

In-place version of [`∇logsoftmax`](/api/NN_Primitives/NNlib#NNlib.∇logsoftmax), writing the gradient into `dx`.

source

## Pooling {#Pooling}

```julia
PoolDims(x_size::NTuple{M}, k::Union{NTuple{<:Any, Int}, Int};
         stride=k, padding=0, dilation=1, ceil_mode=false)  where M
```

Dimensions for a "pooling" operation that can have an arbitrary input size, kernel size, stride, dilation, and channel count.  Used to dispatch onto efficient implementations at compile-time.

With `ceil_mode=true` the output size is rounded up instead of down, as with PyTorch's `ceil_mode=True`: a last window that runs past the padded input is kept, as long as it starts inside the input or the low padding. Its positions past the padded input count neither as input nor as padding: max pooling skips them, and mean pooling leaves them out of the divisor even with `count_include_pad=true`. `ceil_mode` is dropped when it doesn't change the output size.

source

```julia
maxpool(x, k::NTuple{N, Integer}; pad=0, stride=k, ceil_mode=false)
```

Perform max pool operation with window size `k` on input tensor `x`.

Arguments:

* `x` and `k`: Expects `ndim(x) ∈ 3:5`, and always `length(k) == ndim(x) - 2`

* `pad`: See [`pad_zeros`](/api/NN_Primitives/NNlib#NNlib.pad_zeros) for details.

* `stride`: Either a tuple with the same length as `k`, or one integer for all directions. Default is `k`.

* `ceil_mode`: When `true`, round the output size up instead of down, as PyTorch's `ceil_mode=True`: a last window that runs past the padded input is kept if it starts inside the input or the low padding. See [`PoolDims`](/api/NN_Primitives/NNlib#NNlib.PoolDims).

source

```julia
meanpool(x, k::NTuple{N, Integer}; pad=0, stride=k, count_include_pad=true, ceil_mode=false)
```

Perform mean pool operation with window size `k` on input tensor `x`.

Arguments:

* `x` and `k`: Expects `ndim(x) ∈ 3:5`, and always `length(k) == ndim(x) - 2`

* `pad`: See [`pad_zeros`](/api/NN_Primitives/NNlib#NNlib.pad_zeros) for details.

* `stride`: Either a tuple with the same length as `k`, or one integer for all directions. Default is `k`.

* `count_include_pad`: When `true` (the default), padding (zero) elements are included in the averaging denominator, so each output element divides by the full window size `prod(k)`. When `false`, only the in-bounds (non-padding) elements are counted, matching cuDNN's `CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING` and PyTorch's `count_include_pad=false`. Has no effect when `pad == 0`.

* `ceil_mode`: When `true`, round the output size up instead of down, as PyTorch's `ceil_mode=True`: a last window that runs past the padded input is kept if it starts inside the input or the low padding. Its positions past the padded input are left out of the denominator also with `count_include_pad=true`, as in PyTorch. See [`PoolDims`](/api/NN_Primitives/NNlib#NNlib.PoolDims).

source

```julia
lpnormpool(x, p::Real, k::NTuple{N, Integer}; pad=0, stride=k, ceil_mode=false)
```

Perform Lp pool operation with value of the Lp norm `p` and window size `k` on input tensor `x`, also known as LPPool in pytorch. This pooling operator from [Learned-Norm Pooling for Deep Feedforward and Recurrent Neural Networks](https://arxiv.org/abs/1311.1780).

Arguments:

* `x` and `k`: Expects `ndim(x) ∈ 3:5`, and always `length(k) == ndim(x) - 2`

* `p` is restricted to `0 < p < Inf`.

* `pad`: See [`pad_zeros`](/api/NN_Primitives/NNlib#NNlib.pad_zeros) for details.

* `stride`: Either a tuple with the same length as `k`, or one integer for all directions. Default is `k`.

* `ceil_mode`: When `true`, round the output size up instead of down: a last window that runs past the padded input is kept if it starts inside the input or the low padding. See [`PoolDims`](/api/NN_Primitives/NNlib#NNlib.PoolDims).

For all elements `x` in a size `k` window, lpnormpool computes `(∑ᵢ xᵢ^p)^(1 / p)` as an element of the output.

Thus `lpnormpool(x, 1, k) ./ prod(k) ≈ meanpool(x, k)` and `lpnormpool(x, 2, k).^2 ./ prod(k) ≈ meanpool(x.^2, k)`. With `ceil_mode=true`, the sum of a last window that runs past the input is over its elements in the input; PyTorch's `LPPool` instead scales it by `prod(k)` over their number.

source

## Padding {#Padding}

```julia
pad_reflect(x, pad::Tuple; [dims])
pad_reflect(x, pad::Int; [dims])
```

Pad the array `x` reflecting its values across the border.

`pad` can a tuple of integers `(l1, r1, ..., ln, rn)` of some length `2n` that specifies the left and right padding size for each of the dimensions in `dims`. If `dims` is not given,  it defaults to the first `n` dimensions.

If `pad` is an integer, it is applied on both sides on every dimension in `dims`. In this case, `dims`  defaults to the first `ndims(x)-2` dimensions  (i.e. excludes the channel and batch dimension).

See also [`pad_repeat`](/api/NN_Primitives/NNlib#NNlib.pad_repeat), [`pad_symmetric`](/api/NN_Primitives/NNlib#NNlib.pad_symmetric), [`pad_circular`](/api/NN_Primitives/NNlib#NNlib.pad_circular), and [`pad_constant`](/api/NN_Primitives/NNlib#NNlib.pad_constant).

```julia
julia> r = reshape(1:9, 3, 3)
3×3 reshape(::UnitRange{Int64}, 3, 3) with eltype Int64:
 1  4  7
 2  5  8
 3  6  9

julia> pad_reflect(r, (1,2,1,2))
6×6 Matrix{Int64}:
 5  2  5  8  5  2
 4  1  4  7  4  1
 5  2  5  8  5  2
 6  3  6  9  6  3
 5  2  5  8  5  2
 4  1  4  7  4  1
```

source

```julia
pad_symmetric(x, pad::Tuple; [dims])
pad_symmetric(x, pad::Int; [dims])
```

Pad the array `x` reflecting its values symmetrically across the border, i.e. the border values of `x` are present in the padding values, in contrast to [`pad_reflect`](/api/NN_Primitives/NNlib#NNlib.pad_reflect).

`pad` can a tuple of integers `(l1, r1, ..., ln, rn)` of some length `2n` that specifies the left and right padding size for each of the dimensions in `dims`. If `dims` is not given,  it defaults to the first `n` dimensions.

If `pad` is an integer, it is applied on both sides on every dimension in `dims`. In this case, `dims`  defaults to the first `ndims(x)-2` dimensions  (i.e. excludes the channel and batch dimension).

See also [`pad_repeat`](/api/NN_Primitives/NNlib#NNlib.pad_repeat), [`pad_reflect`](/api/NN_Primitives/NNlib#NNlib.pad_reflect), [`pad_circular`](/api/NN_Primitives/NNlib#NNlib.pad_circular), and [`pad_constant`](/api/NN_Primitives/NNlib#NNlib.pad_constant).

```julia
julia> r = reshape(1:9, 3, 3)
3×3 reshape(::UnitRange{Int64}, 3, 3) with eltype Int64:
 1  4  7
 2  5  8
 3  6  9

julia> pad_symmetric(r, (1,2,1,2))
6×6 Matrix{Int64}:
 1  1  4  7  7  4
 1  1  4  7  7  4
 2  2  5  8  8  5
 3  3  6  9  9  6
 3  3  6  9  9  6
 2  2  5  8  8  5
```

source

```julia
pad_circular(x, pad::Tuple; [dims])
pad_circular(x, pad::Int; [dims])
```

Pad the array `x` "circularly" across the border by wrapping around values from the opposite side of `x`.

`pad` can a tuple of integers `(l1, r1, ..., ln, rn)` of some length `2n` that specifies the left and right padding size for each of the dimensions in `dims`. If `dims` is not given,  it defaults to the first `n` dimensions.

If `pad` is an integer, it is applied on both sides on every dimension in `dims`. In this case, `dims`  defaults to the first `ndims(x)-2` dimensions  (i.e. excludes the channel and batch dimension).

The pad length on either side in any dimension must not exceed the size of `x` in that dimension, i.e. `pad_circular` is not able to create abitrary sized tilings of `x`.

See also [`pad_repeat`](/api/NN_Primitives/NNlib#NNlib.pad_repeat), [`pad_reflect`](/api/NN_Primitives/NNlib#NNlib.pad_reflect), [`pad_symmetric`](/api/NN_Primitives/NNlib#NNlib.pad_symmetric), and [`pad_constant`](/api/NN_Primitives/NNlib#NNlib.pad_constant).

```julia
julia> r = reshape(1:9, 3, 3)
3×3 reshape(::UnitRange{Int64}, 3, 3) with eltype Int64:
 1  4  7
 2  5  8
 3  6  9

julia> pad_circular(r, (1,2,1,2))
6×6 Matrix{Int64}:
 9  3  6  9  3  6
 7  1  4  7  1  4
 8  2  5  8  2  5
 9  3  6  9  3  6
 7  1  4  7  1  4
 8  2  5  8  2  5
```

source

```julia
pad_repeat(x, pad::Tuple; [dims])
pad_repeat(x, pad::Int; [dims])
```

Pad the array `x` repeating the values on the border.

`pad` can a tuple of integers `(l1, r1, ..., ln, rn)` of some length `2n` that specifies the left and right padding size for each of the dimensions in `dims`. If `dims` is not given,  it defaults to the first `n` dimensions.

If `pad` is an integer, it is applied on both sides on every dimension in `dims`. In this case, `dims`  defaults to the first `ndims(x)-2` dimensions  (i.e. excludes the channel and batch dimension).

See also [`pad_reflect`](/api/NN_Primitives/NNlib#NNlib.pad_reflect), [`pad_symmetric`](/api/NN_Primitives/NNlib#NNlib.pad_symmetric), [`pad_circular`](/api/NN_Primitives/NNlib#NNlib.pad_circular), and [`pad_constant`](/api/NN_Primitives/NNlib#NNlib.pad_constant).

```julia
julia> r = reshape(1:9, 3, 3)
3×3 reshape(::UnitRange{Int64}, 3, 3) with eltype Int64:
 1  4  7
 2  5  8
 3  6  9

julia> pad_repeat(r, (1,2,3,4))
6×10 Matrix{Int64}:
 1  1  1  1  4  7  7  7  7  7
 1  1  1  1  4  7  7  7  7  7
 2  2  2  2  5  8  8  8  8  8
 3  3  3  3  6  9  9  9  9  9
 3  3  3  3  6  9  9  9  9  9
 3  3  3  3  6  9  9  9  9  9
```

source

```julia
pad_constant(x, pad::Tuple, val = 0; [dims = :])
pad_constant(x, pad::Int, val = 0; [dims = :])
```

Pad the array `x` with the constant value `val`.

`pad` can be a tuple of integers. If it is of some length `2 * length(dims)` that specifies the left and right padding size for each of the dimensions in `dims` as `(l1, r1, ..., ln, rn)`.  If supplied with a tuple of length `length(dims)` instead, it applies symmetric padding. If `dims` is not given, it defaults to all dimensions.

For integer `pad` input, it is applied on both sides on every dimension in `dims`.

See also [`pad_zeros`](/api/NN_Primitives/NNlib#NNlib.pad_zeros), [`pad_repeat`](/api/NN_Primitives/NNlib#NNlib.pad_repeat), [`pad_reflect`](/api/NN_Primitives/NNlib#NNlib.pad_reflect), [`pad_symmetric`](/api/NN_Primitives/NNlib#NNlib.pad_symmetric), and [`pad_circular`](/api/NN_Primitives/NNlib#NNlib.pad_circular).

```julia
julia> r = reshape(1:4, 2, 2)
2×2 reshape(::UnitRange{Int64}, 2, 2) with eltype Int64:
 1  3
 2  4

julia> pad_constant(r, (1, 2, 3, 4), 8)
5×9 Matrix{Int64}:
 8  8  8  8  8  8  8  8  8
 8  8  8  1  3  8  8  8  8
 8  8  8  2  4  8  8  8  8
 8  8  8  8  8  8  8  8  8
 8  8  8  8  8  8  8  8  8

julia> pad_constant(r, 1, 8)
4×4 Matrix{Int64}:
 8  8  8  8
 8  1  3  8
 8  2  4  8
 8  8  8  8

julia> r = reshape(1:27, 3, 3, 3)
3×3×3 reshape(::UnitRange{Int64}, 3, 3, 3) with eltype Int64:
[:, :, 1] =
 1  4  7
 2  5  8
 3  6  9

[:, :, 2] =
 10  13  16
 11  14  17
 12  15  18

[:, :, 3] =
 19  22  25
 20  23  26
 21  24  27

julia> pad_constant(r, (2,1), dims = 1) # assymetric padding
6×3×3 Array{Int64, 3}:
[:, :, 1] =
 0  0  0
 0  0  0
 1  4  7
 2  5  8
 3  6  9
 0  0  0

[:, :, 2] =
  0   0   0
  0   0   0
 10  13  16
 11  14  17
 12  15  18
  0   0   0

[:, :, 3] =
  0   0   0
  0   0   0
 19  22  25
 20  23  26
 21  24  27
  0   0   0

julia> pad_constant(r, (2,1, 3), dims = (1,2)) # padding must always be either the same length as dims, or double it
ERROR: ArgumentError: Could not parse padding (2, 1, 3) and dims (1, 2)
Stacktrace:
[...]
```

source

```julia
pad_zeros(x, pad::Tuple; [dims])
pad_zeros(x, pad::Int; [dims])
```

Pad the array `x` with zeros. Equivalent to [`pad_constant`](/api/NN_Primitives/NNlib#NNlib.pad_constant) with the constant equal to 0.

source

## Convolution {#Convolution}

```julia
conv(x, w; stride = 1, pad = 0, dilation = 1, flipped = false, groups = 1)
```

Apply convolution filter `w` to input `x`. `x` and `w` are 3d/4d/5d tensors in 1d/2d/3d convolutions respectively. They may have real or complex element types.

The dimensions are arranged as follows, with the spatial dimensions first:

* `x`: `(spatial..., channels_in, batch)`

* `w`: `(spatial..., channels_in ÷ groups, channels_out)`

* output: `(spatial_out..., channels_out, batch)`

The output spatial size for each dimension is `(spatial + 2*pad - (dilation*(spatial_w - 1) + 1)) ÷ stride + 1`.

**Keyword arguments**

* `stride`: stride of the filter (`Int` or tuple of one `Int` per spatial dimension).

* `pad`: zero padding. An `Int` or tuple applies symmetric padding; a tuple of `2*ndims` entries gives `(left, right)` padding per spatial dimension.

* `dilation`: spacing between filter elements (atrous convolution).

* `flipped`: if `false` (default), perform true mathematical convolution; if `true`, perform cross-correlation (the usual deep-learning convention).

* `groups`: number of groups for grouped convolution.

**Examples**

```julia
julia> x = rand(Float32, 28, 28, 3, 16);  # (width, height, channels_in, batch)

julia> w = rand(Float32, 5, 5, 3, 8);     # (width, height, channels_in, channels_out)

julia> size(conv(x, w))                    # 28 - 5 + 1 = 24
(24, 24, 8, 16)

julia> size(conv(x, w; stride=2, pad=1))
(13, 13, 8, 16)
```

source

```julia
ConvDims
```

Type system-level information about convolution dimensions. Critical for things like `im2col!()` to generate efficient code, and helpful to reduce the number of kwargs getting passed around.

source

```julia
depthwiseconv(x, w; stride=1, pad=0, dilation=1, flipped=false)
```

Depthwise convolution operation with filter `w` on input `x`. `x` and `w` are 3d/4d/5d tensors in 1d/2d/3d convolutions respectively. The filter `w` has shape `(spatial..., channel_multiplier, channels_in)`.

This is equivalent to a grouped [`conv`](/api/NN_Primitives/NNlib#NNlib.conv) with `groups = channels_in`, and is implemented in terms of it.

source

```julia
DepthwiseConvDims
```

Concrete subclass of `ConvDims` for a depthwise convolution.  Differs primarily due to characterization by `C_in`, `C_mult`, rather than `C_in`, `C_out`.  Useful to be separate from DenseConvDims primarily for channel calculation differences.

source

```julia
DenseConvDims
```

Concrete subclass of `ConvDims` for a normal, dense, conv2d/conv3d.

source

```julia
unfold(x, kernel_size; stride = 1, pad = 0, dilation = 0, flipped = true)
```

Places sliding windows of x into a container tensor of size `(num_windows, window_size, batchsize)`. The window size is determined by the `prod(spatial dims of kernel)*input_channels`. The number of sliding windows will match those of convolution (`conv`) with the same kernel_size and arguments. Note that by default `conv` flips the spatial dimensions of its kernel (default `flipped=false`), whereas `unfold` does not (default `flipped=true`). Uses `NNlib.im2col!` as backend.

See also [`fold`](/api/NN_Primitives/NNlib#NNlib.fold), the adjoint/transpose operator and a potential inverse of `unfold`.

**Example**

The below example demonstrates that `unfold` uses the same sliding windows as `conv`. In general [`batched_mul`](/api/NN_Primitives/NNlib#NNlib.batched_mul) + `unfold` should not be used to achieve convolution.

```julia
julia> x = reshape([100 2 3 40 5 6 700], 7, 1, 1);  # 1D data, 1 channel, batch of 1

julia> w = reshape([1 0 -1], 3, 1, 1);  # 1D conv kernel of length 3

julia> kws = (pad=1, stride=2, flipped=true);  # use same args for conv and unfold

julia> z = NNlib.unfold(x, size(w); kws...)
4×3×1 Array{Int64, 3}:
[:, :, 1] =
  0  100   2
  2    3  40
 40    5   6
  6  700   0

julia> y1 = conv(x, w; kws...)
4×1×1 Array{Int64, 3}:
[:, :, 1] =
  -2
 -38
  34
   6

julia> y2 = z ⊠ w  # ⊠ (\boxtimes) is NNlib.batched_mul
4×1×1 Array{Int64, 3}:
[:, :, 1] =
  -2
 -38
  34
   6
```

source

```julia
fold(y, output_size, kernel_size; stride = 1, pad = 0, dilation = 0, flipped = true)
```

The adjoint/transpose operator of `unfold`. It accumulates sliding windows from the output of `unfold` into a container tensor of size `output_size`. An inverse to `unfold` may be obtained (in some cases) by using `fold` and accounting for scaling issues with a divisor (see example). Uses `NNlib.col2im!` as backend.

See also [`unfold`](/api/NN_Primitives/NNlib#NNlib.unfold).

**Example**

```julia
julia> x = reshape([100 2 3 40 5 6 700], 7, 1, 1);  # 1D data, 1 channel, batch of 1

julia> y = NNlib.unfold(x, (3,1,1))  # sliding window of size 3
5×3×1 Array{Int64, 3}:
[:, :, 1] =
 100   2    3
   2   3   40
   3  40    5
  40   5    6
   5   6  700

julia> z = NNlib.fold(y, size(x), (3,1,1))  # sum of contributions in y. 100 appears once, 40 three times
7×1×1 Array{Int64, 3}:
[:, :, 1] =
 100
   4
   9
 120
  15
  12
 700

julia> divisor = NNlib.fold(NNlib.unfold(ones(size(x)...), (3,1,1)), size(x), (3,1,1))
7×1×1 Array{Float64, 3}:
[:, :, 1] =
 1.0
 2.0
 3.0
 3.0
 3.0
 2.0
 1.0

julia> z ./ divisor
7×1×1 Array{Float64, 3}:
[:, :, 1] =
 100.0
   2.0
   3.0
  40.0
   5.0
   6.0
 700.0
```

In general, an inverse to `unfold` does not exist if `divisor` contains zeros.

source

## Upsampling {#Upsampling}

```julia
upsample_nearest(x, scale::NTuple{S,Int})
upsample_nearest(x; size::NTuple{S,Int})
```

Upsamples the array `x` by integer multiples along the first `S` dimensions. Subsequent dimensions of `x` are not altered.

Either the `scale` factors or the final output `size` can be specified.

See also [`upsample_bilinear`](/api/NN_Primitives/NNlib#NNlib.upsample_bilinear), for two dimensions of an `N=4` array.

**Example**

```julia
julia> upsample_nearest([1 2 3; 4 5 6], (2, 3))
4×9 Matrix{Int64}:
 1  1  1  2  2  2  3  3  3
 1  1  1  2  2  2  3  3  3
 4  4  4  5  5  5  6  6  6
 4  4  4  5  5  5  6  6  6

julia> ans == upsample_nearest([1 2 3; 4 5 6]; size=(4, 9))  # equivalent
true

julia> upsample_nearest([1 2 3; 4 5 6], (2,))
4×3 Matrix{Int64}:
 1  2  3
 1  2  3
 4  5  6
 4  5  6

julia> ans == upsample_nearest([1 2 3; 4 5 6], size=(4,))
true
```

source

```julia
∇upsample_nearest(Δ::AbstractArray{T,3}, scales::NTuple{S, <:Integer}) where T
```

**Arguments**

* `Δ`: Incoming gradient array, backpropagated from downstream layers

* `scales`: scales by which the image was upsampled in the first place

**Outputs**

* `dx`: Downsampled version of `Δ`

source

```julia
upsample_linear(x::AbstractArray{T,3}, scale::Real; align_corners::Bool = true)
upsample_linear(x::AbstractArray{T,3}; size::Integer, align_corners::Bool = true)
```

Upsamples the first dimension of the array `x` by the upsample provided `scale`, using linear interpolation. As an alternative to using `scale`, the resulting array `size` can be directly specified with a keyword argument.

The size of the output is equal to `(scale*S1, S2, S3)`, where `S1, S2, S3 = size(x)`.

source

```julia
∇upsample_linear(Δ::AbstractArray{T,3}; size::Integer, align_corners::Bool = true) where T
```

**Arguments**

* `Δ`: Incoming gradient array, backpropagated from downstream layers

* `size`: Size of the image upsampled in the first place

**Outputs**

* `dx`: Downsampled version of `Δ`

source

```julia
upsample_bilinear(x::AbstractArray{T,4}, scale::NTuple{2,Real}; align_corners::Bool = true, antialias::Bool = false)
upsample_bilinear(x::AbstractArray{T,4}; size::NTuple{2,Integer}, align_corners::Bool = true, antialias::Bool = false)
```

Upsamples the first 2 dimensions of the array `x` by the upsample factors stored in `scale`, using bilinear interpolation. As an alternative to using `scale`, the resulting image `size` can be directly specified with a keyword argument.

The size of the output is equal to `(scale[1]*S1, scale[2]*S2, S3, S4)`, where `S1, S2, S3, S4 = size(x)`.

With `antialias = true`, the output can also be smaller than the input: as in PyTorch's `F.interpolate(x, size, mode="bilinear", antialias=True)` and PIL, the triangle filter is then stretched by the downsampling factor and renormalised at the borders. Pass `align_corners = false` to match them. See also [`upsample_bicubic`](/api/NN_Primitives/NNlib#NNlib.upsample_bicubic).

**Examples**

```julia
julia> x = reshape(Float32[1 2 3; 4 5 6], (2,3,1,1))
2×3×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 1.0  2.0  3.0
 4.0  5.0  6.0

julia> upsample_bilinear(x, (2, 3))
4×9×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 1.0  1.25  1.5  1.75  2.0  2.25  2.5  2.75  3.0
 2.0  2.25  2.5  2.75  3.0  3.25  3.5  3.75  4.0
 3.0  3.25  3.5  3.75  4.0  4.25  4.5  4.75  5.0
 4.0  4.25  4.5  4.75  5.0  5.25  5.5  5.75  6.0

julia> ans == upsample_bilinear(x; size=(4, 9))  # specify ouput size instead
true

julia> upsample_bilinear(x, (2.5, 3.5))  # non-integer scaling factors are allowed
5×10×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 1.0   1.22222  1.44444  1.66667  1.88889  …  2.33333  2.55556  2.77778  3.0
 1.75  1.97222  2.19444  2.41667  2.63889     3.08333  3.30556  3.52778  3.75
 2.5   2.72222  2.94444  3.16667  3.38889     3.83333  4.05556  4.27778  4.5
 3.25  3.47222  3.69444  3.91667  4.13889     4.58333  4.80556  5.02778  5.25
 4.0   4.22222  4.44444  4.66667  4.88889     5.33333  5.55556  5.77778  6.0
```

source

```julia
∇upsample_bilinear(Δ::AbstractArray{T,4}; size::NTuple{2,Integer}, align_corners::Bool = true, antialias::Bool = false) where T
```

**Arguments**

* `Δ`: Incoming gradient array, backpropagated from downstream layers

* `size`: Lateral (W,H) size of the image upsampled in the first place

**Outputs**

* `dx`: Downsampled version of `Δ`

source

```julia
upsample_bicubic(x::AbstractArray{T,4}, scale::NTuple{2,Real}; align_corners::Bool = false, antialias::Bool = false)
upsample_bicubic(x::AbstractArray{T,4}; size::NTuple{2,Integer}, align_corners::Bool = false, antialias::Bool = false)
```

Resizes the first 2 dimensions of the array `x` with bicubic interpolation, to the image `size` given as a keyword argument, or to `floor.(Int, scale .* size(x)[1:2])`. The output can be smaller than the input.

This is PyTorch's `F.interpolate(x, size, mode="bicubic", align_corners=..., antialias=...)` on the same data in the NCHW layout. Without `antialias`, each output pixel is a weighted sum of 4×4 input pixels (the cubic convolution kernel with `a = -0.75`, as in OpenCV), and the image is extended by repeating its border pixels. With `antialias = true`, the kernel is the one of PIL and torchvision's `Resize` (`a = -0.5`): when downsampling, it is stretched by the downsampling factor, so that each output pixel averages all the input pixels it covers, and at the borders the weights are renormalised over the pixels inside the image.

Unlike [`upsample_bilinear`](/api/NN_Primitives/NNlib#NNlib.upsample_bilinear), `align_corners` defaults to `false`, as in PyTorch.

**Examples**

```julia
julia> x = reshape(Float32[1 2; 3 4], (2, 2, 1, 1));

julia> upsample_bicubic(x, (2, 2))  # overshoots the range of `x`
4×4×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 0.683594  1.01562  1.5625   1.89453
 1.34766   1.67969  2.22656  2.55859
 2.44141   2.77344  3.32031  3.65234
 3.10547   3.4375   3.98438  4.31641

julia> ans == upsample_bicubic(x; size = (4, 4))
true

julia> x = reshape(Float32.(1:16), (4, 4, 1, 1));

julia> upsample_bicubic(x; size = (2, 2), antialias = true)
2×2×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 3.93388  11.2397
 5.76033  13.0661
```

source

```julia
∇upsample_bicubic(Δ::AbstractArray{T,4}; size::NTuple{2,Integer}, align_corners::Bool = false, antialias::Bool = false)
```

**Arguments**

* `Δ`: Incoming gradient array, backpropagated from downstream layers

* `size`: Lateral (W,H) size of the image resized in the first place

**Outputs**

* `dx`: Gradient with respect to the input of [`upsample_bicubic`](/api/NN_Primitives/NNlib#NNlib.upsample_bicubic)

source

```julia
upsample_trilinear(x::AbstractArray{T,5}, scale::NTuple{3,Real}; align_corners::Bool = true)
upsample_trilinear(x::AbstractArray{T,5}; size::NTuple{3,Integer}, align_corners::Bool = true)
```

Upsamples the first 3 dimensions of the array `x` by the upsample factors stored in `scale`, using trilinear interpolation. As an alternative to using `scale`, the resulting image `size` can be directly specified with a keyword argument.

The size of the output is equal to `(scale[1]*S1, scale[2]*S2, scale[3]*S3, S4, S5)`, where `S1, S2, S3, S4, S5 = size(x)`.

**Examples**

```julia
upsample_trilinear(x, (2, 3, 4))
upsample_trilinear(x; size=(4, 9, 11))  # specify ouput size instead
upsample_trilinear(x, (2.5, 3.5, pi))  # non-integer scaling factors are allowed
```

source

```julia
∇upsample_trilinear(Δ::AbstractArray{T,5}; size::NTuple{3,Integer}, align_corners::Bool = true) where T
```

**Arguments**

* `Δ`: Incoming gradient array, backpropagated from downstream layers

* `size`: Lateral size & depth (W,H,D) of the image upsampled in the first place

**Outputs**

* `dx`: Downsampled version of `Δ`

source

```julia
pixel_shuffle(x, r::Integer)
```

Pixel shuffling operation, upscaling by a factor `r`.

For 4-arrays representing `N` images, the operation converts input `size(x) == (W, H, r^2*C, N)` to output of size `(r*W, r*H, C, N)`. For `D`-dimensional data, it expects `ndims(x) == D+2` with channel and batch dimensions, and divides the number of channels by `r^D`.

Used in super-resolution networks to upsample towards high resolution features. Reference: Shi et. al., "Real-Time Single Image and Video Super-Resolution ...", CVPR 2016, https://arxiv.org/abs/1609.05158

**Examples**

```julia
julia> x = [10i + j + channel/10 for i in 1:2, j in 1:3, channel in 1:4, batch in 1:1]
2×3×4×1 Array{Float64, 4}:
[:, :, 1, 1] =
 11.1  12.1  13.1
 21.1  22.1  23.1

[:, :, 2, 1] =
 11.2  12.2  13.2
 21.2  22.2  23.2

[:, :, 3, 1] =
 11.3  12.3  13.3
 21.3  22.3  23.3

[:, :, 4, 1] =
 11.4  12.4  13.4
 21.4  22.4  23.4

julia> pixel_shuffle(x, 2)  # 4 channels used up as 2x upscaling of image dimensions
4×6×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 11.1  11.3  12.1  12.3  13.1  13.3
 11.2  11.4  12.2  12.4  13.2  13.4
 21.1  21.3  22.1  22.3  23.1  23.3
 21.2  21.4  22.2  22.4  23.2  23.4

julia> y = [i + channel/10 for i in 1:3, channel in 1:6, batch in 1:1]
3×6×1 Array{Float64, 3}:
[:, :, 1] =
 1.1  1.2  1.3  1.4  1.5  1.6
 2.1  2.2  2.3  2.4  2.5  2.6
 3.1  3.2  3.3  3.4  3.5  3.6

julia> pixel_shuffle(y, 2)  # 1D image, with 6 channels reduced to 3
6×3×1 Array{Float64, 3}:
[:, :, 1] =
 1.1  1.3  1.5
 1.2  1.4  1.6
 2.1  2.3  2.5
 2.2  2.4  2.6
 3.1  3.3  3.5
 3.2  3.4  3.6
```

source

## Rotation {#Rotation}

Rotate images in the first two dimensions of an array.

```julia
imrotate(arr::AbstractArray{T, 4}, θ; method=:bilinear, rotation_center=size(arr) .÷ 2 .+ 1)
```

Rotates an array in the first two dimensions around the center pixel `rotation_center`.  The default value of `rotation_center` is defined such that there is a integer center pixel for even and odd sized arrays which it is rotated around. For an even sized array of size `(4,4)` this would be `(3,3)`, for an odd array of size `(3,3)` this would be `(2,2)` However, `rotation_center` can be also non-integer numbers if specified.

The angle `θ` is interpreted in radians.

The adjoint is defined with ChainRulesCore.jl. This method also runs with CUDA (and in principle all KernelAbstractions.jl supported backends).

**Keywords**

* `method=:bilinear` for bilinear interpolation or `method=:nearest` for nearest neighbour

* `rotation_center=size(arr) .÷ 2 .+ 1` means there is a real center pixel around it is rotated.

**Examples**

```julia
julia> arr = zeros((4,4,1,1)); arr[2,2,1,1] = 1;

julia> arr
4×4×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

julia> NNlib.imrotate(arr, deg2rad(90)) # rotation around (3,3)
4×4×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

julia> NNlib.imrotate(arr, deg2rad(90), rotation_center=(2,2))
4×4×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

julia> arr = zeros((3,3,1,1)); arr[1,2,1,1] = 1
1

julia> arr
3×3×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  1.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> NNlib.imrotate(arr, deg2rad(45))
3×3×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.207107  0.0
 0.0  0.0       0.207107
 0.0  0.0       0.0

julia> NNlib.imrotate(arr, deg2rad(45), method=:nearest)
3×3×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.0  1.0
 0.0  0.0  0.0
 0.0  0.0  0.0
```

source

```julia
∇imrotate(dy, arr::AbstractArray{T, 4}, θ; method=:bilinear,
                                           rotation_center=size(arr) .÷ 2 .+ 1)
```

Adjoint for `imrotate`. Gradient only with respect to `arr` and not `θ`.

**Arguments**

* `dy`: input gradient

* `arr`: Input from primal computation

* `θ`: rotation angle in radians

* `method=:bilinear` or `method=:nearest`

* `rotation_center=size(arr) .÷ 2 .+ 1` rotates around a real center pixel for even and odd sized arrays

source

## Batched Operations {#Batched-Operations}

```julia
batched_mul(A, B) -> C
A ⊠ B  # \boxtimes
```

Batched matrix multiplication. Result has `C[:,:,k...] == A[:,:,k...] * B[:,:,k...]` where `k...` represent  any indices in the last dimensions.

If `ndims(A) == ndims(B) == 3` and `size(B,3) == 1` then instead `C[:,:,k] == A[:,:,k] * B[:,:,1]`, and similarly for `A`.

To transpose each matrix, apply `batched_transpose` to the array, or `batched_adjoint` for conjugate-transpose:

```julia
julia> A, B = randn(2,5,17), randn(5,9,17);

julia> A ⊠ B |> size
(2, 9, 17)

julia> batched_adjoint(A) |> size
(5, 2, 17)

julia> batched_mul(A, batched_adjoint(randn(9,5,17))) |> size
(2, 9, 17)

julia> A ⊠ randn(5,9,1) |> size
(2, 9, 17)

julia> batched_transpose(A) == PermutedDimsArray(A, (2,1,3))
true
```

The equivalent `PermutedDimsArray` may be used in place of `batched_transpose`. Other permutations are also handled by BLAS, provided that the batch index `k` is not the first dimension of the underlying array. Thus `PermutedDimsArray(::Array, (1,3,2))` and `PermutedDimsArray(::Array, (3,1,2))` are fine.

However, `A = PermutedDimsArray(::Array, (3,2,1))` is not acceptable to BLAS, since the batch dimension is the contiguous one: `stride(A,3) == 1`. This will be copied, as doing so is faster than `batched_mul_generic!`.

Both this `copy` and `batched_mul_generic!` produce `@debug` messages, and setting for instance `ENV["JULIA_DEBUG"] = NNlib` will display them.

source

```julia
batched_mul(A::Array{T,3}, B::Matrix)
batched_mul(A::Matrix, B::Array{T,3})
A ⊠ B
```

This is always matrix-matrix multiplication, but either `A` or `B` may lack a batch index.

* When `B` is a matrix, result has `C[:,:,k] == A[:,:,k] * B[:,:]` for all `k`.

* When `A` is a matrix, then `C[:,:,k] == A[:,:] * B[:,:,k]`. This can also be done by reshaping and calling `*`, for instance `A ⊡ B` using TensorCore.jl, but is implemented here using `batched_gemm` instead of `gemm`.

```julia
julia> randn(16,8,32) ⊠ randn(8,4) |> size
(16, 4, 32)

julia> randn(16,8,32) ⊠ randn(8,4,1) |> size  # equivalent
(16, 4, 32)

julia> randn(16,8) ⊠ randn(8,4,32) |> size
(16, 4, 32)
```

See also `batched_vec` to regard `B` as a batch of vectors, `A[:,:,k] * B[:,k]`.

source

```julia
batched_mul!(C, A, B) -> C
batched_mul!(C, A, B, α=1, β=0)
```

In-place batched matrix multiplication, equivalent to `mul!(C[:,:,k], A[:,:,k], B[:,:,k], α, β)` for all `k`. If `size(B,3) == 1` then every batch uses `B[:,:,1]` instead.

This will call `batched_gemm!` whenever possible. For real arrays this means that, for `X ∈ [A,B,C]`, either `stride(X,1)==1` or `stride(X,2)==1`, the latter may be caused by `batched_transpose` or by for instance `PermutedDimsArray(::Array, (3,1,2))`. Unlike `batched_mul` this will never make a copy.

For complex arrays, the wrapper made by `batched_adjoint` must be outermost to be seen. In this case the strided accepted by BLAS are more restricted, if `stride(C,1)==1` then only `stride(AorB::BatchedAdjoint,2) == 1` is accepted.

source

```julia
batched_transpose(A::AbstractArray{T,3})
batched_adjoint(A)
```

Equivalent to applying `transpose` or `adjoint` to each matrix `A[:,:,k]`.

These exist to control how `batched_mul` behaves, as it operates on such matrix slices of an array with `ndims(A)==3`.

`PermutedDimsArray(A, (2,1,3))` is equivalent to `batched_transpose(A)`, and is also understood by `batched_mul` (and more widely supported elsewhere).

```julia
BatchedTranspose{T, S} <: AbstractBatchedMatrix{T, 3}
BatchedAdjoint{T, S}
```

Lazy wrappers analogous to `Transpose` and `Adjoint`, returned by `batched_transpose` etc.

source

```julia
batched_transpose(A::AbstractArray{T,3})
batched_adjoint(A)
```

Equivalent to applying `transpose` or `adjoint` to each matrix `A[:,:,k]`.

These exist to control how `batched_mul` behaves, as it operates on such matrix slices of an array with `ndims(A)==3`.

`PermutedDimsArray(A, (2,1,3))` is equivalent to `batched_transpose(A)`, and is also understood by `batched_mul` (and more widely supported elsewhere).

```julia
BatchedTranspose{T, S} <: AbstractBatchedMatrix{T, 3}
BatchedAdjoint{T, S}
```

Lazy wrappers analogous to `Transpose` and `Adjoint`, returned by `batched_transpose` etc.

source

```julia
batched_vec(A::AbstractArray{T,3}, B::AbstractMatrix)
batched_vec(A::AbstractArray{T,3}, b::AbstractVector)
batched_vec(A::AbstractArray, B::AbstractArray)
```

Batched matrix-vector multiplication. For the 3D case: the result has `C[:,:,k] == A[:,:,k] * B[:,k]` for all `k`, or else `C[:,:,k] == A[:,:,k] * b` for `b::Vector`.

For the general N-D case where `ndims(A) == ndims(B) + 1`: the result has `C[:,k...] == A[:,:,k...] * B[:,k...]` for all batch indices `k...`. The batch dimensions must match: `size(A)[3:end] == size(B)[2:end]`.

With the same argument types, `batched_mul(A, B)` would regard `B` as a fixed matrix, not a batch of vectors. Both reshape and then call `batched_mul(::Array{T,3}, ::Array{T,3})`.

```julia
julia> A, B, b = randn(16,8,32), randn(8,32), randn(8);

julia> batched_vec(A,B) |> size
(16, 32)

julia> batched_vec(A,b) |> size
(16, 32)

julia> A4d, B3d = randn(16,8,10,32), randn(8,10,32);  # 4D and 3D arrays

julia> batched_vec(A4d, B3d) |> size
(16, 10, 32)
```

source

```julia
grouped_mul(W::AbstractArray{<:Any,3}, X::AbstractMatrix, offsets) -> Y
grouped_mul(A::AbstractMatrix, B::AbstractMatrix, offsets) -> C
```

Matrix multiplication over groups of different sizes. `offsets` is an integer vector of nondecreasing cumulative group ends, starting from an implicit zero. Repeated offsets represent empty groups.

For `W` of size `(n, k, G)` and `X` of size `(k, total)`, return `Y` of size `(n, total)`, with `Y[:, r] = W[:, :, g] * X[:, r]` for each group `g`, where `r = (g == 1 ? 0 : offsets[g-1])+1:offsets[g]`. `length(offsets)` must equal `G`. `W` may be a [`batched_transpose`](/api/NN_Primitives/NNlib#NNlib.batched_transpose) or [`batched_adjoint`](/api/NN_Primitives/NNlib#NNlib.batched_adjoint).

For `A` of size `(n, total)` and `B` of size `(total, k)`, return `C` of size `(n, k, length(offsets))`, with `C[:, :, g] = A[:, r] * B[r, :]`.

Entries past the last offset are ignored. The corresponding columns of `Y`, and the matrices of `C` for empty groups, are zero. Both forms support differentiation.

On CUDA 6.5+, supported strided arrays use a single cuBLAS grouped GEMM, with `Float32` accumulation for `Float16` and `BFloat16`. Device offsets are copied to the host once per call, since cuBLAS requires group sizes on the host. Other arrays use a loop of `mul!` calls, parallelized over groups on multithreaded CPUs.

```julia
julia> W = cat(ones(2, 3), fill(2.0, 2, 3); dims=3);

julia> grouped_mul(W, ones(3, 4), [1, 3])
2×4 Matrix{Float64}:
 3.0  6.0  6.0  0.0
 3.0  6.0  6.0  0.0

julia> size(grouped_mul(ones(2, 4), ones(4, 3), [1, 3]))
(2, 3, 2)
```

source

```julia
grouped_mul!(C, A, B, offsets, α=1, β=0) -> C
```

Compute `C = α * grouped_mul(A, B, offsets) + β * C` in place. The destination must have the shape returned by [`grouped_mul`](/api/NN_Primitives/NNlib#NNlib.grouped_mul) and must not alias either input. Entries outside the groups are scaled by `β`. `α` and `β` are converted to the element type of `C`.

source

## Gather and Scatter {#Gather-and-Scatter}

```julia
NNlib.gather(src, idx; checkbounds = true) -> dst
```

Reverse operation of [`scatter`](/api/NN_Primitives/NNlib#NNlib.scatter). Gathers data from source `src` and writes it in a destination `dst` according to the index array `idx`. For each `k` in `CartesianIndices(idx)`, assign values to `dst` according to

```julia
dst[:, ... , k] .= src[:, ... , idx[k]...]
```

Notice that if `idx` is a vector containing integers and `src` is a matrix, previous expression simplifies to

```julia
dst[:, k] .= src[:, idx[k]]
```

and `k` will run over `1:length(idx)`.

The elements of `idx` can be integers or integer tuples and may be repeated. A single `src` column can end up being copied into zero, one, or multiple `dst` columns.

On a GPU, checking that the indices are in range synchronizes the host with the device. A caller that has checked them already can skip that with `checkbounds = false`: an index out of range then gives an undefined result, and may corrupt memory.

See [`gather!`](/api/NN_Primitives/NNlib#NNlib.gather!) for an in-place version.

**Examples**

```julia
julia> NNlib.gather([1,20,300,4000], [2,4,2])
3-element Vector{Int64}:
   20
 4000
   20

julia> NNlib.gather([1 2 3; 4 5 6], [1,3,1,3,1])
2×5 Matrix{Int64}:
 1  3  1  3  1
 4  6  4  6  4
```

source

```julia
gather(src, IJK...; checkbounds = true)
```

Convert the tuple of integer vectors `IJK` to a tuple of `CartesianIndex` and call `gather` on it: `gather(src, CartesianIndex.(IJK...))`.

**Examples**

```julia
julia> src = reshape([1:15;], 3, 5)
3×5 Matrix{Int64}:
 1  4  7  10  13
 2  5  8  11  14
 3  6  9  12  15

julia> NNlib.gather(src, [1, 2], [2, 4])
2-element Vector{Int64}:
  4
 11
```

source

```julia
NNlib.gather!(dst, src, idx; checkbounds = true)
```

Reverse operation of [`scatter!`](/api/NN_Primitives/NNlib#NNlib.scatter!). Gathers data from source `src` and writes it in destination `dst` according to the index array `idx`. For each `k` in `CartesianIndices(idx)`, assign values to `dst` according to

```julia
dst[:, ... , k] .= src[:, ... , idx[k]...]
```

Notice that if `idx` is a vector containing integers, and both `dst` and `src` are matrices, previous expression simplifies to

```julia
dst[:, k] .= src[:, idx[k]]
```

and `k` will run over `1:length(idx)`.

The elements of `idx` can be integers or integer tuples and may be repeated. A single `src` column can end up being copied into zero, one, or multiple `dst` columns.

With `checkbounds = false`, the GPU methods do not check that the indices are in range, see [`gather`](/api/NN_Primitives/NNlib#NNlib.gather).

See [`gather`](/api/NN_Primitives/NNlib#NNlib.gather) for an allocating version.

source

```julia
gather!(dst, src, IJK...; checkbounds = true)
```

Convert the tuple of integer vectors `IJK` to a tuple of `CartesianIndex` and call `gather!` on it: `gather!(dst, src, CartesianIndex.(IJK...))`.

**Examples**

```julia
julia> src = reshape([1:15;], 3, 5)
3×5 Matrix{Int64}:
 1  4  7  10  13
 2  5  8  11  14
 3  6  9  12  15

julia> dst = zeros(Int, 2);

julia> NNlib.gather!(dst, src, [1, 2], [2, 4])
2-element Vector{Int64}:
  4
 11
```

source

```julia
NNlib.scatter(op, src, idx; [init, dstsize, checkbounds])
```

Scatter operation allocating a destination array `dst` and calling `scatter!(op, dst, src, idx)` on it.

* If keyword `init` is provided, it is used to initialize the content of `dst`. Otherwise, the init values is inferred from the reduction operator `op` for some common operators (e.g. `init = 0` for `op = +`).

* If `dstsize` is provided, it will be used to define the size of destination array, otherwise it will be inferred by `src` and `idx`.

* `checkbounds = false` skips the check of the indices on a GPU, see [`scatter!`](/api/NN_Primitives/NNlib#NNlib.scatter!).

See [`scatter!`](/api/NN_Primitives/NNlib#NNlib.scatter!) for full details on how `idx` works.

**Examples**

```julia
julia> NNlib.scatter(+, [10,100,1000], [3,1,2])
3-element Vector{Int64}:
  100
 1000
   10

julia> NNlib.scatter(+, [1 2 3 4; 5 6 7 8], [2,1,1,5])
2×5 Matrix{Int64}:
  5  1  0  0  4
 13  5  0  0  8

julia> NNlib.scatter(*, [10,200,3000], [1,4,2]; init = 10, dstsize = 6)
6-element Vector{Int64}:
   100
 30000
    10
  2000
    10
    10
```

source

```julia
NNlib.scatter!(op, dst, src, idx; checkbounds = true)
```

Scatter operation, which writes data in `src` into `dst` at locations `idx`. A binary reduction operator `op` is applied during the scatter. For each index `k` in `idx`, accumulates values in `dst` according to

```julia
dst[:, ..., idx[k]...] = (op).(dst[:, ..., idx[k]...], src[:, ..., k...])
```

See also [`scatter`](/api/NN_Primitives/NNlib#NNlib.scatter), [`gather`](/api/NN_Primitives/NNlib#NNlib.gather).

**Arguments**

* `op`: Operations to be applied on `dst` and `src`, e.g. `+`, `-`, `*`, `/`, `max`, `min` and `mean`.

* `dst`: The destination for `src` to aggregate to. This argument will be mutated.

* `src`: The source data for aggregating.

* `idx`: The mapping for aggregation from source (index) to destination (value).        The `idx` array can contain either integers or tuples.

* `checkbounds`: On a GPU, checking that the indices are in range synchronizes the host        with the device. A caller that has checked them already can skip that with        `checkbounds = false`: an index out of range then gives an undefined result,        and may corrupt memory.

**Examples**

```julia
julia> NNlib.scatter!(+, ones(3), [10,100], [1,3])
3-element Vector{Float64}:
  11.0
   1.0
 101.0

julia> NNlib.scatter!(*, fill(0.5, 2, 4), [1 10; 100 1000], [3,2])
2×4 Matrix{Float64}:
 0.5    5.0   0.5  0.5
 0.5  500.0  50.0  0.5
```

source

## Sampling {#Sampling}

````julia
grid_sample(input::AbstractArray{T, 4}, grid::AbstractArray{T, 4}; padding_mode = :zeros)
grid_sample(input::AbstractArray{T, 5}, grid::AbstractArray{T, 4}; padding_mode = :zeros)

Given `input`, compute output by sampling `input` values at pixel
locations from `grid`. Uses bilinear interpolation to calculate output values.

This implementation assumes the extrema (`-1` and `1`) are considered
as referring to the center points of the input’s corner pixels
(i.e. align corners is `true`).

# Arguments

- `input`: Input array in `(W_in, H_in, [D_in,] C, N)` shape.
- `grid`: Input grid in `(2, W_out, H_out, [D_out,] N)` shape.
    Where for each `(W_out, H_out, [D_out,] N)` grid contains `(x, y [,z])`
    coordinates that specify sampling locations normalized by the `input` shape.

    Therefore, `x`, `y` and [`z`] should have values in `[-1, 1]` range.
    For example, `(x = -1, y = -1, [z = -1])` is the left-top[-front] pixel of `input`,
    and `(x = 1, y = 1, [z = 1])` is the right-bottom-back pixel of `input`.

    Out-of-bound values are handled according to the `padding_mode`.
- `padding_mode`: Out-of-bound padding.
    `:zeros` to use `0` for out-of-bound grid locations.
    `:border` to use border values for out-of-bound grid locations.
    Default is `:zeros`.

# Returns

`(W_out, H_out, [D_out,] C, N)` sampled grid from `input`.

# Examples

In the example below, grid contains two out-of-bound sampling locations,
which are handled differently, depending on the `padding_mode`.

```jldoctest
julia> x = reshape(collect(1.0:4.0), (2, 2, 1, 1))
2×2×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
1.0  3.0
2.0  4.0

julia> grid = Array{Float64}(undef, 2, 3, 2, 1);

julia> grid[:, 1, 1, 1] .= (-3, -1);

julia> grid[:, 2, 1, 1] .= (0, -1);

julia> grid[:, 3, 1, 1] .= (1, -1);

julia> grid[:, 1, 2, 1] .= (-1, 1);

julia> grid[:, 2, 2, 1] .= (0, 1);

julia> grid[:, 3, 2, 1] .= (3, 1);

julia> grid_sample(x, grid; padding_mode=:zeros)
3×2×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
0.0  3.0
1.5  3.5
2.0  0.0

julia> grid_sample(x, grid; padding_mode=:border)
3×2×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
1.0  3.0
1.5  3.5
2.0  4.0
```
````

source

```julia
∇grid_sample(Δ::AbstractArray{T, 4}, input::AbstractArray{T, 4}, grid::AbstractArray{T, 4}; padding_mode = :zeros) where T
```

**Arguments**

* `Δ`: Input gradient in `(W_out, H_out, C, N)` shape   (same as output of the primal computation).

* `input`: Input from primal computation in `(W_in, H_in, C, N)` shape.

* `grid`: Grid from primal computation in `(2, W_out, H_out, N)` shape.

* `padding_mode`: Out-of-bound padding.   `:zeros` to use `0` for out-of-bound grid locations.   `:border` to use border values for out-of-bound grid locations.   Should be the same as in primal computation.   Default is `:zeros`.

**Returns**

`dinput` (same shape as `input`) and `dgrid` (same shape as `grid`) gradients.

source

## Losses {#Losses}

```julia
ctc_loss(ŷ, y)
```

Computes the connectionist temporal classification loss between `ŷ` and `y`. `ŷ` must be a classes-by-time matrices, i.e., each row represents a class and each column represents a time step. Additionally, the `logsoftmax` function will be applied to `ŷ`, so `ŷ` must be the raw activation values from the neural network and not, for example, the activations after being passed through a `softmax` activation function. `y` must be a 1D array of the labels associated with `ŷ`. The blank label is assumed to be the last label category in `ŷ`, so it is equivalent to `size(ŷ, 1)`. Used for sequence-to-sequence classification problems such as speech recognition and handwriting recognition where the exact time-alignment of the output (e.g., letters) is not needed to solve the problem. See [Graves et al. (2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) or [Graves (2012)](https://www.cs.toronto.edu/~graves/preprint.pdf#chapter.7) for mathematical details.

source

## Miscellaneous {#Miscellaneous}

```julia
logsumexp(x; dims = :)
```

Computes `log.(sum(exp.(x); dims))` in a numerically stable way. Without `dims` keyword this returns a scalar. For `Float16` and `BFloat16` inputs, it is computed and returned in `Float32`.

See also [`logsoftmax`](/api/NN_Primitives/NNlib#NNlib.logsoftmax).

source

```julia
glu(x, dim = 1)
```

The gated linear unit from the ["Language Modeling with Gated Convolutional Networks"](https://arxiv.org/abs/1612.08083) paper.

Calculates `a .* sigmoid(b)`, where `x` is split in half along given dimension `dim` to form `a` and `b`.

See also [`swiglu`](/api/NN_Primitives/NNlib#NNlib.swiglu).

source

```julia
swiglu(x, dim = 1)
```

The swish-gated linear unit from the ["GLU Variants Improve Transformer"](https://arxiv.org/abs/2002.05202) paper, the feed-forward activation of Llama, PaLM and most language models since.

Calculates `swish.(gate) .* up`, where `x` is split in half along given dimension `dim` to form `gate` (the first half) and `up` (the second half). This is the layout of the fused gate and up projections of Llama-style models, and the reverse of [`glu`](/api/NN_Primitives/NNlib#NNlib.glu), where the gate is the second half.

See also [`swish`](/api/NN_Primitives/ActivationFunctions#NNlib.swish).

**Examples**

```julia
ffn = Chain(Dense(d => 2h), NNlib.swiglu, Dense(h => d))
```

source

```julia
@disallow_spawns ex
```

Disallow NNlib to use `@spawn` on divisible workloads. i.e. within `conv` etc.

source

::: tip Tip

`within_gradient` function currently doesn't work for Enzyme. Prefer to use `LuxLib.Utils.within_autodiff` if needed. Though pay heed that this function is not part of the public API.

:::

```julia
within_gradient(x) --> Bool
```

Returns `false` except when used inside a `gradient` call, when it returns `true`. Useful for Flux regularisation layers which behave differently during training and inference.

This should work with any ChainRules-based differentiation package, in which case `x` is ignored. But Tracker.jl overloads `within_gradient(x::TrackedArray)`, thus for widest use you should pass it an array whose gradient is of interest. There is also an overload for ForwardDiff.jl's `Dual` types (and arrays of them). Enzyme.jl and Mooncake.jl are supported through package extensions, and ignore `x` too.

**Examples**

```julia
julia> using ForwardDiff, Zygote, NNlib

julia> f_good(x) = if NNlib.within_gradient(x)
                     @show 10x
                   else
                     x
                   end;

julia> Zygote.withgradient(f_good, 1.0)
10x = 10.0
(val = 10.0, grad = (10.0,))

julia> ForwardDiff.derivative(f_good, 1.0)
10x = Dual{ForwardDiff.Tag{typeof(f_good), Float64}}(10.0,10.0)
10.0

julia> f_bad(x, y) = if any(NNlib.within_gradient, (x, y))
                       @show x * y
                     else
                       x / y
                     end;

julia> Zygote.withgradient(f_bad, 2.0, 3.0)
(val = 0.6666666666666666, grad = (0.3333333333333333, -0.2222222222222222))

julia> ForwardDiff.derivative(x -> f_bad(x, 3.0), 2.0)
x * y = Dual{ForwardDiff.Tag{var"#9#10", Float64}}(6.0,3.0)
3.0
```

What goes wrong in `f_bad` is that Zygote knows `any` to be non-differentiable, and thus completely ignores its contents. This is not a perfect mechanism, and the only style recommended is precisely that of `f_good` above.

source

::: tip Tip

Use `LuxLib.API.bias_activation!!` or `LuxLib.API.bias_activation` instead of `NNlib.bias_act!`.

:::

```julia
bias_act!(σ, x, b)
```

This is equivalent to `x .= σ.(x .+ b)`, also replacing `sigmoid` & `tanh` with `sigmoid_fast` & `tanh_fast`. It will only overwrite `x` when `x isa StridedArray{<:AbstractFloat}`.

When used within a gradient, it will overwrite only when `σ` has a method of `derivatives_given_output` which does not need the input at all. Such methods are defined by e.g. `@scalar_rule relu(x) Ω > 0` where the derivative contains only `Ω` (the output) not `x`.

::: warning Warning

This is not safe to use if `x` is still needed for the gradient of some other function. Incorrect use will give silently wrong answers. It is intended mainly for Flux layers, in which the previous operation is known to be safe, e.g. `bias_act!(σ, weight * input, bias)` for a `Dense` layer.

:::

source

## Dropout {#Dropout}

::: tip Tip

Use `LuxLib.API.dropout` instead of `NNlib.dropout`.

:::

```julia
dropout([rng], A, p; [dims])
```

Returns an array in which each element of `A` is either replaced with zero, with probability `p`, or else multiplied by `1/(1-p)`.

By default every element is treated independently. With keyword `dims=1`, a choice is made for every value of the 1st index i.e. each row of a matrix is either zero or not.

Optional first argument is the random number generator used. Inside a Reactant-compiled function, pass one (e.g. a `Reactant.ReactantRNG`) to get a new mask on every call, since `default_rng()` there is seeded once at compile time.

**Examples**

```julia
julia> dropout(ones(2, 10), 0.2)
2×10 Matrix{Float64}:
 1.25  1.25  0.0   1.25  1.25  1.25  1.25  1.25  1.25  1.25
 1.25  1.25  1.25  0.0   1.25  1.25  0.0   1.25  1.25  1.25

julia> mean(dropout(ones(10^4, 5), 0.2), dims=1)
1×5 Matrix{Float64}:
 0.998  1.00075  0.99125  0.99575  1.00075

julia> dropout(ones(5, 5), 0.7, dims=1)  # whole row the same
5×5 Matrix{Float64}:
 3.33333  3.33333  3.33333  3.33333  3.33333
 0.0      0.0      0.0      0.0      0.0
 0.0      0.0      0.0      0.0      0.0
 3.33333  3.33333  3.33333  3.33333  3.33333
 0.0      0.0      0.0      0.0      0.0

julia> mean(dropout(ones(10^4, 5), 0.3, dims=1), dims=1)
1×5 Matrix{Float64}:
 1.00571  1.00571  1.00571  1.00571  1.00571
```

source

```julia
dropout!(B, A, p; [dims])
```

This does exactly `B .= dropout(A, p; dims)`, or rather, it's the implementation of out-of-place [`dropout`](/api/NN_Primitives/NNlib#NNlib.dropout).

source

## Normalization {#Normalization}

```julia
normalise(x; dims=ndims(x), eps=1f-5)
```

Normalise `x` to zero mean and unit standard deviation across the dimension(s) given by `dims`. Per default, `dims` is the last dimension. `eps` is a small term added to the variance for numerical stability.

It is [`layernorm`](/api/NN_Primitives/NNlib#NNlib.layernorm) without scale and bias. For half-precision (`Float16`/`BFloat16`) inputs the statistics are computed in `Float32`.

**Examples**

```julia
julia> using Statistics

julia> x = [90, 100, 110, 130, 70];

julia> y = NNlib.normalise(x);

julia> isapprox(std(y; corrected=false), 1, atol=1e-5)
true
```

source

```julia
batchnorm(g, b, x, running_mean=nothing, running_var=nothing, momentum=0.1f0;
          eps=1f-5, training=true, track_stats=true)
```

Functional [batch normalization](https://arxiv.org/abs/1502.03167). `g` and `b` are the per-channel scale and bias (either may be `nothing` for no affine transform); `x` are the feature maps. For an input with `N` dimensions the `N-1`th is the channel dimension (the usual convention for `WHCN` images); statistics are computed over every `D_1×…×D_{N-2}×1×D_N` slice, so per channel.

If `running_mean`/`running_var` are supplied:

* with `training=true` (default) they are updated **in place** (when `track_stats=true`) with an exponential moving average controlled by `momentum`, while statistics of the current batch are used to normalise;

* with `training=false` they are used to normalise (inference / test mode).

`eps` is added to the variance for numerical stability. On the GPU, 3D/4D/5D `CuArray`s are dispatched to the cuDNN implementation.

For half-precision (`Float16`/`BFloat16`) feature maps the statistics are computed in `Float32` and the result is cast back; `g`, `b`, `running_mean` and `running_var` must then be `Float32`.

See also [`instancenorm`](/api/NN_Primitives/NNlib#NNlib.instancenorm), [`groupnorm`](/api/NN_Primitives/NNlib#NNlib.groupnorm), [`layernorm`](/api/NN_Primitives/NNlib#NNlib.layernorm).

source

```julia
∇batchnorm(g, b, x, dy, running_mean=nothing, running_var=nothing, momentum=0.1f0;
           eps=1f-5, training=true)
```

Gradient of [`batchnorm`](/api/NN_Primitives/NNlib#NNlib.batchnorm): given the upstream gradient `dy`, return `(dg, db, dx)`. On the GPU, 3D/4D/5D `CuArray`s are dispatched to the cuDNN implementation in `NNlibCUDACUDNNExt`; this generic method is the fallback used everywhere else and is itself differentiable (second-order).

source

```julia
instancenorm(g, b, x, running_mean=nothing, running_var=nothing, momentum=0.1f0;
             eps=1f-5, training=true, track_stats=false)
```

Functional [instance normalization](https://arxiv.org/abs/1607.08022). Arguments match [`batchnorm`](/api/NN_Primitives/NNlib#NNlib.batchnorm), but for an input with `N > 2` dimensions statistics are computed over every `D_1×…×D_{N-2}×1×1` slice, so per channel **and** per sample in the batch. When tracked, the running statistics (length `size(x, N-1)`) accumulate the per-channel average across the batch.

Normalising a dense CPU or GPU array with its own statistics, with `g` and `b` of length `size(x, N-1)`, runs fused kernels; other cases use the generic code.

See also [`batchnorm`](/api/NN_Primitives/NNlib#NNlib.batchnorm), [`groupnorm`](/api/NN_Primitives/NNlib#NNlib.groupnorm), [`layernorm`](/api/NN_Primitives/NNlib#NNlib.layernorm).

source

```julia
∇instancenorm(g, b, x, dy, running_mean=nothing, running_var=nothing, momentum=0.1f0;
              eps=1f-5, training=true)
```

Gradient of [`instancenorm`](/api/NN_Primitives/NNlib#NNlib.instancenorm), returning `(dg, db, dx)`.

source

```julia
groupnorm(g, b, x, G::Integer; eps=1f-5)
```

Functional [group normalization](https://arxiv.org/abs/1803.08494). `g` and `b` are the per-channel scale and bias (either may be `nothing`); `x` are the feature maps. For an input with `N > 2` dimensions the `N-1`th is the channel dimension; its `C = size(x, N-1)` channels are split into `G` groups (`G` must divide `C`) and statistics are computed over each group together with the spatial dimensions, per sample. `eps` is added to the variance.

Normalising a dense CPU or GPU array, with `g` and `b` of length `C`, runs fused kernels; other cases use the generic code.

See also [`batchnorm`](/api/NN_Primitives/NNlib#NNlib.batchnorm), [`instancenorm`](/api/NN_Primitives/NNlib#NNlib.instancenorm), [`layernorm`](/api/NN_Primitives/NNlib#NNlib.layernorm).

source

```julia
∇groupnorm(g, b, x, dy, G::Integer; eps=1f-5)
```

Gradient of [`groupnorm`](/api/NN_Primitives/NNlib#NNlib.groupnorm), returning `(dg, db, dx)`.

source

```julia
layernorm(g, b, x; dims=1, eps=1f-5)
```

Functional [layer normalization](https://arxiv.org/abs/1607.06450). `g` and `b` are the scale and bias (either may be `nothing`); `x` is normalised over the dimensions `dims` (the leading dimension by default).

`g` and `b`, when given, must broadcast against the normalised region, e.g. have size `size(x)[dims]` with singleton trailing dimensions.

Normalising a dense CPU or GPU array over leading `dims` (`1:k`), with `g` and `b` of the shape of those dimensions, runs fused kernels; other cases use the generic code.

See also [`normalise`](/api/NN_Primitives/NNlib#NNlib.normalise), [`batchnorm`](/api/NN_Primitives/NNlib#NNlib.batchnorm), [`instancenorm`](/api/NN_Primitives/NNlib#NNlib.instancenorm), [`groupnorm`](/api/NN_Primitives/NNlib#NNlib.groupnorm).

source

```julia
∇layernorm(g, b, x, dy; dims=1, eps=1f-5)
```

Gradient of [`layernorm`](/api/NN_Primitives/NNlib#NNlib.layernorm), returning `(dg, db, dx)`.

source

```julia
rmsnorm(g, x; dims=1, eps=1f-5)
```

Functional [root mean square normalization](https://arxiv.org/abs/1910.07467). `x` is divided by its root mean square over the dimensions `dims` (the leading dimension by default) and multiplied by the scale `g` (`nothing` for no scaling):

```julia
y = g .* x ./ sqrt.(mean(abs2, x; dims) .+ eps)
```

Unlike [`layernorm`](/api/NN_Primitives/NNlib#NNlib.layernorm), the mean is not subtracted and there is no bias.

`g`, when given, must broadcast against the normalised region, e.g. have size `size(x)[dims]` with singleton trailing dimensions. `eps` is added to the mean square for numerical stability.

For half-precision (`Float16`/`BFloat16`) inputs the mean square is computed in `Float32` and the result is cast back; `g` must then be `Float32`.

See also [`layernorm`](/api/NN_Primitives/NNlib#NNlib.layernorm), [`normalise`](/api/NN_Primitives/NNlib#NNlib.normalise).

**Examples**

```julia
julia> x = Float32[3 0; 4 2];

julia> NNlib.rmsnorm(nothing, x) ≈ x ./ sqrt.(Float32[12.5 2])
true

julia> NNlib.rmsnorm(Float32[2, 1], x) ≈ Float32[2, 1] .* NNlib.rmsnorm(nothing, x)
true
```

source

```julia
∇rmsnorm(g, x, dy; dims=1, eps=1f-5)
```

Gradient of [`rmsnorm`](/api/NN_Primitives/NNlib#NNlib.rmsnorm), returning `(dg, dx)`.

source

## Internal NNlib Functions {#Internal-NNlib-Functions}

These functions are not part of the public API and are subject to change without notice.

```julia
batched_transpose(A::AbstractArray{T,3})
batched_adjoint(A)
```

Equivalent to applying `transpose` or `adjoint` to each matrix `A[:,:,k]`.

These exist to control how `batched_mul` behaves, as it operates on such matrix slices of an array with `ndims(A)==3`.

`PermutedDimsArray(A, (2,1,3))` is equivalent to `batched_transpose(A)`, and is also understood by `batched_mul` (and more widely supported elsewhere).

```julia
BatchedTranspose{T, S} <: AbstractBatchedMatrix{T, 3}
BatchedAdjoint{T, S}
```

Lazy wrappers analogous to `Transpose` and `Adjoint`, returned by `batched_transpose` etc.

source

```julia
∇conv_filter_direct!(dw, x, dy, cdims; alpha=1, beta=0)
```

Calculate the gradient imposed upon `w` in the convolution `y = x * w`.

source

```julia
_check_trivial_rotations!(out, arr, θ, rotation_center)
```

When `θ = 0 || π /2 || π || 3/2 || π` and if `rotation_center`  is in the middle of the array. For an even array of size 4, the rotation_center would need to be 2.5. For an odd array of size 5, the rotation_center would need to be 3.

In those cases, rotations are trivial just by reversing or swapping some axes.

source

```julia
NNlib.fast_act(f, [x::AbstractArray])
```

Replaces `f == tanh` with [`tanh_fast`](/api/NN_Primitives/ActivationFunctions#NNlib.tanh_fast), etc.

Takes an optional 2nd argument, so that you can disable this replacement for some array or element types.

source

```julia
spectrogram(waveform;
    pad::Int = 0, n_fft::Int, hop_length::Int, window,
    center::Bool = true, power::Real = 2.0,
    normalized::Bool = false, window_normalized::Bool = false,
)
```

Create a spectrogram or a batch of spectrograms from a raw audio signal.

**Arguments**

* `pad::Int`:   Then amount of padding to apply on both sides.

* `window_normalized::Bool`:   Whether to normalize the waveform by the window’s L2 energy.

* `power::Real`:   Exponent for the magnitude spectrogram (must be ≥ 0)   e.g., `1` for magnitude, `2` for power, etc.   If `0`, complex spectrum is returned instead.

See [`stft`](/api/NN_Primitives/NNlib#NNlib.stft) for other arguments.

**Returns**

Spectrogram in the shape `(T, F, B)`, where `T` is the number of window hops and `F = n_fft ÷ 2 + 1`.

source

```julia
is_strided(A::AbstractArray) -> Bool
```

This generalises `A isa StridedArray` to treat wrappers like `A::PermutedDimsArray`, for which it returns `is_strided(parent(A))`.

It returns `true` for `CuArray`s, and `PermutedDimsArray`s of those.

Other wrappers (defined outside Base, LinearAlgebra) are assumed not to break strided-ness, and hence also return `is_strided(parent(A))`. This correctly handles things like `NamedDimsArray` wihch don't alter indexing. However, it's a little pessimistic in that e.g. a `view` of such a container will return `false`, even in cases where the same `view` of `parent(A)` would be a `StridedArray`.

source

```julia
conv_direct!(y, x, w, cdims; alpha=1, beta=0)
```

Direct convolution implementation; used for debugging, tests, and mixing/matching of strange datatypes within a single convolution.  Uses naive nested for loop implementation and does not attempt to optimize performance.  Rather, this implementation is intended to be maximally understandable and debuggable, to aid in testing other, more performant implementations.  We also explicitly support mixing and matching of strange datatypes, so that if the user really wants to convolve an image of `UInt8`'s with a `Float16` kernel, storing the result in a `Float32` output, there is at least a function call for that madness.

The keyword arguments `alpha` and `beta` control accumulation behavior; this function calculates `y = alpha * x * w + beta * y`, therefore by setting `beta` to a nonzero value, the user is able to accumulate values into a preallocated `y` buffer, or by setting `alpha` to a nonunitary value, an arbitrary gain factor can be applied.

By defaulting `beta` to `false`, we make use of the Bradbury promotion trick to override `NaN`'s that may pre-exist within our output buffer, as `false*NaN == 0.0`, whereas `0.0*NaN == NaN`.  Only set `beta` if you are certain that none of the elements within `y` are `NaN`.

The basic implementation performs 3-dimensional convolution; 1-dimensional and 2- dimensional cases are supported by simply reshaping `y`, `x` and `w`, for which wrapper methods are available.

source

```julia
gemm!()
```

Low-level gemm!() call with pointers, borrowed from Knet.jl

Calculates `C = alpha*op(A)*op(B) + beta*C`, where:

* `transA` and `transB` set `op(X)` to be either `identity()` or `transpose()`

* alpha and beta are scalars

* op(A) is an (M, K) matrix

* op(B) is a (K, N) matrix

* C is an (M, N) matrix.

source

```julia
calc_padding_regions(dims)
```

Padding is a jerk.  A HUGE jerk that tries to sneak a bunch of conditionals and edge cases (quite literally) into our beautiful stencil operations such as convolution, pooling, etc...  The way we deal with this is to, first, deal with everything in 3d, and then define a single padding region helper function that returns the seven regions that all 3d operations must deal with, including the central "unpadded" region where we can run at full bore, not paying any attention to padding.

source

```julia
_prepare_imrotate(arr, θ, rotation_center)
```

Prepate `sin` and `cos`, creates the output array and converts type of `rotation_center` if required.

source

```julia
insert_singleton_spatial_dimension(cdims::ConvDims)
```

When converting a 1d convolution to a 2d, or a 2d to a 3d, we need to insert a singleton spatial dimension at the end of the spatial dimensions.  This does so for a ConvDims.

source

```julia
_fast_broadcast!(f, x, y, z...)
```

This does `x .= f.(x, y, z...)`, but works around an issue with broadcasting that prevents SIMD in such cases. Can perhaps be removed once https://github.com/JuliaLang/julia/issues/43153 is fixed.

Has an `rrule` to avoid mutation within derivatives.

::: warning Warning

Not intended for general use. Uses `@inbounds` but does not check sizes! Assumes that `f` has no derivative!

:::

source

```julia
hann_window(
    window_length::Int, ::Type{T} = Float32; periodic::Bool = true,
) where T <: Real
```

Hann window function (ref: [Window function § Hann and Hamming windows - Wikipedia](https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows)).

$w\[n] = \frac{1}{2}\[1 - \cos(\frac{2 \pi n}{N - 1})]$

Where $N$ is the window length.

```julia
julia> lineplot(hann_window(100); width=30, height=10)
     ┌──────────────────────────────┐
   1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠚⠉⠉⠉⠢⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡔⠁⠀⠀⠀⠀⠀⠘⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠞⠀⠀⠀⠀⠀⠀⠀⠀⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⢀⡎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢣⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⡎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢦⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⢀⠞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢆⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⢀⡜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢇⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢦⠀⠀⠀⠀│
     │⠀⠀⠀⢠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠣⡀⠀⠀│
   0 │⣀⣀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢤⣀│
     └──────────────────────────────┘
     ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀100⠀
```

**Arguments:**

* `window_length::Int`: Size of the window.

* `::Type{T}`: Elemet type of the window.

**Keyword Arguments:**

* `periodic::Bool`: If `true` (default), returns a window to be used as   periodic function. If `false`, return a symmetric window.
  Following always holds:

```julia
julia> N = 256;

julia> hann_window(N; periodic=true) ≈ hann_window(N + 1; periodic=false)[1:end - 1]
true

julia> hann_window(N) ≈ hamming_window(N; α=0.5f0, β=0.5f0)
true
```

**Returns:**

Vector of length `window_length` and eltype `T`.

source

```julia
_rng_from_array(x)
```

Return the random number generator most appropriate for `x`: `CUDA.default_rng()` for `CuArray`, else `Random.default_rng()`

source

```julia
istft(y;
    n_fft::Int, hop_length::Int = n_fft ÷ 4, window = nothing,
    center::Bool = true, normalized::Bool = false,
    return_complex::Bool = false,
    original_length::Union{Nothing, Int} = nothing,
)
```

Inverse Short-time Fourier Transform.

Return the least squares estimation of the original signal (D. Griffin and J. Lim, "Signal estimation from modified short-time Fourier transform", 1984): the inverse transforms of the frames are multiplied by the window, overlap-added, and divided by the overlap-added squared window. Samples where the latter vanishes (e.g. the first one for `center=false` and a `hann_window`) cannot be recovered and are set to zero.

**Arguments:**

* `y`: Input complex array in the `(n_fft, n_frames, B)` shape.   Where `B` is the optional batch dimension.

**Keyword Arguments:**

* `n_fft::Int`: Size of Fourier transform.

* `hop_length::Int`: Distance between neighboring sliding window frames.

* `window`: Window function that was applied to the input of `stft`.   If `nothing` (default), then no window was applied.

* `center::Bool`: Whether input to `stft` was padded on both sides   so that $t$-th frame is centered at time $t \times \text{hop length}$.   Padding is done with `pad_reflect` function.

* `normalized::Bool`: Whether input to `stft` was normalized.

* `return_complex::Bool`: Whether the output should be complex,   or if the input should be assumed to derive from a real signal and window.

* `original_length::Union{Nothing, Int}`: Optional size of the first dimension   of the input to `stft`. Helps restoring the exact `stft` input size.   Otherwise, the array might be a bit shorter.

source

```julia
transpose_swapbatch(x::AbstractArray)
```

Given an AbstractArray, swap its batch and channel axes, as we must during transposed convolution.  We do this to the operands during convolution, and then again to the output once we're done.

source

```julia
transpose_pad(cdims::ConvDims)
```

Transposed convolution can be calculated in terms of typical convolution with some extra padding.  This method computes the padding of the convolution that would result in the transposed convolution of two operands, in essence taking care of that "extra padding". Note that this method should almost always be accompanied by a call that predilates one of the operands.

source

```julia
power_to_db(s; ref::Real = 1f0, amin::Real = 1f-10, top_db::Real = 80f0)
```

Convert a power spectrogram (amplitude squared) to decibel (dB) units.

**Arguments**

* `s`: Input power.

* `ref`: Scalar w.r.t. which the input is scaled.

* `amin`: Minimum threshold for `s`.

* `top_db`: Threshold the output at `top_db` below the peak:   `max.(s_db, maximum(s_db) - top_db)`.

**Returns**

`s_db ~= 10 * log10(s) - 10 * log10(ref)`

source

```julia
col2im!(x, col, cdims, beta=0, rows=1:M)
```

Does the inverse of `im2col!()`, converting `col` back into a 3d image, used for backward passes, transposed convolutions, etc...  `x` is first scaled by `beta`, then the output pixels `rows`, held in the first `length(rows)` rows of `col`, are added into it.

Note that this method has not been optimized in the same way as `im2col()` has, because it is slightly more complicated due to the more chaotic data access patterns, and I'm not desperate enough yet.

source

```julia
storage_type(A) -> Type
```

Removes all wrappers to return the `Array` or `CuArray` (or whatever) type within.

```julia
julia> view(reshape(ones(10)',2,5),:, 3:4) |> storage_type
Array{Float64,1}

julia> reshape(sparse(rand(10)), 5,2) |> storage_type
SparseVector{Float64,Int64}
```

source

```julia
im2col_dims(c::ConvDims)
```

im2col calculates, for each output pixel, the "convolution" of N kernels where N is the number of output channels, by doing a matrix multiply.  This function gives the dimensions of that matrix, with one copy per thread along the third.

The im2col convolutions only hold a tile of this matrix per task, so an array of these dimensions is a valid, but oversized, `col` workspace for them.

source

```julia
reverse_indices(idx)
```

Return the reverse indices of `idx`. The indices of `idx` will be values, and values of `idx` will be index.

**Arguments**

* `idx`: The indices to be reversed. Accepts array or cuarray of integer, tuple or `CartesianIndex`.

source

```julia
∇conv_filter_im2col!(dw, x, dy, cdims; col=nothing, alpha=1, beta=0, ntasks=nthreads())
```

Conv backward pass onto the weights using im2col and GEMM; stores the result in `dw`. See [`conv_im2col!`](/api/NN_Primitives/NNlib#NNlib.conv_im2col!) for explanation of optional parameters.

source

```julia
conv_im2col!(y, x, w, cdims; col=nothing, alpha=1, beta=0, ntasks=nthreads())
```

Perform a convolution using im2col and GEMM, store the result in `y`.  The  kwargs `alpha` and `beta` control accumulation behavior; internally this operation is implemented as a matrix multiply that boils down to `y = alpha * x * w + beta * y`, thus by setting `beta` to a nonzero value, multiple results can be accumulated into `y`, or by setting `alpha` to a nonunitary value, various gain factors can be applied.

Note for the particularly performance-minded, you can provide a pre-allocated workspace `col` of size `(rows, K, ntasks)`, see `conv_im2col_dims`; each of up to `size(col, 3)` tasks processes the output positions `rows` at a time.

source

```julia
∇conv_data_direct!(dx, dy, w, cdims; alpha=1, beta=0)
```

Calculate the gradient imposed upon `x` in the convolution `y = x * w`.

source

Performs dimensional consistency checks and return the dimensionality of the scattered objects.

source

```julia
∇conv_data_im2col!(dx, w, dy, cdims; col=nothing, alpha=1, beta=0, ntasks=nthreads())
```

Conv2d backward pass onto the input using im2col and GEMM; stores the result in `dx`. See [`conv_im2col!`](/api/NN_Primitives/NNlib#NNlib.conv_im2col!) for explanation of optional parameters.

source

```julia
storage_typejoin(A, B, C, ...) -> Type
```

Reduces with `Base.promote_typejoin`, in order that this conveys useful information for dispatching to BLAS. It does not tell you what container to allocate:

```julia
julia> storage_typejoin(rand(2), rand(Float32, 2))
Array{T,1} where T

julia> eltype(ans) <: LinearAlgebra.BlasFloat
false

julia> storage_typejoin(rand(2), rand(2,3), rand(2,3,4))
Array{Float64,N} where N
```

source

```julia
add_blanks(z)
```

Adds blanks to the start and end of `z`, and between items in `z`

source

```julia
∇filter_im2col_dims(c::ConvDims)
```

Like [`im2col_dims`](/api/NN_Primitives/NNlib#NNlib.im2col_dims), for a single thread.

Note: in the future, this may return `Dims{2}` instead of `Dims{3}`.

source

\_bilinear_helper(yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int)

Some helper variables

source

```julia
_triangular_filterbanks(
    freq_points::Vector{Float32}, all_freqs::Vector{Float32})
```

Create triangular filter banks.

**Arguments:**

* `freq_points::Vector{Float32}`: Filter midpoints of size `n_filters`.

* `all_freqs::Vector{Float32}`: Frequency points of size `n_freqs`.

**Returns:**

Array of size `(n_freqs, n_filters)`.

source

```julia
db_to_power(s_db; ref::Real = 1f0)
```

Inverse of [`power_to_db`](/api/NN_Primitives/NNlib#NNlib.power_to_db).

source

```julia
predilated_size(x_size::Tuple, dilation::Tuple)
```

Calculate the size of a predilated `x` given a particular dilation factor.  This is used within `predilate()` and `transpose_cdims()`.

source

```julia
stft(x;
    n_fft::Int, hop_length::Int = n_fft ÷ 4, window = nothing,
    center::Bool = true, normalized::Bool = false,
)
```

Short-time Fourier transform (STFT).

The STFT computes the Fourier transform of short overlapping windows of the input, giving frequency components of the signal as they change over time.

$Y\[\omega, m] = \sum\_{k = 0}^{N - 1} \text{window}\[k] \text{input}\[m \times \text{hop length} + k] \exp(-j \frac{2 \pi \omega k}{\text{n fft}})$

where $N$ is the window length, $\omega$ is the frequency $0 \le \omega < \text{n fft}$ and $m$ is the index of the sliding window.

**Arguments:**

* `x`: Input, must be either a 1D time sequence (`(L,)` shape)   or a 2D batch of time sequence (`(L, B)` shape).

**Keyword Arguments:**

* `n_fft::Int`: Size of Fourier transform.

* `hop_length::Int`: Distance between neighboring sliding window frames.

* `window`: Optional window function to apply.   Must be 1D vector `0 < length(window) ≤ n_fft`.   If window is shorter than `n_fft`, it is padded with zeros on both sides.   If `nothing` (default), then no window is applied.

* `center::Bool`: Whether to pad input on both sides so that $t$-th frame   is centered at time $t \times \text{hop length}$.   Padding is done with `pad_reflect` function.

* `normalized::Bool`: Whether to return normalized STFT,   i.e. multiplied with $\text{n fft}^{-0.5}$.

**Returns:**

Complex array of shape `(n_fft, n_frames, B)`, where `B` is the optional batch dimension.

source

```julia
hamming_window(
    window_length::Int, ::Type{T} = Float32; periodic::Bool = true,
    α::T = T(0.54), β::T = T(0.46),
) where T <: Real
```

Hamming window function (ref: [Window function § Hann and Hamming windows - Wikipedia](https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows)). Generalized version of `hann_window`.

$w\[n] = \alpha - \beta \cos(\frac{2 \pi n}{N - 1})$

Where $N$ is the window length.

```julia
julia> lineplot(hamming_window(100); width=30, height=10)
     ┌──────────────────────────────┐
   1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠚⠉⠉⠉⠢⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠎⠁⠀⠀⠀⠀⠀⠈⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣⡀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⢰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⣠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠳⡀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⢰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡄⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⡰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⠀│
     │⠀⠀⠀⢀⠴⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢄⠀⠀⠀│
     │⠀⢀⡠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠳⣀⠀│
   0 │⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉│
     └──────────────────────────────┘
     ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀100⠀
```

**Arguments:**

* `window_length::Int`: Size of the window.

* `::Type{T}`: Elemet type of the window.

**Keyword Arguments:**

* `periodic::Bool`: If `true` (default), returns a window to be used as   periodic function. If `false`, return a symmetric window.
  Following always holds:

```julia
julia> N = 256;

julia> hamming_window(N; periodic=true) ≈ hamming_window(N + 1; periodic=false)[1:end - 1]
true
```

* `α::Real`: Coefficient α in the equation above.

* `β::Real`: Coefficient β in the equation above.

**Returns:**

Vector of length `window_length` and eltype `T`.

source

```julia
maximum_dims(dims)
```

Given an array of `CartesianIndex{N}` or `NTuple{N,Int}`, returns a tuple containing the maximum of all the 1st entries, all the 2nd entries, and so on up to `N`.

Given an array of integers, returns `(maximum(dims),)`.

(These arguments are what [`scatter`](/api/NN_Primitives/NNlib#NNlib.scatter) understands.)

source

```julia
batched_transpose(A::AbstractArray{T,3})
batched_adjoint(A)
```

Equivalent to applying `transpose` or `adjoint` to each matrix `A[:,:,k]`.

These exist to control how `batched_mul` behaves, as it operates on such matrix slices of an array with `ndims(A)==3`.

`PermutedDimsArray(A, (2,1,3))` is equivalent to `batched_transpose(A)`, and is also understood by `batched_mul` (and more widely supported elsewhere).

```julia
BatchedTranspose{T, S} <: AbstractBatchedMatrix{T, 3}
BatchedAdjoint{T, S}
```

Lazy wrappers analogous to `Transpose` and `Adjoint`, returned by `batched_transpose` etc.

source

```julia
_rotate_coordinates(sinθ, cosθ, i, j, rotation_center, round_or_floor)
```

This rotates the coordinates and either applies round(nearest neighbour) or floor for :bilinear interpolation)

source

```julia
melscale_filterbanks(;
    n_freqs::Int, n_mels::Int, sample_rate::Int,
    fmin::Float32 = 0f0, fmax::Float32 = Float32(sample_rate ÷ 2))
```

Create triangular Mel scale filter banks (ref: [Mel scale - Wikipedia](https://en.wikipedia.org/wiki/Mel_scale)). Each column is a filterbank that highlights its own frequency.

**Arguments:**

* `n_freqs::Int`: Number of frequencies to highlight.

* `n_mels::Int`: Number of mel filterbanks.

* `sample_rate::Int`: Sample rate of the audio waveform.

* `fmin::Float32`: Minimum frequency in Hz.

* `fmax::Float32`: Maximum frequency in Hz.

**Returns:**

Filterbank matrix of shape `(n_freqs, n_mels)` where each column is a filterbank.

```julia
julia> n_mels = 8;

julia> fb = melscale_filterbanks(; n_freqs=200, n_mels, sample_rate=16000);

julia> plot = lineplot(fb[:, 1]);

julia> for i in 2:n_mels
           lineplot!(plot, fb[:, i])
       end

julia> plot
     ┌────────────────────────────────────────┐
   1 │⠀⡀⢸⠀⢸⠀⠀⣧⠀⠀⢸⡄⠀⠀⠀⣷⠀⠀⠀⠀⠀⣷⠀⠀⠀⠀⠀⠀⢀⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⡇⢸⡆⢸⡇⠀⣿⠀⠀⡜⡇⠀⠀⢰⠋⡆⠀⠀⠀⢰⠁⡇⠀⠀⠀⠀⠀⡸⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⣿⢸⡇⡇⡇⢰⠹⡄⠀⡇⢱⠀⠀⢸⠀⢣⠀⠀⠀⡜⠀⢸⡀⠀⠀⠀⢀⠇⠀⠈⡇⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⣿⡇⡇⡇⡇⢸⠀⡇⢀⠇⠸⡀⠀⡇⠀⠸⡀⠀⢀⠇⠀⠀⢇⠀⠀⠀⡸⠀⠀⠀⠸⡄⠀⠀⠀⠀⠀⠀⠀│
     │⢠⢻⡇⡇⡇⢱⢸⠀⢇⢸⠀⠀⡇⢀⠇⠀⠀⡇⠀⢸⠀⠀⠀⠸⡀⠀⢠⠇⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀│
     │⢸⢸⡇⢱⡇⢸⡇⠀⢸⢸⠀⠀⢣⢸⠀⠀⠀⢸⠀⡇⠀⠀⠀⠀⢇⠀⡜⠀⠀⠀⠀⠀⠈⢇⠀⠀⠀⠀⠀⠀│
     │⢸⢸⡇⢸⠀⢸⡇⠀⢸⡇⠀⠀⢸⡎⠀⠀⠀⠈⣶⠁⠀⠀⠀⠀⠸⣤⠃⠀⠀⠀⠀⠀⠀⠘⡆⠀⠀⠀⠀⠀│
     │⢸⠀⡇⢸⠀⠀⡇⠀⠀⡇⠀⠀⠀⡇⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⢱⡀⠀⠀⠀⠀│
     │⢸⢸⡇⢸⠀⢸⡇⠀⢸⡇⠀⠀⢸⢇⠀⠀⠀⢀⠿⡀⠀⠀⠀⠀⢰⠛⡄⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀│
     │⢸⢸⡇⡸⡇⢸⡇⠀⢸⢸⠀⠀⡜⢸⠀⠀⠀⢸⠀⡇⠀⠀⠀⠀⡎⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠘⡆⠀⠀⠀│
     │⢸⢸⡇⡇⡇⡸⢸⠀⡎⢸⠀⠀⡇⠈⡆⠀⠀⡇⠀⢸⠀⠀⠀⢰⠁⠀⠘⡆⠀⠀⠀⠀⠀⠀⠀⠀⠸⡄⠀⠀│
     │⡇⢸⡇⡇⡇⡇⢸⠀⡇⠈⡆⢰⠁⠀⡇⠀⢰⠁⠀⠈⡆⠀⠀⡎⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀│
     │⡇⢸⢸⡇⡇⡇⠸⣰⠃⠀⡇⡸⠀⠀⢸⠀⡜⠀⠀⠀⢣⠀⢸⠁⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠈⢇⠀│
     │⡇⡇⢸⠇⢸⡇⠀⣿⠀⠀⢣⡇⠀⠀⠸⣄⠇⠀⠀⠀⠸⡀⡇⠀⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⡄│
   0 │⣇⣇⣸⣀⣸⣀⣀⣟⣀⣀⣸⣃⣀⣀⣀⣿⣀⣀⣀⣀⣀⣿⣀⣀⣀⣀⣀⣀⣈⣇⣀⣀⣀⣀⣀⣀⣀⣀⣀⣱│
     └────────────────────────────────────────┘
     ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀200⠀
```

source

```julia
logaddexp(a, b)
```

Adds log-space `a` and `b` such that the result equals `log(exp(a)+exp(b))`

source

```julia
im2col!(col, x, cdims, rows=1:M)
```

Converts a 3d image `x` into a matrix `col` for usage with GEMM-calculated convolution. Patches of `x` of size (kernel_w, kernel_h, kernel_d, C_in) will be extracted and laid out along the rows of `col`, one for each output pixel.  Only the output pixels `rows` (out of the `M` of the image, flattened) are extracted, into the first `length(rows)` rows of `col`.  This routine is used by all im2col-based convolutions, just with extra singleton dimensions added in the case of `2d` or `1d` images.

source

```julia
predilate(x, dilation::Tuple)
```

Places elements of `x` within a lattice of zeros, used in expressing a transposed convolution in terms of normal convolution.  Note that while we call this "predilation" for aesthetic reasons, you are typically passing a "stride" value into here.  Yes, transposed convolution is confusing.

source

```julia
safe_div(x, y)
```

Returns `x/y` unless `y==0`, in which case it just returns `x`. (Used internally by `scatter`.)

source

```julia
flipweight(w::AbstractArray)
```

Reorders the weight tensor for supporting both convolution and cross-correlation operations.

source
