# # Qwen3 Implementation from Scratch

# This is an implementation of Qwen 3 ([blog](https://qwenlm.github.io/blog/qwen3/) and
# [technical report](https://arxiv.org/abs/2505.09388)) from scratch based on the pytorch
# [implementation](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb)
# [developed in Pytorch under the Apache License 2.0](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt).

# ## Package Imports

using BFloat16s, ConcreteStructs, LinearAlgebra, Lux, Random, Reactant
using HuggingFaceTokenizers, PythonCall, SafeTensors, Scratch, JSON3

const huggingface_hub = pyimport("huggingface_hub")

# ## Qwen3 Configuration

@kwdef struct Qwen3Config{F}
    version::String
    vocab_size::Int
    context_length::Int
    emb_dim::Int
    n_heads::Int
    n_layers::Int
    hidden_dim::Int
    head_dim::Int
    n_kv_groups::Int
    rope_base::Float32
    dtype::F
    reasoning_model::Bool = true
end

function Qwen3Config(version::String; kwargs...)
    if version == "0.6B"
        return Qwen3Config(;
            version,
            vocab_size=151_936,
            context_length=40_960,
            emb_dim=1024,
            n_heads=16,
            n_layers=28,
            hidden_dim=3072,
            head_dim=128,
            n_kv_groups=8,
            rope_base=1.0f6,
            dtype=bf16,
            kwargs...,
        )
    elseif version == "1.7B"
        return Qwen3Config(;
            version,
            vocab_size=151_936,
            context_length=40_960,
            emb_dim=2048,
            n_heads=16,
            n_layers=28,
            hidden_dim=6144,
            head_dim=128,
            n_kv_groups=8,
            rope_base=1.0f6,
            dtype=bf16,
            kwargs...,
        )
    elseif version == "4B"
        return Qwen3Config(;
            version,
            vocab_size=151_936,
            context_length=40_960,
            emb_dim=2560,
            n_heads=32,
            n_layers=36,
            hidden_dim=9728,
            head_dim=128,
            n_kv_groups=8,
            rope_base=1.0f6,
            dtype=bf16,
            kwargs...,
        )
    elseif version == "8B"
        return Qwen3Config(;
            version,
            vocab_size=151_936,
            context_length=40_960,
            emb_dim=4096,
            n_heads=32,
            n_layers=36,
            hidden_dim=12288,
            head_dim=128,
            n_kv_groups=8,
            rope_base=1.0f6,
            dtype=bf16,
            kwargs...,
        )
    elseif version == "14B"
        return Qwen3Config(;
            version,
            vocab_size=151_936,
            context_length=40_960,
            emb_dim=5120,
            n_heads=40,
            n_layers=40,
            hidden_dim=17408,
            head_dim=128,
            n_kv_groups=8,
            rope_base=1.0f6,
            dtype=bf16,
            kwargs...,
        )
    elseif version == "32B"
        return Qwen3Config(;
            version,
            vocab_size=151_936,
            context_length=40_960,
            emb_dim=5120,
            n_heads=64,
            n_layers=64,
            hidden_dim=25600,
            head_dim=128,
            n_kv_groups=8,
            rope_base=1.0f6,
            dtype=bf16,
            kwargs...,
        )
    end

    return throw(ArgumentError("Unknown Qwen3 version $version"))
end

fn_to_dtype(::Type{T}) where {T} = T
fn_to_dtype(::typeof(f16)) = Float16
fn_to_dtype(::typeof(f32)) = Float32
fn_to_dtype(::typeof(f64)) = Float64
fn_to_dtype(::typeof(bf16)) = BFloat16

# ## Model Definition

function Qwen3MLP(cfg::Qwen3Config)
    return Chain(;
        proj=Parallel(
            .*;
            gate_proj=Dense(cfg.emb_dim => cfg.hidden_dim, swish; use_bias=false),
            up_proj=Dense(cfg.emb_dim => cfg.hidden_dim; use_bias=false),
        ),
        down_proj=Dense(cfg.hidden_dim => cfg.emb_dim; use_bias=false),
        name="Qwen3MLP",
    )
end

Qwen3RMSNorm(emb_dim::Int, eps) = AlternatePrecision{Float32}(RMSNorm(emb_dim; epsilon=eps))

@concrete struct GroupedQueryAttention <: AbstractLuxContainerLayer{(
    :q_proj, :k_proj, :v_proj, :o_proj, :q_norm, :k_norm
)}
    q_proj
    k_proj
    v_proj
    o_proj
    q_norm
    k_norm
    d_in::Int
    num_heads::Int
    num_kv_groups::Int
    head_dim::Int
end

function GroupedQueryAttention(d_in, num_heads, num_kv_groups; head_dim=nothing)
    @assert num_heads % num_kv_groups == 0 "num_heads must be divisible by num_kv_groups"

    if head_dim === nothing
        @assert d_in % num_heads == 0 "`d_in` must be divisible by `num_heads` if \
                                       `head_dim` is not set"
        head_dim = d_in ÷ num_heads
    end

    d_out = num_heads * head_dim

    return GroupedQueryAttention(
        Dense(d_in, d_out; use_bias=false),
        Dense(d_in, num_kv_groups * head_dim; use_bias=false),
        Dense(d_in, num_kv_groups * head_dim; use_bias=false),
        Dense(d_out, d_in; use_bias=false),
        Qwen3RMSNorm(head_dim, 1.0f-6),
        Qwen3RMSNorm(head_dim, 1.0f-6),
        d_in,
        num_heads,
        num_kv_groups,
        head_dim,
    )
end

function apply_rope(x::AbstractArray{T}, cos_cache, sin_cache) where {T}
    return T.(apply_rotary_embedding(x, cos_cache, sin_cache; seq_dim=3))
end

function _gqa_qkv_norm_rope(attn::GroupedQueryAttention, x, cos_cache, sin_cache, ps, st)
    _, num_tokens, B = size(x)

    ## apply projections
    queries, st_q_proj = attn.q_proj(x, ps.q_proj, st.q_proj)
    keys, st_k_proj = attn.k_proj(x, ps.k_proj, st.k_proj)
    values, st_v_proj = attn.v_proj(x, ps.v_proj, st.v_proj)

    ## reshape to (head_dim, num_heads/num_kv_groups, num_tokens, batch)
    queries = reshape(queries, attn.head_dim, attn.num_heads, num_tokens, B)
    keys = reshape(keys, attn.head_dim, attn.num_kv_groups, num_tokens, B)
    values = reshape(values, attn.head_dim, attn.num_kv_groups, num_tokens, B)

    ## apply normalization
    queries, st_q_norm = attn.q_norm(queries, ps.q_norm, st.q_norm)
    keys, st_k_norm = attn.k_norm(keys, ps.k_norm, st.k_norm)

    ## apply RoPE
    queries = apply_rope(queries, cos_cache, sin_cache)
    keys = apply_rope(keys, cos_cache, sin_cache)

    return queries, keys, values, st_q_proj, st_k_proj, st_v_proj, st_q_norm, st_k_norm
end

function _gqa_out_proj(
    attn::GroupedQueryAttention,
    context,
    num_tokens,
    B,
    ps,
    st_q_proj,
    st_k_proj,
    st_v_proj,
    st_q_norm,
    st_k_norm,
    st_o_proj_in,
)
    context_flat = reshape(context, attn.head_dim * attn.num_heads, num_tokens, B)
    proj, st_o_proj = attn.o_proj(context_flat, ps.o_proj, st_o_proj_in)
    return (
        proj,
        (;
            q_proj=st_q_proj,
            k_proj=st_k_proj,
            v_proj=st_v_proj,
            o_proj=st_o_proj,
            q_norm=st_q_norm,
            k_norm=st_k_norm,
        ),
    )
end

function (attn::GroupedQueryAttention)((x, cos_cache, sin_cache), ps, st::NamedTuple)
    _, num_tokens, B = size(x)
    queries, keys, values, st_q_proj, st_k_proj, st_v_proj, st_q_norm, st_k_norm = _gqa_qkv_norm_rope(
        attn, x, cos_cache, sin_cache, ps, st
    )

    context = scaled_dot_product_attention(
        queries,
        keys,
        values;
        head_dim=1,
        token_dim=3,
        is_causal=true,
        return_attention_weights=false,
    )[1]

    return _gqa_out_proj(
        attn,
        context,
        num_tokens,
        B,
        ps,
        st_q_proj,
        st_k_proj,
        st_v_proj,
        st_q_norm,
        st_k_norm,
        st.o_proj,
    )
end

function (attn::GroupedQueryAttention)(
    (x, cos_cache, sin_cache, kv_cache)::Tuple{Any,Any,Any,Tuple}, ps, st::NamedTuple
)
    _, num_tokens, B = size(x)
    queries, keys, values, st_q_proj, st_k_proj, st_v_proj, st_q_norm, st_k_norm = _gqa_qkv_norm_rope(
        attn, x, cos_cache, sin_cache, ps, st
    )

    k_cache, v_cache = kv_cache
    k_cache[:, :, 1:num_tokens, :] = keys
    v_cache[:, :, 1:num_tokens, :] = values

    context = scaled_dot_product_attention(
        queries,
        keys,
        values;
        head_dim=1,
        token_dim=3,
        is_causal=true,
        return_attention_weights=false,
    )[1]

    return _gqa_out_proj(
        attn,
        context,
        num_tokens,
        B,
        ps,
        st_q_proj,
        st_k_proj,
        st_v_proj,
        st_q_norm,
        st_k_norm,
        st.o_proj,
    )
end

function (attn::GroupedQueryAttention)(
    (x, cos_pos, sin_pos, pos, kv_cache)::Tuple{Any,Any,Any,Any,Tuple}, ps, st::NamedTuple
)
    _, num_tokens, B = size(x)
    queries, keys, values, st_q_proj, st_k_proj, st_v_proj, st_q_norm, st_k_norm = _gqa_qkv_norm_rope(
        attn, x, cos_pos, sin_pos, ps, st
    )

    k_cache, v_cache = kv_cache
    k_cache[:, :, pos, :] = reshape(keys, attn.head_dim, attn.num_kv_groups, B)
    v_cache[:, :, pos, :] = reshape(values, attn.head_dim, attn.num_kv_groups, B)

    context = scaled_dot_product_attention(
        queries,
        k_cache,
        v_cache;
        head_dim=1,
        token_dim=3,
        kv_lengths=pos,
        return_attention_weights=false,
    )[1]

    return _gqa_out_proj(
        attn,
        context,
        num_tokens,
        B,
        ps,
        st_q_proj,
        st_k_proj,
        st_v_proj,
        st_q_norm,
        st_k_norm,
        st.o_proj,
    )
end

@concrete struct Qwen3Attention <: AbstractLuxContainerLayer{(
    :self_attn, :mlp, :input_layernorm, :post_attention_layernorm
)}
    self_attn <: GroupedQueryAttention
    mlp
    input_layernorm
    post_attention_layernorm
end

function Qwen3Attention(cfg::Qwen3Config)
    return Qwen3Attention(
        GroupedQueryAttention(cfg.emb_dim, cfg.n_heads, cfg.n_kv_groups; cfg.head_dim),
        Qwen3MLP(cfg),
        Qwen3RMSNorm(cfg.emb_dim, 1.0f-6),
        Qwen3RMSNorm(cfg.emb_dim, 1.0f-6),
    )
end

function _block_forward(block::Qwen3Attention, x, attn_args::Tuple, ps, st::NamedTuple)
    ## shortcut connection for attention block
    shortcut = x
    x, st_norm1 = block.input_layernorm(x, ps.input_layernorm, st.input_layernorm)
    x, st_attn = block.self_attn((x, attn_args...), ps.self_attn, st.self_attn)
    x = x .+ shortcut

    ## shortcut connection for feed-forward block
    shortcut = x
    x, st_norm2 = block.post_attention_layernorm(
        x, ps.post_attention_layernorm, st.post_attention_layernorm
    )
    x, st_ff = block.mlp(x, ps.mlp, st.mlp)
    x = x .+ shortcut

    return (
        x,
        (;
            self_attn=st_attn,
            mlp=st_ff,
            input_layernorm=st_norm1,
            post_attention_layernorm=st_norm2,
        ),
    )
end

function (block::Qwen3Attention)((x, cos_cache, sin_cache), ps, st::NamedTuple)
    return _block_forward(block, x, (cos_cache, sin_cache), ps, st)
end

function (block::Qwen3Attention)(
    (x, cos_cache, sin_cache, kv_cache)::Tuple{Any,Any,Any,Tuple}, ps, st::NamedTuple
)
    return _block_forward(block, x, (cos_cache, sin_cache, kv_cache), ps, st)
end

function (block::Qwen3Attention)(
    (x, cos_pos, sin_pos, pos, kv_cache)::Tuple{Any,Any,Any,Any,Tuple}, ps, st::NamedTuple
)
    return _block_forward(block, x, (cos_pos, sin_pos, pos, kv_cache), ps, st)
end

@concrete struct Qwen3 <:
                 AbstractLuxContainerLayer{(:embed_tokens, :blocks, :norm, :lm_head)}
    embed_tokens
    blocks
    norm
    lm_head
    cfg::Qwen3Config
end

function Qwen3(cfg::Qwen3Config)
    return Qwen3(
        Embedding(cfg.vocab_size => cfg.emb_dim),
        Tuple([Qwen3Attention(cfg) for _ in 1:(cfg.n_layers)]),
        Qwen3RMSNorm(cfg.emb_dim, 1.0f-6),
        Dense(cfg.emb_dim, cfg.vocab_size; use_bias=false),
        cfg,
    )
end

function init_kv_cache(
    cfg::Qwen3Config, max_cache_len::Int, batch_size::Int, dev=reactant_device(; force=true)
)
    dtype = fn_to_dtype(cfg.dtype)
    z = zeros(dtype, cfg.head_dim, cfg.n_kv_groups, max_cache_len, batch_size)
    return ntuple(_ -> (dev(z), dev(z)), cfg.n_layers)
end

function LuxCore.initialstates(rng::AbstractRNG, m::Qwen3)
    head_dim = m.cfg.head_dim === nothing ? m.cfg.emb_dim ÷ m.cfg.n_heads : m.cfg.head_dim
    (; cos_cache, sin_cache) = compute_rotary_embedding_params(
        head_dim, m.cfg.context_length; base=m.cfg.rope_base, dtype=Float32
    )
    return (;
        cos_cache,
        sin_cache,
        embed_tokens=LuxCore.initialstates(rng, m.embed_tokens),
        blocks=LuxCore.initialstates(rng, m.blocks),
        norm=LuxCore.initialstates(rng, m.norm),
        lm_head=LuxCore.initialstates(rng, m.lm_head),
    )
end

function _forward_transformer(qwen3::Qwen3, in_idx, ps, st::NamedTuple)
    x, st_embed_tokens = qwen3.embed_tokens(in_idx, ps.embed_tokens, st.embed_tokens)

    st_blocks = ()
    for (i, block) in enumerate(qwen3.blocks)
        x, st_block_new = block((x, st.cos_cache, st.sin_cache), ps.blocks[i], st.blocks[i])
        st_blocks = (st_blocks..., st_block_new)
    end
    return x, st_embed_tokens, st_blocks
end

function _forward_transformer(qwen3::Qwen3, in_idx, ps, st::NamedTuple, kv_cache::Tuple)
    x, st_embed_tokens = qwen3.embed_tokens(in_idx, ps.embed_tokens, st.embed_tokens)

    st_blocks = ()
    for (i, block) in enumerate(qwen3.blocks)
        x, st_block_new = block(
            (x, st.cos_cache, st.sin_cache, kv_cache[i]), ps.blocks[i], st.blocks[i]
        )
        st_blocks = (st_blocks..., st_block_new)
    end
    return x, st_embed_tokens, st_blocks
end

function _apply_head(qwen3::Qwen3, x, ps, st::NamedTuple, st_embed_tokens, st_blocks)
    x, st_norm = qwen3.norm(x, ps.norm, st.norm)
    logits, st_lm_head = qwen3.lm_head(
        fn_to_dtype(qwen3.cfg.dtype).(x), ps.lm_head, st.lm_head
    )

    return (
        logits,
        (;
            cos_cache=st.cos_cache,
            sin_cache=st.sin_cache,
            embed_tokens=st_embed_tokens,
            blocks=st_blocks,
            norm=st_norm,
            lm_head=st_lm_head,
        ),
    )
end

function (qwen3::Qwen3)(in_idx::AbstractArray, ps, st::NamedTuple)
    x, st_embed_tokens, st_blocks = _forward_transformer(qwen3, in_idx, ps, st)
    return _apply_head(qwen3, x, ps, st, st_embed_tokens, st_blocks)
end

function (qwen3::Qwen3)((in_idx, token_idx)::Tuple{<:AbstractArray,Any}, ps, st::NamedTuple)
    x, st_embed_tokens, st_blocks = _forward_transformer(qwen3, in_idx, ps, st)
    x_last = x[:, token_idx, :]
    return _apply_head(qwen3, x_last, ps, st, st_embed_tokens, st_blocks)
end

function (qwen3::Qwen3)(
    (in_idx, token_idx, kv_cache)::Tuple{<:AbstractArray,Any,Tuple}, ps, st::NamedTuple
)
    x, st_embed_tokens, st_blocks = _forward_transformer(qwen3, in_idx, ps, st, kv_cache)
    x_last = reshape(x[:, token_idx, :], size(x, 1), size(x, 3))
    return _apply_head(qwen3, x_last, ps, st, st_embed_tokens, st_blocks)
end

function (qwen3::Qwen3)(
    (token_id, pos, kv_cache, _)::Tuple{<:AbstractArray,Any,Tuple,Val{:decode}},
    ps,
    st::NamedTuple,
)
    x, st_embed_tokens = qwen3.embed_tokens(token_id, ps.embed_tokens, st.embed_tokens)

    cos_pos = reshape(st.cos_cache[:, pos], :, 1)
    sin_pos = reshape(st.sin_cache[:, pos], :, 1)

    st_blocks = ()
    for (i, block) in enumerate(qwen3.blocks)
        x, st_block_new = block(
            (x, cos_pos, sin_pos, pos, kv_cache[i]), ps.blocks[i], st.blocks[i]
        )
        st_blocks = (st_blocks..., st_block_new)
    end

    x_last = reshape(x, size(x, 1), size(x, 3))
    return _apply_head(qwen3, x_last, ps, st, st_embed_tokens, st_blocks)
end

# ## Model Weights and Tokenizer from HuggingFace

function download_qwen3_weights_from_huggingface(cfg::Qwen3Config)
    return download_qwen3_weights_from_huggingface(cfg.reasoning_model, cfg.version)
end

function download_qwen3_weights_from_huggingface(use_reasoning_model::Bool, version::String)
    repo_id = "Qwen/Qwen3-$(version)" * (use_reasoning_model ? "" : "-Base")
    local_dir = @get_scratch!("Qwen3-$(version)-$(use_reasoning_model)")

    tokenizer_file = huggingface_hub.hf_hub_download(;
        repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir
    )

    if version == "0.6B"
        weights_file = huggingface_hub.hf_hub_download(;
            repo_id=repo_id, filename="model.safetensors", local_dir=local_dir
        )
        weights_dict = load_safetensors(string(weights_file))
    else
        repo_dir = huggingface_hub.snapshot_download(; repo_id=repo_id, local_dir=local_dir)
        index_path = joinpath(string(repo_dir), "model.safetensors.index.json")

        index = JSON3.read(index_path)

        weights_dict = Dict()
        for filename in Set(values(index["weight_map"]))
            shard_path = joinpath(string(repo_dir), filename)
            shard = load_safetensors(shard_path)
            merge!(weights_dict, shard)
        end
    end

    return weights_dict, string(tokenizer_file), repo_id
end

# ## Qwen3 Tokenizer

struct Qwen3Tokenizer
    tokenizer::Tokenizer
    special_to_id::Dict{String,Int32}
    pad_token_id::Int32
    eos_token_id::Int32
    apply_chat_template::Bool
    add_generation_prompt::Bool
    add_thinking::Bool
end

function Base.show(io::IO, tokenizer::Qwen3Tokenizer)
    return print(
        io,
        "Qwen3Tokenizer(apply_chat_template=$(tokenizer.apply_chat_template), add_generation_prompt=$(tokenizer.add_generation_prompt), add_thinking=$(tokenizer.add_thinking))",
    )
end

const SPECIALS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
]

const SPLIT_RE = r"(<\|[^>]+?\|>)"

token_to_id(tokenizer::Qwen3Tokenizer, s) = token_to_id(tokenizer.tokenizer, s)
function token_to_id(tokenizer::Tokenizer, s)
    return pyconvert(Int32, tokenizer.py_tokenizer.token_to_id(s)) + Int32(1)
end

function split_with_delims(text::String, re::Regex)
    parts = String[]
    last_end = 1
    for m in eachmatch(re, text)
        if m.offset > last_end
            push!(parts, text[last_end:(m.offset - 1)])
        elseif m.offset == 1
            push!(parts, "")
        end
        push!(parts, m.match)
        last_end = m.offset + length(m.match)
    end
    if last_end ≤ lastindex(text)
        push!(parts, text[last_end:end])
    end
    return parts
end

function Qwen3Tokenizer(
    tokenizer_file_path::String;
    repo_id=nothing,
    apply_chat_template::Bool=true,
    add_generation_prompt::Bool=false,
    add_thinking::Bool=false,
)
    tok = HuggingFaceTokenizers.from_file(Tokenizer, tokenizer_file_path)
    special_to_id = Dict(s => token_to_id(tok, s) for s in SPECIALS)
    pad_token_id = special_to_id["<|endoftext|>"]
    eos_token_id = pad_token_id
    if repo_id !== nothing && !occursin("Base", repo_id)
        eos_token = "<|im_end|>"
    else
        eos_token = "<|endoftext|>"
    end
    if haskey(special_to_id, eos_token)
        eos_token_id = special_to_id[eos_token]
    end
    return Qwen3Tokenizer(
        tok,
        special_to_id,
        pad_token_id,
        eos_token_id,
        apply_chat_template,
        add_generation_prompt,
        add_thinking,
    )
end

function wrap_chat(tokenizer::Qwen3Tokenizer, user_msg::AbstractString)
    s = "<|im_start|>user\n$(user_msg)<|im_end|>\n"
    if tokenizer.add_generation_prompt
        s *= "<|im_start|>assistant"
        if tokenizer.add_thinking
            s *= "\n"
        else
            s *= "\n<think>\n\n</think>\n\n"
        end
    end
    return s
end

function HuggingFaceTokenizers.encode(
    tok::Qwen3Tokenizer, text; chat_wrapped::Bool=tok.apply_chat_template
)
    stripped = strip(text)
    if haskey(tok.special_to_id, stripped) && !occursin('\n', stripped)
        return [tok.special_to_id[stripped]]
    end

    chat_wrapped && (text = wrap_chat(tok, text))

    ids = Int32[]
    for part in filter(!isempty, split_with_delims(text, SPLIT_RE))
        if haskey(tok.special_to_id, part)
            push!(ids, tok.special_to_id[part])
        else
            append!(ids, encode(tok.tokenizer, string(part)).ids .+ Int16(1))
        end
    end
    return ids
end

function HuggingFaceTokenizers.decode(tok::Qwen3Tokenizer, ids::Vector{<:Integer})
    return decode(tok.tokenizer, ids .- Int16(1); skip_special_tokens=false)
end

# ## Pretrained Model Weights

get_weights_tensor(tensor::AbstractArray, ::Type{T}) where {T} = collect(T, tensor)

function get_weights_tensor(dict, key, dtype::Type{T}, dev; permute::Bool=false) where {T}
    tensor = dict[key]
    if permute
        tensor = permutedims(tensor, Tuple(reverse(1:ndims(tensor))))
    end
    return get_weights_tensor(tensor, dtype) |> dev
end

function load_weights_from_dict(weights_dict, cfg::Qwen3Config, dev)
    dtype = fn_to_dtype(cfg.dtype)

    function get_tensor(key; kwargs...)
        return get_weights_tensor(weights_dict, key, dtype, dev; kwargs...)
    end

    embed_tokens = (; weight=get_tensor("model.embed_tokens.weight"; permute=true))

    blocks = Vector{Any}(undef, cfg.n_layers)
    for l in 1:(cfg.n_layers)
        prefix = "model.layers.$(l - 1)"
        sa_prefix = "$(prefix).self_attn"

        blocks[l] = (;
            self_attn=merge(
                NamedTuple(
                    k => (; weight=get_tensor("$(sa_prefix).$k.weight")) for
                    k in (:q_proj, :k_proj, :v_proj, :o_proj)
                ),
                NamedTuple(
                    k => (; scale=get_tensor("$(sa_prefix).$k.weight")) for
                    k in (:q_norm, :k_norm)
                ),
            ),
            mlp=(;
                proj=(;
                    gate_proj=(; weight=get_tensor("$(prefix).mlp.gate_proj.weight")),
                    up_proj=(; weight=get_tensor("$(prefix).mlp.up_proj.weight")),
                ),
                down_proj=(; weight=get_tensor("$(prefix).mlp.down_proj.weight")),
            ),
            input_layernorm=(; scale=get_tensor("$(prefix).input_layernorm.weight")),
            post_attention_layernorm=(;
                scale=get_tensor("$(prefix).post_attention_layernorm.weight")
            ),
        )
    end
    blocks = Tuple(blocks)

    norm = (; scale=get_weights_tensor(weights_dict, "model.norm.weight", dtype, dev))

    if haskey(weights_dict, "lm_head.weight")
        lm_head = (; weight=get_weights_tensor(weights_dict, "lm_head.weight", dtype, dev))
    else
        ## Weight tying with the embedding matrix. We will share the weights here to
        ## reduce memory usage.
        lm_head = (; weight=transpose(embed_tokens.weight))
    end

    return (; embed_tokens, blocks, norm, lm_head)
end

function setup_model(
    version::String, dev; weights_dict::Union{Nothing,Dict}=nothing, kwargs...
)
    return setup_model(Qwen3Config(version; kwargs...), dev; weights_dict)
end

function setup_model(cfg::Qwen3Config, dev; weights_dict::Union{Nothing,Dict}=nothing)
    model = Qwen3(cfg)

    st = Lux.initialstates(Random.default_rng(), model) |> dev

    if weights_dict !== nothing
        ps = load_weights_from_dict(weights_dict, cfg, dev)
    else
        ps = Lux.initialparameters(Random.default_rng(), model) |> dev
        ps = ps |> cfg.dtype
    end

    return model, ps, st
end

# ## Running the model without dynamic sizes

function get_padded_size(seq_len::Int, context_length::Int)
    return min(max(512, nextpow(2, seq_len)), context_length)
end

function get_max_cache_len(total_len::Int, context_length::Int)
    return min(context_length, max(256, nextpow(2, total_len)))
end

function get_prefill_size(prompt_len::Int, max_cache_len::Int)
    return min(max_cache_len, max(32, nextpow(2, prompt_len)))
end

function padded_input_and_mask_len(x::AbstractMatrix, v, cfg::Qwen3Config, pad_token_id)
    return padded_input_and_mask_len(
        x, v, get_padded_size(size(x, 1) + v !== nothing, cfg.context_length), pad_token_id
    )
end

function padded_input_and_mask_len(x::AbstractMatrix, v, padded_sz::Int, pad_token_id)
    if padded_sz > size(x, 1)
        x_padded = similar(x, (padded_sz, size(x, 2)))
        x_padded[1:size(x, 1), :] .= x
        if v === nothing
            x_padded[(size(x, 1) + 1):end, :] .= pad_token_id
        else
            x_padded[(size(x, 1) + 1), :] = v[1, :]
            x_padded[(size(x, 1) + 2):end, :] .= pad_token_id
        end
    else
        x_padded = x
    end
    return (
        x_padded,
        Reactant.promote_to(
            Reactant.TracedRNumber{Int32}, padded_sz - (size(x, 1) + (v !== nothing))
        ),
    )
end

# ## Helpers to generate text

function _logits_to_tokens(logits, ::Type{T}) where {T}
    predictions = T.(argmax(logits; dims=1))
    return mod1.(predictions, T(size(logits, 1)))
end

function predict_next_token(
    model, padded_token_ids::AbstractMatrix{T}, input_mask_len, ps, st
) where {T}
    last_token_idx = size(padded_token_ids, 1) - input_mask_len
    logits, stₙ = model((padded_token_ids, last_token_idx), ps, st)
    return _logits_to_tokens(logits, T), stₙ
end

function prefill_next_token!(
    model::Qwen3, padded_token_ids::AbstractMatrix{T}, prompt_len, ps, st, kv_cache
) where {T}
    logits, stₙ = model((padded_token_ids, prompt_len, kv_cache), ps, st)
    return _logits_to_tokens(logits, T), stₙ
end

function decode_next_token!(
    model::Qwen3, token_id::AbstractMatrix{T}, pos, ps, st, kv_cache
) where {T}
    logits, stₙ = model((token_id, pos, kv_cache, Val(:decode)), ps, st)
    return _logits_to_tokens(logits, T), stₙ
end

function update_token_ids_and_mask!(
    padded_token_ids, input_mask_len, cur_num_tokens, next_token
)
    next_token_idx = safe_increment(cur_num_tokens)
    padded_token_ids[next_token_idx, :] = next_token[1, :]
    return input_mask_len - eltype(input_mask_len)(1), next_token_idx
end

function update_token_ids_with_shift!(token_ids, next_token)
    token_ids[1:(end - 1), :] = token_ids[2:end, :]
    token_ids[end, :] = next_token[1, :]
    return nothing
end

safe_increment(x) = x + one(x)

function generate_tokens_compiled(
    model::Qwen3,
    padded_prompt::AbstractMatrix,
    prompt_len,
    max_new_tokens,
    eos_ids::AbstractVector,
    ::Val{max_cache_len},
    ps,
    st,
) where {max_cache_len}
    B = size(padded_prompt, 2)
    dtype = fn_to_dtype(model.cfg.dtype)
    kv_cache = ntuple(
        _ -> (
            Reactant.@opcall(
                fill(
                    zero(dtype),
                    (model.cfg.head_dim, model.cfg.n_kv_groups, max_cache_len, B),
                )
            ),
            Reactant.@opcall(
                fill(
                    zero(dtype),
                    (model.cfg.head_dim, model.cfg.n_kv_groups, max_cache_len, B),
                )
            ),
        ),
        model.cfg.n_layers,
    )
    output_tokens = Reactant.@opcall fill(Int32(0), (max_cache_len, B))

    logits, st_cur = model((padded_prompt, prompt_len, kv_cache), ps, st)
    next_token = _logits_to_tokens(logits, eltype(padded_prompt))

    step = Int32(1)
    output_tokens[step, :] = next_token[1, :]
    eos_ids_col = Reactant.@opcall reshape(eos_ids, length(eos_ids), 1)
    hit_eos = all(any(eos_ids_col .== next_token; dims=1))

    Reactant.@trace while (step < max_new_tokens) & (!hit_eos)
        pos = prompt_len + step
        logits, st_cur = model((next_token, pos, kv_cache, Val(:decode)), ps, st_cur)
        next_token = _logits_to_tokens(logits, eltype(padded_prompt))
        step = step + Int32(1)
        output_tokens[step, :] = next_token[1, :]
        hit_eos = all(any(eos_ids_col .== next_token; dims=1))
    end

    return output_tokens, step
end

mutable struct CachedReactantThunks
    generation_cache::Dict{Tuple{Qwen3Config,Int,Int},Reactant.Compiler.Thunk}
end

function CachedReactantThunks()
    return CachedReactantThunks(Dict{Tuple{Qwen3Config,Int,Int},Reactant.Compiler.Thunk}())
end

function cache_and_retrieve_generation!(
    cache::CachedReactantThunks,
    prefill_len::Integer,
    max_cache_len::Integer,
    model::Qwen3,
    padded_prompt,
    prompt_len_traced,
    max_new_tokens_traced,
    eos_ids,
    ps,
    st,
)
    key = (model.cfg, Int(prefill_len), Int(max_cache_len))
    haskey(cache.generation_cache, key) && return cache.generation_cache[key]

    println()
    @warn "Compiling Qwen3 generation loop for $(model.cfg.version) with \
           prefill_len=$(prefill_len), max_cache_len=$(max_cache_len). This might take a \
           while... (However this is only done once per model per bucket)"

    return cache.generation_cache[key] = @compile generate_tokens_compiled(
        model,
        padded_prompt,
        prompt_len_traced,
        max_new_tokens_traced,
        eos_ids,
        Val(Int(max_cache_len)),
        ps,
        st,
    )
end

const CACHED_THUNKS = CachedReactantThunks()

generate_text(args...; kwargs...) = generate_text!(CACHED_THUNKS, args...; kwargs...)

function _prepare_eos_ids(eos_token_id, dev)
    if eos_token_id === nothing
        return dev(Int32[-1])
    elseif eos_token_id isa Integer
        return dev(Int32[Int32(eos_token_id)])
    else
        ids = collect(Int32, eos_token_id)
        return isempty(ids) ? dev(Int32[-1]) : dev(ids)
    end
end

_matches_eos(tok_id::Integer, eos::Integer) = tok_id == eos
_matches_eos(tok_id::Integer, eos) = tok_id in eos
_matches_eos(::Integer, ::Nothing) = false

function generate_text!(
    compile_cache::CachedReactantThunks,
    model::Qwen3,
    prompt::String,
    ps,
    st,
    max_new_tokens,
    tokenizer,
)
    raw_ids = encode(tokenizer, prompt)
    max_context_length = model.cfg.context_length
    if length(raw_ids) > max_context_length
        raw_ids = raw_ids[(end - max_context_length + 1):end]
    end
    prompt_len = length(raw_ids)
    max_new_tokens = min(max_new_tokens, max(0, max_context_length - prompt_len))
    max_new_tokens <= 0 && return 0.0

    dev = get_device(ps)
    max_cache_len = get_max_cache_len(prompt_len + max_new_tokens, max_context_length)
    prefill_len = get_prefill_size(prompt_len, max_cache_len)

    padded_ids = fill(Int32(tokenizer.pad_token_id), prefill_len, 1)
    padded_ids[1:prompt_len, 1] .= Int32.(raw_ids)
    padded_token_ids = dev(padded_ids)

    prompt_len_traced = ConcreteRNumber{Int32}(prompt_len)
    max_new_tokens_traced = ConcreteRNumber{Int32}(max_new_tokens)
    eos_ids = _prepare_eos_ids(tokenizer.eos_token_id, dev)

    gen_fn = cache_and_retrieve_generation!(
        compile_cache,
        prefill_len,
        max_cache_len,
        model,
        padded_token_ids,
        prompt_len_traced,
        max_new_tokens_traced,
        eos_ids,
        ps,
        st,
    )

    start_time = time()
    output_tokens, num_generated_dev = gen_fn(
        model,
        padded_token_ids,
        prompt_len_traced,
        max_new_tokens_traced,
        eos_ids,
        Val(max_cache_len),
        ps,
        st,
    )
    output_host = Array(output_tokens)
    num_generated = Int(num_generated_dev)
    total_time = time() - start_time

    generated_tokens = output_host[1:num_generated, 1]
    tokens_to_decode = if _matches_eos(generated_tokens[end], tokenizer.eos_token_id)
        generated_tokens[1:(end - 1)]
    else
        generated_tokens
    end
    print(decode(tokenizer, tokens_to_decode))
    println()
    return num_generated / total_time
end

# ## Entry Point

function run_model_selection()
    printstyled("Which model do you want to run? \n"; color=:cyan, bold=true)
    choices = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]
    for (i, choice) in enumerate(choices)
        printstyled("    $(i). $(choice)\n"; color=:light_blue)
    end
    printstyled("  Enter your choice: "; color=:cyan)
    choice = parse(Int, readline(stdin))
    if choice ∉ 1:length(choices)
        error("Invalid choice: $(choice). Expected an integer between 1 and \
               $(length(choices))")
    end

    printstyled("Do you want to use the reasoning model? [y/N] "; color=:cyan)
    reasoning = readline(stdin) == "y"
    println()
    return choices[choice], reasoning
end

function get_model_and_tokenizer(version, reasoning)
    cfg = Qwen3Config(version; reasoning_model=reasoning)
    rdev = reactant_device(; force=true)
    weights_dict, tokenizer_file, repo_id = download_qwen3_weights_from_huggingface(cfg)
    tokenizer = Qwen3Tokenizer(
        tokenizer_file;
        repo_id,
        add_generation_prompt=cfg.reasoning_model,
        add_thinking=cfg.reasoning_model,
    )
    model, ps, st = setup_model(cfg, rdev; weights_dict)
    return model, ps, st, tokenizer
end

function main()
    @info "Text Generation with Qwen-3 powered by Lux, Reactant & XLA."

    version, reasoning = run_model_selection()
    model, ps, st, tokenizer = get_model_and_tokenizer(version, reasoning)

    while true
        printstyled(
            "Prompt (type \"exit\" to quit the program or \
             \"model selection\" to change the model): ";
            color=:cyan,
            bold=true,
        )
        prompt = readline(stdin)

        prompt == "exit" && break

        if prompt == "model selection"
            version, reasoning = run_model_selection()
            model, ps, st, tokenizer = get_model_and_tokenizer(version, reasoning)
            continue
        end

        tokens_per_second = generate_text(model, prompt, ps, st, 100_000, tokenizer)
        println("\nTokens per second: $tokens_per_second\n\n")
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
