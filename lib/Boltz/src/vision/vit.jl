# Custom Layers needed for ViT
"""
    MultiHeadAttention(in_planes::Int, number_heads::Int; qkv_bias::Bool=false,
                       attention_dropout_rate::T=0.0f0,
                       projection_dropout_rate::T=0.0f0) where {T}

Multi-head self-attention layer
"""
struct MultiHeadAttention{Q, A, P} <:
       Lux.AbstractExplicitContainerLayer{(:qkv_layer, :attention_dropout, :projection)}
    number_heads::Int
    qkv_layer::Q
    attention_dropout::A
    projection::P
end

function MultiHeadAttention(in_planes::Int, number_heads::Int; qkv_bias::Bool=false,
                            attention_dropout_rate::T=0.0f0,
                            projection_dropout_rate::T=0.0f0) where {T}
    @assert in_planes % number_heads==0 "`in_planes` should be divisible by `number_heads`"
    qkv_layer = Dense(in_planes, in_planes * 3; bias=qkv_bias)
    attention_dropout = Dropout(attention_dropout_rate)
    projection = Chain(Dense(in_planes, in_planes), Dropout(projection_dropout_rate))

    return MultiHeadAttention(number_heads, qkv_layer, attention_dropout, projection)
end

function (m::MultiHeadAttention)(x::AbstractArray{T, 3}, ps, st) where {T}
    nfeatures, seq_len, batch_size = size(x)

    x_reshaped = reshape(x, nfeatures, seq_len * batch_size)
    qkv, st_qkv = m.qkv_layer(x_reshaped, ps.qkv_layer, st.qkv_layer)
    qkv_reshaped = reshape(qkv, nfeatures ÷ m.number_heads, m.number_heads, seq_len,
                           3 * batch_size)
    query, key, value = fast_chunk(qkv_reshaped, Val(3), Val(4))

    scale = convert(T, sqrt(size(query, 1) / m.number_heads))
    key_reshaped = reshape(permutedims(key, (2, 1, 3, 4)), m.number_heads,
                           nfeatures ÷ m.number_heads, seq_len * batch_size)
    query_reshaped = reshape(query, nfeatures ÷ m.number_heads, m.number_heads,
                             seq_len * batch_size)

    attention = softmax(batched_mul(query_reshaped, key_reshaped) .* scale)
    attention, st_attention = m.attention_dropout(attention, ps.attention_dropout,
                                                  st.attention_dropout)

    value_reshaped = reshape(value, nfeatures ÷ m.number_heads, m.number_heads,
                             seq_len * batch_size)
    pre_projection = reshape(batched_mul(attention, value_reshaped),
                             (nfeatures, seq_len, batch_size))
    y, st_projection = m.projection(reshape(pre_projection, size(pre_projection, 1), :),
                                    ps.projection, st.projection)

    st_ = (qkv_layer=st_qkv, attention=st_attention, projection=st_projection)
    return reshape(y, :, seq_len, batch_size), st_
end

"""
    ClassTokens(dim; init=Lux.zeros32)

Appends class tokens to an input with embedding dimension `dim` for use in many vision
transformer namels.
"""
struct ClassTokens{I} <: Lux.AbstractExplicitLayer
    dim::Int
    init::I
end

ClassTokens(dim::Int; init=Lux.zeros32) = ClassTokens(dim, init)

Lux.initialparameters(rng::AbstractRNG, c::ClassTokens) = (token=c.init(rng, c.dim, 1, 1),)

_fill_like(y::AbstractArray{T, 3}) where {T} = fill!(similar(y, 1, 1, size(y, 3)), one(T))
ChainRulesCore.@non_differentiable _fill_like(y)

function (m::ClassTokens)(x::AbstractArray{T, 3}, ps, st) where {T}
    # Generic Alternative: Repeat is extremely inefficient on GPUs and even in general
    tokens = ps.token .* _fill_like(x)
    return hcat(tokens, x), st
end

"""
    ViPosEmbedding(embedsize, npatches;
                   init = (rng, dims...) -> randn(rng, Float32, dims...))

Positional embedding layer used by many vision transformer-like namels.
"""
struct ViPosEmbedding{I} <: Lux.AbstractExplicitLayer
    embedding_size::Int
    number_patches::Int
    init::I
end

function ViPosEmbedding(embedding_size::Int, number_patches::Int;
                        init=(rng, dims...) -> randn(rng, Float32, dims...))
    return ViPosEmbedding(embedding_size, number_patches, init)
end

function Lux.initialparameters(rng::AbstractRNG, v::ViPosEmbedding)
    return (vectors=v.init(rng, v.embedding_size, v.number_patches),)
end

(v::ViPosEmbedding)(x, ps, st) = x .+ ps.vectors, st

# Helper Functions
"""
    transformer_encoder(in_planes, depth, number_heads; mlp_ratio = 4.0f0, dropout = 0.0f0)

Transformer as used in the base ViT architecture.
([reference](https://arxiv.org/abs/2010.11929)).

## Arguments

  - `in_planes`: number of input channels
  - `depth`: number of attention blocks
  - `number_heads`: number of attention heads
  - `mlp_ratio`: ratio of MLP layers to the number of input channels
  - `dropout_rate`: dropout rate
"""
function transformer_encoder(in_planes, depth, number_heads; mlp_ratio=4.0f0,
                             dropout_rate=0.0f0)
    hidden_planes = floor(Int, mlp_ratio * in_planes)
    layers = [Chain(SkipConnection(Chain(LayerNorm((in_planes, 1); affine=true),
                                         MultiHeadAttention(in_planes, number_heads;
                                                            attention_dropout_rate=dropout_rate,
                                                            projection_dropout_rate=dropout_rate)),
                                   +),
                    SkipConnection(Chain(LayerNorm((in_planes, 1); affine=true),
                                         Chain(Dense(in_planes => hidden_planes, gelu),
                                               Dropout(dropout_rate),
                                               Dense(hidden_planes => in_planes),
                                               Dropout(dropout_rate));
                                         disable_optimizations=true), +)) for _ in 1:depth]
    return Chain(layers...; disable_optimizations=true)
end

function patch_embedding(imsize::Tuple{<:Int, <:Int}=(224, 224); in_channels=3,
                         patch_size::Tuple{<:Int, <:Int}=(16, 16), embed_planes=768,
                         norm_layer=in_planes -> NoOpLayer(), flatten=true)
    im_width, im_height = imsize
    patch_width, patch_height = patch_size

    @assert (im_width % patch_width == 0) && (im_height % patch_height == 0)
    "Image dimensions must be divisible by the patch size."

    return Chain(Conv(patch_size, in_channels => embed_planes; stride=patch_size),
                 flatten ? flatten_spatial : identity, norm_layer(embed_planes))
end

# ViT Implementation
function vision_transformer(; imsize::Tuple{<:Int, <:Int}=(256, 256), in_channels::Int=3,
                            patch_size::Tuple{<:Int, <:Int}=(16, 16), embed_planes::Int=768,
                            depth::Int=6, number_heads=16, mlp_ratio=4.0f0,
                            dropout_rate=0.1f0, embedding_dropout_rate=0.1f0,
                            pool::Symbol=:class, num_classes::Int=1000, kwargs...)
    @assert pool in (:class, :mean) "Pool type must be either :class (class token) or :mean (mean pooling)"
    number_patches = prod(imsize .÷ patch_size)

    return Chain(Chain(patch_embedding(imsize; in_channels, patch_size, embed_planes),
                       ClassTokens(embed_planes),
                       ViPosEmbedding(embed_planes, number_patches + 1),
                       Dropout(embedding_dropout_rate),
                       transformer_encoder(embed_planes, depth, number_heads; mlp_ratio,
                                           dropout_rate),
                       ((pool == :class) ? WrappedFunction(x -> x[:, 1, :]) :
                        WrappedFunction(seconddimmean)); disable_optimizations=true),
                 Chain(LayerNorm((embed_planes,); affine=true),
                       Dense(embed_planes, num_classes, tanh); disable_optimizations=true);
                 disable_optimizations=true)
end

const VIT_CONFIGS = Dict(:tiny => (depth=12, embed_planes=192, number_heads=3),
                         :small => (depth=12, embed_planes=384, number_heads=6),
                         :base => (depth=12, embed_planes=768, number_heads=12),
                         :large => (depth=24, embed_planes=1024, number_heads=16),
                         :huge => (depth=32, embed_planes=1280, number_heads=16),
                         :giant => (depth=40, embed_planes=1408, number_heads=16,
                                    mlp_ratio=48 / 11),
                         :gigantic => (depth=48, embed_planes=1664, number_heads=16,
                                       mlp_ratio=64 / 13))

function vision_transformer(name::Symbol; kwargs...)
    assert_name_present_in(name, keys(VIT_CONFIGS))
    model = vision_transformer(; VIT_CONFIGS[name]..., kwargs...)
    return initialize_model(name, model; kwargs...)
end
