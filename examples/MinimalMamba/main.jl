# # A Simple & Minimal Implementation of Mamba

using ConcreteStructs, Lux, Random, Reactant
using HuggingFaceTokenizers, Scratch, PythonCall, JSON3

# Load some python libraries for loading pretrained weights

const huggingface_hub = pyimport("huggingface_hub")
const torch = pyimport("torch")

# ## Selective Scan Algorithm

# Implementation of the selective scan algorithm. First we implement a reference version
# that sequentially goes over the sequence.

function selective_scan_reference(
    u::AbstractArray{T,3}, ## [d_in, l, b]
    Δ::AbstractArray{T,3}, ## [d_in, l, b]
    A::AbstractArray{T,2}, ## [n, d_in]
    B::AbstractArray{T,3}, ## [n, l, b]
    C::AbstractArray{T,3}, ## [n, l, b]
    D::AbstractArray{T,1}, ## [d_in]
) where {T}
    Δ′ = reshape(Δ, 1, size(Δ)...)

    ## Discretize continuous parameters (A, B)
    ΔA = exp.(Δ′ .* reshape(A, size(A)..., 1, 1))  ## [n, d_in, l, b]
    ΔBu = (
        Δ′ .* reshape(B, size(B, 1), 1, size(B, 2), size(B, 3)) .* reshape(u, 1, size(u)...)
    ) ## [n, d_in, l, b]

    ## Perform selective scan with a sequential implementation for correctness verification
    x = fill!(similar(u, size(A, 1), size(u, 1), size(u, 3)), 0) ## [n, d_in, b]
    y = similar(u)
    @trace for i in Int32(1):Int32(size(u, 2))
        @. x = ΔA[:, :, i, :] * x + ΔBu[:, :, i, :]
        tmp = sum(x .* reshape(C[:, i, :], size(C, 1), 1, size(C, 3)); dims=1)
        y[:, i, :] = reshape(tmp, size(u, 1), size(u, 3))
    end
    @. y += u * D

    return y
end

# This trick is based off of https://arxiv.org/abs/2311.06281

function selective_scan_cumsum(
    u::AbstractArray{T,3}, ## [d_in, l, b]
    Δ::AbstractArray{T,3}, ## [d_in, l, b]
    A::AbstractArray{T,2}, ## [n, d_in]
    B::AbstractArray{T,3}, ## [n, l, b]
    C::AbstractArray{T,3}, ## [n, l, b]
    D::AbstractArray{T,1}, ## [d_in]
) where {T}
    Δ′ = reshape(Δ, 1, size(Δ)...)

    ## Discretize continuous parameters (A, B)
    ΔA_log = Δ′ .* reshape(A, size(A)..., 1, 1)  ## [n, d_in, l, b]

    ΔBu = (
        Δ′ .* reshape(B, size(B, 1), 1, size(B, 2), size(B, 3)) .* reshape(u, 1, size(u)...)
    ) ## [n, d_in, l, b]

    ΔA_log[:, :, 1, :] = 0

    log_ΔA_cumsum = cumsum(ΔA_log; dims=3)
    max_log = maximum(log_ΔA_cumsum; dims=3)
    log_ΔA_cumsum_shift = log_ΔA_cumsum .- max_log
    ΔA_cumsum = exp.(log_ΔA_cumsum_shift) .* exp.(max_log)

    x = ΔBu ./ (ΔA_cumsum .+ eltype(ΔA_log)(1e-6))
    x = cumsum(x; dims=3) .* ΔA_cumsum  ## [n, d_in, l, b]

    y = dropdims(
        sum(x .* reshape(C, size(C, 1), 1, size(C, 2), size(C, 3)); dims=1); dims=1
    )
    @. y += u * D

    return y
end

complex_log(x::T) where {T} = complex(log(max(abs(x), eps(T))), (x < 0) * T(pi))

function logaddexp(a, b)
    return ifelse(
        real(a) == -Inf,
        b,
        ifelse(
            real(b) == -Inf,
            a,
            ifelse(real(a) > real(b), a + log(1 + exp(b - a)), b + log(1 + exp(a - b))),
        ),
    )
end

function logcumsumexp(x::AbstractArray{T,4}; dims) where {T}
    return accumulate(logaddexp, x; dims, init=T(-Inf))
end

function compute_lcse_when_positive(ΔBu, ΔA_star)
    ΔBu_log = log.(ΔBu)
    x_log = logcumsumexp(ΔBu_log .- ΔA_star; dims=3) .+ ΔA_star
    return exp.(x_log)
end

function compute_lcse_when_unconfirmed(ΔBu, ΔA_star)
    ΔBu_log = complex_log.(ΔBu)
    x_log = logcumsumexp(ΔBu_log .- ΔA_star; dims=3) .+ ΔA_star
    return exp.(real.(x_log)) .* cos.(imag.(x_log))
end

function selective_scan_logcumsumexp(
    u::AbstractArray{T,3}, ## [d_in, l, b]
    Δ::AbstractArray{T,3}, ## [d_in, l, b]
    A::AbstractArray{T,2}, ## [n, d_in]
    B::AbstractArray{T,3}, ## [n, l, b]
    C::AbstractArray{T,3}, ## [n, l, b]
    D::AbstractArray{T,1}, ## [d_in]
) where {T}
    Δ′ = reshape(Δ, 1, size(Δ)...)

    ## Discretize continuous parameters (A, B)
    ΔA_log = Δ′ .* reshape(A, size(A)..., 1, 1)  ## [n, d_in, l, b]

    ΔBu = (
        Δ′ .* reshape(B, size(B, 1), 1, size(B, 2), size(B, 3)) .* reshape(u, 1, size(u)...)
    ) ## [n, d_in, l, b]

    ΔA_star = pad_zeros(cumsum(ΔA_log[:, :, 2:end, :]; dims=3), (1, 0); dims=3)

    ## adding this check doesn't affect performance much for the negative case
    ## however, we get a dramatic speedup for the all positive case
    ΔBu_positive = all(>(0), ΔBu)
    @trace if ΔBu_positive
        x = compute_lcse_when_positive(ΔBu, ΔA_star)
    else
        x = compute_lcse_when_unconfirmed(ΔBu, ΔA_star)
    end

    y = dropdims(
        sum(x .* reshape(C, size(C, 1), 1, size(C, 2), size(C, 3)); dims=1); dims=1
    )
    @. y += u * D

    return y
end

# TODO: benchmark the associative scan implementation from
# https://github.com/vvvm23/mamba-jax/blob/main/mamba_jax/kernels/reference.py#L8

# ## Mamba Architecture

struct MambaModelArgs
    d_model::Int
    n_layer::Int
    d_inner::Int
    vocab_size::Int
    d_state::Int
    expand::Int
    dt_rank::Int
    d_conv::Int
    pad_vocab_size_multiple::Int
    conv_bias::Bool
    bias::Bool
end

function MambaModelArgs(;
    d_model,
    n_layer,
    vocab_size,
    d_state=16,
    expand=2,
    dt_rank::Union{Int,String}="auto",
    d_conv=4,
    pad_vocab_size_multiple=8,
    conv_bias=true,
    bias=false,
)
    d_inner = Int(expand * d_model)

    if dt_rank isa String
        @assert dt_rank == "auto"
        dt_rank = ceil(Int, d_model / 16)
    end

    if vocab_size % pad_vocab_size_multiple != 0
        vocab_size += (pad_vocab_size_multiple - vocab_size % pad_vocab_size_multiple)
    end

    return MambaModelArgs(
        d_model,
        n_layer,
        d_inner,
        vocab_size,
        d_state,
        expand,
        dt_rank,
        d_conv,
        pad_vocab_size_multiple,
        conv_bias,
        bias,
    )
end

# ### Mamba Block

@concrete struct MambaBlock <:
                 AbstractLuxContainerLayer{(:in_proj, :conv1d, :ssm, :out_proj)}
    in_proj
    conv1d
    ssm
    out_proj
    d_inner::Int
end

function MambaBlock(args::MambaModelArgs)
    return MambaBlock(
        Dense(args.d_model => args.d_inner * 2; use_bias=args.bias),
        Conv(
            (args.d_conv,),
            args.d_inner => args.d_inner,
            swish;
            use_bias=args.conv_bias,
            groups=args.d_inner,
            pad=SamePad(),
        ),
        SSM(args),
        Dense(args.d_inner => args.d_model; use_bias=args.bias),
        args.d_inner,
    )
end

function (block::MambaBlock)(x::AbstractArray{T,3}, ps, st) where {T}
    x_and_res, st_in_proj = block.in_proj(x, ps.in_proj, st.in_proj)

    x = @view x_and_res[1:(block.d_inner), :, :]
    res = @view x_and_res[(block.d_inner + 1):end, :, :]

    x = permutedims(x, (2, 1, 3))  ## l d_in b
    x, st_conv1d = block.conv1d(x, ps.conv1d, st.conv1d)
    x = permutedims(x, (2, 1, 3))  ## d_in l b

    y, st_ssm = block.ssm(x, ps.ssm, st.ssm)
    y = y .* swish.(res)

    output, st_out_proj = block.out_proj(y, ps.out_proj, st.out_proj)

    return (
        output, (; in_proj=st_in_proj, conv1d=st_conv1d, ssm=st_ssm, out_proj=st_out_proj)
    )
end

@concrete struct SSM <: AbstractLuxContainerLayer{(:x_proj, :dt_proj)}
    x_proj
    dt_proj
    d_state::Int
    d_inner::Int
    dt_rank::Int
end

function SSM(args::MambaModelArgs)
    return SSM(
        Dense(args.d_inner => args.dt_rank + args.d_state * 2; use_bias=false),
        Dense(args.dt_rank => args.d_inner, softplus; use_bias=true),
        args.d_state,
        args.d_inner,
        args.dt_rank,
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, ssm::SSM)
    A_log = log.(Float32.(repeat(reshape(1:(ssm.d_state), :, 1); outer=(1, ssm.d_inner))))
    D = ones(Float32, ssm.d_inner)
    return (;
        x_proj=LuxCore.initialparameters(rng, ssm.x_proj),
        dt_proj=LuxCore.initialparameters(rng, ssm.dt_proj),
        A_log,
        D,
    )
end

function (ssm::SSM)(x::AbstractArray{T,3}, ps, st) where {T}
    n, _ = size(ps.A_log)

    A = -exp.(ps.A_log)

    x_dbl, st_x_proj = ssm.x_proj(x, ps.x_proj, st.x_proj)

    Δ = x_dbl[1:(ssm.dt_rank), :, :]
    B = x_dbl[(ssm.dt_rank + 1):(ssm.dt_rank + n), :, :]
    C = x_dbl[(ssm.dt_rank + n + 1):end, :, :]

    Δ, st_dt_proj = ssm.dt_proj(Δ, ps.dt_proj, st.dt_proj)

    y = selective_scan_reference(x, Δ, A, B, C, ps.D)

    return y, (; x_proj=st_x_proj, dt_proj=st_dt_proj)
end

@concrete struct ResidualBlock <: AbstractLuxWrapperLayer{:block}
    block
end

function ResidualBlock(args::MambaModelArgs)
    return ResidualBlock(SkipConnection(Chain(RMSNorm(args.d_model), MambaBlock(args)), +))
end

@concrete struct Mamba <: AbstractLuxContainerLayer{(:embedding, :layers)}
    embedding
    layers
end

function Mamba(args::MambaModelArgs)
    return Mamba(
        Embedding(args.vocab_size => args.d_model),
        Chain(;
            blocks=Chain([ResidualBlock(args) for _ in 1:(args.n_layer)]),
            norm=RMSNorm(args.d_model),
        ),
    )
end

function (mamba::Mamba)(x::AbstractArray{T,2}, ps, st) where {T}
    x, st_embedding = mamba.embedding(x, ps.embedding, st.embedding)
    x, st_layers = mamba.layers(x, ps.layers, st.layers)

    ## Weight-sharing between the embedding layer and the last layer
    sz = size(x)[2:end]
    logits = ps.embedding.weight' * reshape(x, size(x, 1), :)

    return (
        reshape(logits, size(logits, 1), sz...),
        (; embedding=st_embedding, layers=st_layers),
    )
end

# ## Utilities for Loading from HuggingFace

"""
    download_mamba_weights_from_huggingface(pretrained_model_name::String)

Download pretrained weights from HuggingFace Hub.

## Arguments

  - pretrained_model_name: One of
    * 'state-spaces/mamba-2.8b-slimpj'
    * 'state-spaces/mamba-2.8b'
    * 'state-spaces/mamba-1.4b'
    * 'state-spaces/mamba-790m'
    * 'state-spaces/mamba-370m'
    * 'state-spaces/mamba-130m'

## Returns

  - `MambaModelArgs` from huggingface_hub `config.json`.
  - A dictionary containing the pretrained weights.
"""
function download_mamba_weights_from_huggingface(pretrained_model_name::String)
    local_dir = @get_scratch!("lux-mamba-$(replace(pretrained_model_name, "/" => "-"))")

    config_file = huggingface_hub.hf_hub_download(;
        repo_id=pretrained_model_name, filename="config.json", local_dir=local_dir
    )
    config = JSON3.read(read(string(config_file), String))

    mamba_config = MambaModelArgs(;
        d_model=config[:d_model], n_layer=config[:n_layer], vocab_size=config[:vocab_size]
    )

    weights_file = huggingface_hub.hf_hub_download(;
        repo_id=pretrained_model_name, filename="pytorch_model.bin", local_dir=local_dir
    )
    weights = torch.load(weights_file; weights_only=true, mmap=true, map_location="cpu")

    return mamba_config, weights
end

function get_weights_tensor(tensor::PythonCall.Py, ::Type{T}; permute::Bool=false) where {T}
    x = pyconvert(Array{T}, tensor)
    permuted = permute ? permutedims(x, Tuple(reverse(1:ndims(x)))) : x
    return permuted
end

function get_weights_tensor(dict, key, dev, ::Type{T}=Float32; kwargs...) where {T}
    return get_weights_tensor(dict[key], T; kwargs...) |> dev
end

function load_weights_from_dict(weights_dict, config::MambaModelArgs, dev)
    function get_tensor(key; kwargs...)
        return get_weights_tensor(weights_dict, key, dev, Float32; kwargs...)
    end

    embedding = (; weight=get_tensor("backbone.embedding.weight"; permute=true))

    blocks = Vector{Any}(undef, config.n_layer)
    for i in 1:(config.n_layer)
        prefix = "backbone.layers.$(i - 1)"
        mixer_prefix = "$prefix.mixer"

        blocks[i] = (;
            :layer_1 => (; scale=get_tensor("$prefix.norm.weight")),
            :layer_2 => (;
                in_proj=(; weight=get_tensor("$mixer_prefix.in_proj.weight")),
                conv1d=(;
                    weight=get_tensor("$mixer_prefix.conv1d.weight"; permute=true),
                    bias=get_tensor("$mixer_prefix.conv1d.bias"),
                ),
                ssm=(;
                    x_proj=(; weight=get_tensor("$mixer_prefix.x_proj.weight")),
                    dt_proj=(;
                        weight=get_tensor("$mixer_prefix.dt_proj.weight"),
                        bias=get_tensor("$mixer_prefix.dt_proj.bias"),
                    ),
                    A_log=get_tensor("$mixer_prefix.A_log"; permute=true),
                    D=get_tensor("$mixer_prefix.D"; permute=true),
                ),
                out_proj=(; weight=get_tensor("$mixer_prefix.out_proj.weight")),
            ),
        )
    end

    blocks = NamedTuple{Tuple(Symbol("layer_$i") for i in 1:(config.n_layer))}(blocks)

    return (;
        embedding, layers=(; blocks, norm=(; scale=get_tensor("backbone.norm_f.weight")))
    )
end

# ## Tokenizer

struct MambaTokenizer
    tokenizer::Tokenizer
    pad_token_id::Int32
    eos_token_id::Int32
end

const SPLIT_RE = r"(<\|[^>]+?\|>)"

function MambaTokenizer()
    tok = HuggingFaceTokenizers.from_pretrained(Tokenizer, "EleutherAI/gpt-neox-20b")
    return MambaTokenizer(
        tok, token_to_id(tok, "<|endoftext|>"), token_to_id(tok, "<|endoftext|>")
    )
end

token_to_id(tokenizer::MambaTokenizer, s) = token_to_id(tokenizer.tokenizer, s)
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

function HuggingFaceTokenizers.encode(tok::MambaTokenizer, text)
    ids = Int32[]
    for part in filter(!isempty, split_with_delims(text, SPLIT_RE))
        append!(ids, encode(tok.tokenizer, string(part)).ids .+ Int16(1))
    end
    return ids
end

function HuggingFaceTokenizers.decode(tok::MambaTokenizer, ids::Vector{<:Integer})
    return decode(tok.tokenizer, ids .- Int16(1); skip_special_tokens=false)
end

# ## Text Generation Utilities

function weighted_sample(
    rng, items::AbstractVector, weights::AbstractVector, n::Int; temperature::Number=1
)
    @assert length(items) == length(weights)

    weights = weights .^ inv(eltype(weights)(temperature))
    weights = weights ./ sum(weights)
    cumprobs = reshape(cumsum(weights), :, 1)
    random_vals = rand(rng, 1, n)

    indices = dropdims(sum(cumprobs .< random_vals; dims=1); dims=1) .+ 1
    return items[indices]
end

# Setting top_k to 1 will disable sampling and instead return the argmax. For larger
# values of top_k, we sample from the top_k most likely tokens.

function predict_next_token(
    rng,
    model,
    token_ids::AbstractVector{T},
    input_mask_len,
    ps,
    st;
    top_k::Int=32,
    temperature::Number=1,
) where {T}
    token_ids = Reactant.materialize_traced_array(reshape(token_ids, :, 1))

    logits, stₙ = model(token_ids, ps, st)
    next_token_logits = logits[:, end - input_mask_len, 1]

    if top_k == 1
        predictions = T.(argmax(next_token_logits))
    else
        top_k_idxs = partialsortperm(next_token_logits, 1:top_k; rev=true)
        top_k_logits = next_token_logits[Reactant.materialize_traced_array(top_k_idxs)]
        predictions = weighted_sample(rng, T.(top_k_idxs), top_k_logits, 1; temperature)
    end

    predictions = mod1.(predictions, T(size(logits, 1)))
    return predictions, stₙ
end

function update_token_ids_and_mask!(
    padded_token_ids::AbstractVector, input_mask_len, cur_num_tokens, next_token::Number
)
    @trace if input_mask_len == 0
        cur_num_tokens += eltype(cur_num_tokens)(1)
        @allowscalar padded_token_ids[cur_num_tokens] = next_token
    else
        L = length(padded_token_ids)
        padded_token_ids[1:(L - 1)] = padded_token_ids[2:L]
        @allowscalar padded_token_ids[L] = next_token
    end
    return input_mask_len - eltype(input_mask_len)(1), cur_num_tokens
end

function generate_chunk_of_text(
    rng,
    model,
    padded_token_ids,
    input_mask_len,
    cur_num_tokens,
    ps,
    st,
    n_tokens,
    top_k,
    temperature,
)
    next_n_tokens = similar(padded_token_ids, n_tokens)
    @trace track_numbers = false for i in 1:n_tokens
        next_token, st = predict_next_token(
            rng, model, padded_token_ids, input_mask_len, ps, st; top_k, temperature
        )
        next_token_scalar = @allowscalar next_token[1]
        input_mask_len, cur_num_tokens = update_token_ids_and_mask!(
            padded_token_ids, input_mask_len, cur_num_tokens, next_token_scalar
        )
        @allowscalar next_n_tokens[i] = next_token_scalar
    end
    return next_n_tokens, input_mask_len, cur_num_tokens, st
end

function generate_text(
    model::Mamba,
    prompt::String,
    ps,
    st,
    max_new_tokens::Int,
    tokenizer::MambaTokenizer;
    chunk_size::Int=128,
    top_k::Int=32,
    temperature::Number=1,
)
    rdev = reactant_device()

    token_ids = encode(tokenizer, prompt)
    print(decode(tokenizer, token_ids))
    padding_size_to_compile = min(2048, max(length(token_ids) + max_new_tokens, 512))
    if length(token_ids) > padding_size_to_compile
        @warn "Prompt is longer than $(padding_size_to_compile) tokens; truncating to \
               last $(padding_size_to_compile) tokens."
        padded_token_ids = token_ids[(end - padding_size_to_compile + 1):end]
    else
        padded_token_ids = pad_constant(
            token_ids,
            (0, padding_size_to_compile - length(token_ids)),
            eltype(token_ids)(tokenizer.pad_token_id),
        )
        @assert length(padded_token_ids) == padding_size_to_compile
    end
    padded_token_ids = rdev(padded_token_ids)

    rng = Random.default_rng() |> rdev
    cur_num_tokens = ConcreteRNumber(Int32(length(padded_token_ids)))
    input_mask_len = ConcreteRNumber(Int32(padding_size_to_compile - length(token_ids)))

    chunked_text_genfn = @compile generate_chunk_of_text(
        rng,
        model,
        padded_token_ids,
        input_mask_len,
        cur_num_tokens,
        ps,
        st,
        chunk_size,
        top_k,
        temperature,
    )

    n_tokens_generated = 0
    total_time = 0.0
    while n_tokens_generated < max_new_tokens
        start_time = time()
        next_n_tokens, input_mask_len, cur_num_tokens, st = chunked_text_genfn(
            rng,
            model,
            padded_token_ids,
            input_mask_len,
            cur_num_tokens,
            ps,
            st,
            chunk_size,
            top_k,
            temperature,
        )
        total_time += time() - start_time

        n_tokens_generated += length(next_n_tokens)
        next_n_tokens_jl = vec(Array(next_n_tokens))
        for token in next_n_tokens_jl
            token == tokenizer.eos_token_id && return nothing
            print(decode(tokenizer, [token]))
        end
    end
    tokens_per_second = n_tokens_generated / total_time
    println()
    @info "Tokens per second: $(tokens_per_second)"

    return nothing
end

tokenizer = MambaTokenizer()

config, weights_dict = download_mamba_weights_from_huggingface("state-spaces/mamba-130m");
model = Mamba(config)

rdev = reactant_device()

ps_from_hgf = load_weights_from_dict(weights_dict, config, rdev);
st = Lux.initialstates(Random.default_rng(), model) |> rdev;

generate_text(model, "Mamba is the ", ps_from_hgf, st, 128, tokenizer; chunk_size=32)

# x = Int32.(rand(1:(config.vocab_size), 4, 32))
# x = reshape(encode(tok, "Mamba is the "), :, 1)

# res = model(x, ps_from_hgf, st)

# decode(tok, [argmax(res[1][:, end, 1])])

# ## Entry Point

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
