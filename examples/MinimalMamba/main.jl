# # A Simple & Minimal Implementation of Mamba

using ConcreteStructs, Lux, Random, Reactant
using HuggingFaceTokenizers, Scratch, PythonCall, JSON3

## Load some python libraries for loading pretrained weights

const huggingface_hub = pyimport("huggingface_hub")
const torch = pyimport("torch")

# ## Selective Scan Algorithm

# Implementation of the selective scan algorithm. First we implement a reference version
# that sequentially goes over the sequence.

## TODO: Implement the version based on associative_scan for good performance

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
    @trace for i in 1:size(u, 2)
        @. x = ΔA[:, :, i, :] * x + ΔBu[:, :, i, :]
        tmp = batched_matmul(
            x,
            reshape(C[:, i, :], size(C, 1), 1, size(C, 3));
            lhs_contracting_dim=1,
            rhs_contracting_dim=1,
            lhs_batching_dims=(3,),
            rhs_batching_dims=(3,),
        ) ## [d_in, 1, b]
        y[:, i, :] = reshape(tmp, size(u, 1), size(u, 3))
    end
    @. y += u * D

    return y
end

#=
d_in, l, n, n = 3, 4, 5, 6
u = randn(Float32, d_in, l, n) |> Reactant.to_rarray;
Δ = randn(Float32, d_in, l, n) |> Reactant.to_rarray;
A = randn(Float32, n, d_in) |> Reactant.to_rarray;
B = randn(Float32, n, l, n) |> Reactant.to_rarray;
C = randn(Float32, n, l, n) |> Reactant.to_rarray;
D = randn(Float32, d_in) |> Reactant.to_rarray;

@code_hlo selective_scan_reference(u, Δ, A, B, C, D)
=#

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

    Δ = @view x_dbl[1:(ssm.dt_rank), :, :]
    B = @view x_dbl[(ssm.dt_rank + 1):(ssm.dt_rank + n), :, :]
    C = @view x_dbl[(ssm.dt_rank + n + 1):end, :, :]

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

config, weights_dict = download_mamba_weights_from_huggingface("state-spaces/mamba-130m");
model = Mamba(config)

ps_from_hgf = load_weights_from_dict(weights_dict, config, cpu_device());
ps = Lux.initialparameters(Random.default_rng(), model);
st = Lux.initialstates(Random.default_rng(), model);

x = Int32.(rand(1:(config.vocab_size), 4, 32))

model(x, ps_from_hgf, st)
