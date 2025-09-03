# # A Simple & Minimal Implementation of Mamba

using ConcreteStructs, Lux, Random, Reactant

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

struct ModelArgs
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

function ModelArgs(;
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

    return ModelArgs(
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
    model_args::ModelArgs
    in_proj
    conv1d
    ssm
    out_proj
end

function MambaBlock(args::ModelArgs)
    return MambaBlock(
        args,
        Dense(args.d_model => args.d_inner * 2; use_bias=args.bias),
        Conv(
            (args.d_conv,),
            args.d_inner => args.d_inner,
            swish;
            use_bias=args.conv_bias,
            groups=args.d_inner,
            pad=args.d_conv - 1,
        ),
        SSM(args),
        Dense(args.d_inner => args.d_model; use_bias=args.bias),
    )
end

function (ssm::MambaBlock)(x::AbstractArray{T,3}, ps, st) where {T}
    return error("Not implemented yet")
end

@concrete struct SSM <: AbstractLuxContainerLayer{(:x_proj, :dt_proj)}
    x_proj
    dt_proj
    d_state::Int
    d_inner::Int
end

function SSM(args::ModelArgs)
    return SSM(
        Dense(args.d_inner => args.dt_rank + args.d_state * 2; use_bias=false),
        Dense(args.dt_rank => args.d_inner; use_bias=true),
        args.d_state,
        args.d_inner,
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, mamba::SSM)
    A = repeat(reshape(1:(mamba.model_args.d_state), :, 1), (1, mamba.model_args.d_inner))
    A_log = log.(Float32.(A))
    D = ones(Float32, mamba.model_args.d_inner)
    return (;
        x_proj=LuxCore.initialparameters(rng, mamba.x_proj),
        dt_proj=LuxCore.initialparameters(rng, mamba.dt_proj),
        A_log,
        D,
    )
end

function (ssm::SSM)(x::AbstractArray{T,3}, ps, st) where {T}
    return error("Not implemented yet")
end

@concrete struct ResidualBlock <: AbstractLuxWrapperLayer{:block}
    block
end

function ResidualBlock(args::ModelArgs)
    return ResidualBlock(SkipConnection(Chain(RMSNorm(args.d_model), MambaBlock(args)), +))
end

@concrete struct Mamba <: AbstractLuxContainerLayer{(:embedding, :layers, :norm_f)}
    embedding
    layers
    norm_f
end

function Mamba(args::ModelArgs)
    return Mamba(
        Embedding(args.vocab_size => args.d_model),
        Tuple([MambaBlock(args) for _ in 1:(args.n_layer)]),
        RMSNorm(args.d_model),
    )
end

function (mamba::Mamba)(x::AbstractArray{T,3}, ps, st) where {T}
    return error("Not implemented yet")
end
