# # Minimal Mamba

using Lux, Random, Reactant

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

d_in, l, n, n = 3, 4, 5, 6
u = randn(Float32, d_in, l, n) |> Reactant.to_rarray;
Δ = randn(Float32, d_in, l, n) |> Reactant.to_rarray;
A = randn(Float32, n, d_in) |> Reactant.to_rarray;
B = randn(Float32, n, l, n) |> Reactant.to_rarray;
C = randn(Float32, n, l, n) |> Reactant.to_rarray;
D = randn(Float32, d_in) |> Reactant.to_rarray;

@code_hlo selective_scan_reference(u, Δ, A, B, C, D)

# function selective_scan(
#     u::AbstractArray{T,3}, ## [d_in, l, b]
#     Δ::AbstractArray{T,3}, ## [d_in, l, b]
#     A::AbstractArray{T,2}, ## [n, d_in]
#     B::AbstractArray{T,3}, ## [n, l, b]
#     C::AbstractArray{T,3}, ## [n, l, b]
#     D::AbstractArray{T,1}, ## [d_in]
# ) where {T}
#     Δ′ = reshape(Δ, 1, size(Δ)...)

#     ## Discretize continuous parameters (A, B)
#     ΔA = exp.(Δ′ .* reshape(A, size(A)..., 1, 1))  ## [n, d_in, l, b]
#     ΔBu = (
#         Δ′ .* reshape(B, size(B, 1), 1, size(B, 2), size(B, 3)) .* reshape(u, 1, size(u)...)
#     ) ## [n, d_in, l, b]

#     TODO: use associative_scan
# end
