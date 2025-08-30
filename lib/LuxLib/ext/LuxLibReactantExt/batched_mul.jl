function Impl.batched_matmul(
    x::AnyTracedRArray{xT,N}, y::AnyTracedRArray{yT,N}; kwargs...
) where {xT,yT,N}
    return Impl.batched_matmul(
        materialize_traced_array(x), materialize_traced_array(y); kwargs...
    )
end

function Impl.batched_matmul(
    x::TracedRArray{xT,N},
    y::TracedRArray{yT,N};
    lhs_contracting_dim::Int=2,
    rhs_contracting_dim::Int=1,
    lhs_batching_dims::Dims{M}=ntuple(Base.Fix2(+, 2), Val(N - 2)),
    rhs_batching_dims::Dims{M}=ntuple(Base.Fix2(+, 2), Val(N - 2)),
) where {xT,yT,N,M}
    Impl.assert_batched_matmul_checks(
        x, y; lhs_contracting_dim, rhs_contracting_dim, lhs_batching_dims, rhs_batching_dims
    )

    T = promote_type(Reactant.unwrapped_eltype(x), Reactant.unwrapped_eltype(y))

    # TODO: broadcast_to_size for the batching dims

    # z: [batching_dims..., lhs_non_contracting_dim, rhs_non_contracting_dim]
    z = @opcall dot_general(
        T.(x),
        T.(y);
        contracting_dimensions=(Int[lhs_contracting_dim], Int[rhs_contracting_dim]),
        batching_dimensions=(
            collect(Int, lhs_batching_dims), collect(Int, rhs_batching_dims)
        ),
    )
    return permutedims(z, (N - 1, N, 1:(N - 2)...))
end
