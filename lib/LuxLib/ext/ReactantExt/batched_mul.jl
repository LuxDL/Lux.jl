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

    x_repeats, y_repeats, _ = Impl.get_batched_matmul_repeat_dims(
        x, y, lhs_batching_dims, rhs_batching_dims
    )
    if !all(==(1), x_repeats)
        x_sz = size(x) .* x_repeats
        x = @opcall broadcast_in_dim(x, collect(Int, 1:N), collect(Int, x_sz))
    end
    if !all(==(1), y_repeats)
        y_sz = size(y) .* y_repeats
        y = @opcall broadcast_in_dim(y, collect(Int, 1:N), collect(Int, y_sz))
    end

    # z: [batching_dims..., lhs_non_contracting_dim, rhs_non_contracting_dim]
    z = @opcall dot_general(
        T.(x),
        T.(y);
        contracting_dimensions=(Int[lhs_contracting_dim], Int[rhs_contracting_dim]),
        batching_dimensions=(
            collect(Int, reverse(lhs_batching_dims)),
            collect(Int, reverse(rhs_batching_dims)),
        ),
    )
    return permutedims(z, (N - 1, N, (N - 2):-1:1...))
end
