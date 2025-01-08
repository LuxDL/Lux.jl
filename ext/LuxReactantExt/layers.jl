# Avoid unrolling recurrent layers
function Lux.applyrecurrence(
        r::Recurrence{False}, ::Type{<:ReactantDevice}, x::AbstractArray{T, N}, ps,
        st::NamedTuple) where {T, N}
    slice_dim = r.ordering isa Lux.TimeLastIndex ? N : N - 1
    L = size(x, slice_dim)
    colons_before = ntuple(Returns(Colon()), slice_dim - 1)
    colons_after = ntuple(Returns(Colon()), N - slice_dim)

    (out, carry), stₙ = LuxCore.apply(
        r.cell, x[colons_before..., 1, colons_after...], ps, st)
    @trace for i in 2:L
        (out, carry), stₙ = LuxCore.apply(
            r.cell, (x[colons_before..., i, colons_after...], carry), ps, stₙ)
    end
    return out, stₙ
end
