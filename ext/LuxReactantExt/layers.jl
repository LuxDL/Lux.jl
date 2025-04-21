# Embedding
function (e::Lux.Embedding)(x::TracedRNumber{<:Reactant.ReactantInt}, ps, st::NamedTuple)
    return ps.weight[:, x], st
end

# Recurrent Layers
function (r::Lux.Recurrence)(x::AnyTracedRArray, ps, st::NamedTuple)
    idxs = ntuple(Returns(Colon()), ndims(x) - 1)
    N = Lux.time_dimension_size(x, r.ordering)

    (out, carry), st = r.cell(Lux.get_time_dimension(x, 1, r.ordering), ps, st)
    sequence = similar(x, size(out)..., N)

    sequence[idxs..., 1] = out
    @trace for i in 2:N
        (out, carry), st = r.cell(Lux.get_time_dimension(x, i, r.ordering), ps, st)
        sequence[idxs..., i] = out
    end

    r.return_sequence isa False && return (out, st)
    return LuxOps.eachslice(sequence, Val(ndims(sequence))), st
end
