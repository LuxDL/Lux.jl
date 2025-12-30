# Recurrent Layers
function (r::Lux.Recurrence)(x::AnyTracedRArray, ps, st::NamedTuple)
    idxs = ntuple(Returns(Colon()), ndims(x) - 1)
    N = time_dimension_size(x, r.ordering)

    # execute the first step to get the types
    tmp = get_time_dimension(x, 1, r.ordering)
    carry, _ = init_recurrent_state(r.cell, tmp, ps, st)
    (tmp_result, _), _ = r.cell(tmp, ps, st)

    final_result = similar(tmp_result)
    sequence = similar(tmp_result, size(tmp_result)..., N)
    @trace checkpointing = r.checkpointing mincut = r.mincut for i in 1:N
        (out, carry), st = r.cell((get_time_dimension(x, i, r.ordering), carry), ps, st)
        final_result[idxs...] = out
        sequence[idxs..., i] = out
    end

    r.return_sequence isa False && return (final_result, st)
    return eachslice(sequence; dims=ndims(sequence)), st
end
