# Embedding
function (e::Lux.Embedding)(x::TracedRNumber{<:Reactant.ReactantInt}, ps, st::NamedTuple)
    return ps.weight[:, x], st
end

# Recurrent Layers
# TODO: Once we can eliminate dead-args in while loop we should remove this case and only
#       use the later function for maintenance purposes.
function (r::Lux.Recurrence{False})(x::AnyTracedRArray, ps, st::NamedTuple)
    if r.ordering isa Lux.TimeLastIndex ||
            (r.ordering isa Lux.BatchLastIndex && ndims(x) == 2)
        idxs = ntuple(Returns(Colon()), ndims(x) - 1)
        (out, carry), st = r.cell(x[idxs..., 1], ps, st)
        @trace for i in 2:size(x, ndims(x))
            (out, carry), st = r.cell((x[idxs..., i], carry), ps, st)
        end
        return out, st
    elseif r.ordering isa Lux.BatchLastIndex
        idxs = ntuple(Returns(Colon()), ndims(x) - 2)
        (out, carry), st = r.cell(x[idxs..., 1, :], ps, st)
        @trace for i in 2:size(x, ndims(x) - 1)
            (out, carry), st = r.cell((x[idxs..., i, :], carry), ps, st)
        end
        return out, st
    else
        error("Unknown ordering: $(r.ordering)")
    end
end

function (r::Lux.Recurrence{True})(x::AnyTracedRArray, ps, st::NamedTuple)
    if r.ordering isa Lux.TimeLastIndex ||
            (r.ordering isa Lux.BatchLastIndex && ndims(x) == 2)
        idxs = ntuple(Returns(Colon()), ndims(x) - 1)
        (out, carry), st = r.cell(x[idxs..., 1], ps, st)
        sequence = similar(out, size(out)..., size(x, ndims(x)))
        sequence[idxs..., 1] .= out
        @trace for i in 2:size(x, ndims(x))
            (out, carry), st = r.cell((x[idxs..., i], carry), ps, st)
            sequence[idxs..., i] = out
        end
    elseif r.ordering isa Lux.BatchLastIndex
        idxs = ntuple(Returns(Colon()), ndims(x) - 2)
        (out, carry), st = r.cell(x[idxs..., 1, :], ps, st)
        sequence = similar(out, size(out)..., size(x, ndims(x) - 1))
        sequence[idxs..., :, 1] .= out
        @trace for i in 2:size(x, ndims(x) - 1)
            (out, carry), st = r.cell((x[idxs..., i, :], carry), ps, st)
            sequence[idxs..., :, i] = out
        end
    else
        error("Unknown ordering: $(r.ordering)")
    end
    return (out, eachslice(sequence; dims = ndims(sequence))), st
end
