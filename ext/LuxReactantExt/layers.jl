# Embedding
function (e::Lux.Embedding)(x::TracedRNumber{<:Reactant.ReactantInt}, ps, st::NamedTuple)
    return ps.weight[:, x], st
end

# Recurrent Layers
function (r::Lux.Recurrence{False})(x::AnyTracedRArray, ps, st::NamedTuple)
    if r.ordering isa Lux.TimeLastIndex ||
       (r.ordering isa Lux.BatchLastIndex && ndims(x) == 2)
        idxs = ntuple(Returns(Colon()), ndims(x) - 1)
        (out, carry), st = r.cell(x[idxs..., 1], ps, st)
        T = size(x, ndims(x))
        @trace for i in 2:T
            (out, carry), st = r.cell((x[idxs..., i], carry), ps, st)
        end
        return out, st
    elseif r.ordering isa Lux.BatchLastIndex
        idxs = ntuple(Returns(Colon()), ndims(x) - 2)
        (out, carry), st = r.cell(x[idxs..., 1, :], ps, st)
        T = size(x, ndims(x) - 1)
        @trace for i in 2:T
            (out, carry), st = r.cell((x[idxs..., i, :], carry), ps, st)
        end
        return out, st
    else
        error("Unknown ordering: $(r.ordering)")
    end
end

# TODO: We need to implement the return sequence version as well
