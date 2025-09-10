@stable default_mode = "disable" function apply(model::AbstractLuxLayer, x, ps, st)
    return model(x, ps, st)
end

function stateless_apply(model::AbstractLuxLayer, x, ps)
    return first(apply(model, x, ps, Internal.get_empty_state(model)))
end

# New Interface that circumvents having to manage the state manually
function (model::AbstractLuxLayer)(x, ps, st)
    xs = x isa Tuple ? x : (x,)
    smodel = StatefulLuxLayerImpl.NamedTupleStatefulLuxLayer(model, ps, st)
    output = apply(typeof(model), smodel, xs...)
    return output, StatefulLuxLayerImpl.get_states_as_namedtuple(smodel)
end

# fallback for wrapped layers
function apply(::Type{<:AbstractLuxWrapperLayer}, model, xs...)
    smodel = only(getfield(model, :smodels))
    return apply(smodel.model, xs, smodel.ps, smodel.st)
end
