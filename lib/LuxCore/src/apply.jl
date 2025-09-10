@stable default_mode = "disable" function apply(model::AbstractLuxLayer, x, ps, st)
    return model(x, ps, st)
end

function stateless_apply(model::AbstractLuxLayer, x, ps)
    return first(apply(model, x, ps, Internal.get_empty_state(model)))
end

# New Interface that circumvents having to manage the state manually

# TODO: allow for stateless AbstractLuxLayers as well

function (model::AbstractLuxContainerLayer)(x, ps, st)
    xs = x isa Tuple ? x : (x,)
    smodel = StatefulLuxLayerImpl.NamedTupleStatefulLuxLayer(model, ps, st)
    res = apply(typeof(model), smodel, xs...)
    # return res, smodel.st # TODO: correctly handle the states
    return res
end
