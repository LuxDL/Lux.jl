@stable default_mode = "disable" function apply(model::AbstractLuxLayer, x, ps, st)
    return model(x, ps, st)
end

function stateless_apply(model::AbstractLuxLayer, x, ps)
    return first(apply(model, x, ps, Internal.get_empty_state(model)))
end

# New Interface that circumvents having to manage the state manually
