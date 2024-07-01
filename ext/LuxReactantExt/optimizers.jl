function simple_optimizers_apply!(ps, gs, leaf::Leaf{<:Descent})
    @. ps -= leaf.rule.eta * gs
end

function simple_optimizers_apply!(::Descent, st_opt, ps, gs)
    Lux.recursive_map(simple_optimizers_apply!, ps, gs, st_opt)
end

function simple_optimizers_apply!(opt, st_opt, ps, gs)
    throw(ArgumentError("Optimizer $(typeof(opt)) not yet supported."))
end
