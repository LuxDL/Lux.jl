# These are meant to be used internally for compiling certain lux optiomization
function simple_optimizers_apply!(ps, gs, leaf::Leaf{<:Descent})
    @. ps -= leaf.rule.eta * gs
end

for opt in (Descent,)
    @eval function simple_optimizers_apply!(::$(opt), st_opt, ps, gs)
        recursive_map(simple_optimizers_apply!, ps, gs, st_opt)
    end
end

function simple_optimizers_apply!(opt, st_opt, ps, gs)
    throw(ArgumentError("Optimizer $(typeof(opt)) not yet supported."))
end
