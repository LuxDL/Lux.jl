module LuxCoreReactantExt

using LuxCore: AbstractLuxLayer, LuxCore
using Reactant: Reactant

# Avoid tracing though models since it won't contain anything useful
function Reactant.make_tracer(
        seen, @nospecialize(model::AbstractLuxLayer), @nospecialize(path), mode; kwargs...
    )
    return model
end

LuxCore.replicate(rng::Reactant.TracedRNG) = copy(rng)
LuxCore.replicate(rng::Reactant.ConcreteRNG) = copy(rng)

end
