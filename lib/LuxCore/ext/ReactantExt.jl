module ReactantExt

using LuxCore: AbstractLuxLayer, LuxCore
using Reactant: Reactant

# Avoid tracing though models since it won't contain anything useful
function Reactant.make_tracer(
    seen, @nospecialize(model::AbstractLuxLayer), @nospecialize(path), mode; kwargs...
)
    return model
end

function Reactant.traced_type_inner(
    @nospecialize(T::Type{<:AbstractLuxLayer}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    return T
end

LuxCore.replicate(rng::Reactant.ReactantRNG) = copy(rng)

end
