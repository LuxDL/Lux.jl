module ReactantExt

using LuxCore: AbstractLuxLayer, LuxCore
using Reactant: Reactant

# Avoid tracing though models since it won't contain anything useful. In `TracedToTypes` mode,
# which builds the key of Reactant's `@trace` call cache, add the model itself, so that layers
# of one type with different settings don't share one traced call.
function Reactant.make_tracer(
    seen, @nospecialize(model::AbstractLuxLayer), @nospecialize(path), mode; kwargs...
)
    if mode == Reactant.TracedToTypes
        push!(path, model)
        return nothing
    end
    return model
end

function Reactant.traced_type_inner(
    @nospecialize(T::Type{<:AbstractLuxLayer}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
)
    return T
end

LuxCore.replicate(rng::Reactant.ReactantRNG) = copy(rng)

end
