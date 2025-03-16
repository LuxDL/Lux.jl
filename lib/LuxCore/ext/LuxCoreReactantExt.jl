module LuxCoreReactantExt

using LuxCore: AbstractLuxLayer, LuxCore
using Reactant: Reactant

# Avoid tracing though models since it won't contain anything useful
function Reactant.make_tracer(
    seen, @nospecialize(model::AbstractLuxLayer), @nospecialize(path), mode; kwargs...
)
    return model
end

@static if isdefined(Reactant, :ConcreteIFRTArray)
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
else
    function Reactant.traced_type_inner(
        @nospecialize(T::Type{<:AbstractLuxLayer}),
        seen,
        @nospecialize(mode::Reactant.TraceMode),
        @nospecialize(track_numbers::Type),
        @nospecialize(sharding)
    )
        return T
    end
end

LuxCore.replicate(rng::Reactant.TracedRNG) = copy(rng)
LuxCore.replicate(rng::Reactant.ConcreteRNG) = copy(rng)

end
