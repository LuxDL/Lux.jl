@static if isdefined(Reactant, :ConcreteIFRTArray)
    function Reactant.traced_type_inner(
        @nospecialize(T::Type{StatefulLuxLayer{ST,M,psT,stT}}),
        seen,
        @nospecialize(mode::Reactant.TraceMode),
        @nospecialize(track_numbers::Type),
        @nospecialize(sharding),
        @nospecialize(runtime)
    ) where {ST,M,psT,stT}
        return StatefulLuxLayer{
            ST,
            M,
            Reactant.traced_type(psT, seen, mode, track_numbers, sharding, runtime),
            Reactant.traced_type(stT, seen, mode, track_numbers, sharding, runtime),
        }
    end
else
    function Reactant.traced_type_inner(
        @nospecialize(T::Type{StatefulLuxLayer{ST,M,psT,stT}}),
        seen,
        @nospecialize(mode::Reactant.TraceMode),
        @nospecialize(track_numbers::Type),
        @nospecialize(sharding)
    ) where {ST,M,psT,stT}
        return StatefulLuxLayer{
            ST,
            M,
            Reactant.traced_type(psT, seen, mode, track_numbers, sharding),
            Reactant.traced_type(stT, seen, mode, track_numbers, sharding),
        }
    end
end

function Reactant.make_tracer(
    seen, @nospecialize(model::StatefulLuxLayer), @nospecialize(path), mode; kwargs...
)
    return StatefulLuxLayer(
        model.model,
        Reactant.make_tracer(seen, model.ps, (path..., :ps), mode; kwargs...),
        Reactant.make_tracer(seen, model.st, (path..., :st), mode; kwargs...),
        Reactant.make_tracer(seen, model.st_any, (path..., :st_any), mode; kwargs...),
        model.fixed_state_type,
    )
end
