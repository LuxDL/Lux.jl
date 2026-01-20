function Reactant.traced_type_inner(
    @nospecialize(T::Type{StatefulLuxLayer{ST,M,psT,stT}}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
) where {ST,M,psT,stT}
    return StatefulLuxLayer{
        ST,
        M,
        Reactant.traced_type_inner(psT, seen, mode, track_numbers, ndevices, runtime),
        Reactant.traced_type_inner(stT, seen, mode, track_numbers, ndevices, runtime),
    }
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
