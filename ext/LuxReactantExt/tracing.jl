# Avoid tracing though models since it won't contain anything useful
function Reactant.make_tracer(
        seen, @nospecialize(model::AbstractLuxLayer), @nospecialize(path), mode; kwargs...
)
    return model
end
