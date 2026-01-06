function Lux.Serialization.export_as_tf_saved_model_internal(
    saved_model_path::String, model::AbstractLuxLayer, x, ps, st
)
    compiled_model = @compile serializable = true model(x, ps, st)

    # get the locations of the model inputs, parameters and states
    input_locations = Union{String,Int}[]

    ## Handle inputs
    cache = Reactant.OrderedIdDict()
    Reactant.make_tracer(cache, x, (), Reactant.ConcreteToTraced)
    append!(input_locations, 1:length(values(cache)))

    ## Handle parameters and states
    cache = Reactant.OrderedIdDict()
    state_dict = Dict()
    Reactant.make_tracer(cache, (ps, st), (), Reactant.ConcreteToTraced)
    for (key, value) in cache
        path = only(value.paths)
        if path[1] == 1
            path_str = __pretty_name(ps, path[2:end])
            path_str = "ps$(path_str)"
        else
            path_str = __pretty_name(st, path[2:end])
            path_str = "st$(path_str)"
        end
        state_dict[path_str] = key
        push!(input_locations, path_str)
    end

    # save the model
    Reactant.Serialization.export_as_tf_saved_model(
        compiled_model, saved_model_path, v"1.8.5", input_locations, state_dict
    )

    return nothing
end

function __pretty_name(x, path)
    path_name = ""
    for p in path
        if p isa Int
            path_name *= "." * string(fieldname(typeof(x), p))
        else
            path_name *= ".$(p)"
        end
        x = Reactant.traced_getfield(x, p)
    end
    return path_name
end
