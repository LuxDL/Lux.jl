module LuxComponentArraysExt

using ComponentArrays, Lux

# Empty NamedTuple: Hack to avoid breaking precompilation
function ComponentArrays.ComponentArray(data::Vector{Any}, axes::Tuple{FlatAxis})
    length(data) == 0 && return ComponentArray(Float32[], axes)
    return ComponentArray{Any, 1, typeof(data), typeof(axes)}(data, axes)
end

Lux.__named_tuple(ca::ComponentArray) = NamedTuple(ca)

end
