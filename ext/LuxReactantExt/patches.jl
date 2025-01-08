Utils.vec(x::AnyTracedRArray) = Reactant.TracedUtils.materialize_traced_array(vec(x))

# XXX: Use PoolDims once EnzymeJAX supports stablehlo.reduce_window adjoint
Lux.calculate_pool_dims(g::Lux.GlobalPoolMode, ::TracedRArray) = g

# Embedding
function (e::Lux.Embedding)(x::TracedRNumber{<:Reactant.ReactantInt}, ps, st::NamedTuple)
    return ps.weight[:, x], st
end
