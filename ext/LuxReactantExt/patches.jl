Utils.vec(x::AnyTracedRArray) = ReactantCore.materialize_traced_array(vec(x))

# XXX: Use PoolDims once EnzymeJAX supports stablehlo.reduce_window adjoint
Lux.calculate_pool_dims(g::Lux.GlobalPoolMode, ::TracedRArray) = g

# Optimisers setup
function Lux.Training.optimisers_setup_with_jit(opt, ps)
    return @jit Optimisers.setup(opt, ps)
end
