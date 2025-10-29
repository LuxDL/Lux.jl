Utils.vec(x::AnyTracedRArray) = ReactantCore.materialize_traced_array(vec(x))

# XXX: Use PoolDims once EnzymeJAX supports stablehlo.reduce_window adjoint
Lux.calculate_pool_dims(g::Lux.GlobalPoolMode, ::TracedRArray) = g

# rsqrt
LuxOps.rsqrt(x::TracedRNumber) = @opcall rsqrt(x)

# convert eltype
function Utils.convert_eltype(
    ::Type{T}, x::Reactant.ConcretePJRTNumber{S}
) where {T<:Number,S}
    return Reactant.ConcretePJRTNumber{T}(x)
end
function Utils.convert_eltype(
    ::Type{T}, x::Reactant.ConcreteIFRTNumber{S}
) where {T<:Number,S}
    return Reactant.ConcreteIFRTNumber{T}(x)
end
