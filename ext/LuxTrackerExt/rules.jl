# SimpleChains.jl: DON'T REPLACE THESE WITH @grad_from_chainrules
for T1 in (:TrackedArray, :AbstractArray), T2 in (:TrackedArray, :AbstractArray)
    T1 === :AbstractArray && T2 === :AbstractArray && continue

    @eval function Lux.apply_simple_chain(layer, x::$(T1), ps::$(T2), dev::LuxCPUDevice)
        return Tracker.track(Lux.apply_simple_chain, layer, x, ps, dev)
    end
end

Tracker.@grad function Lux.apply_simple_chain(layer, x, ps, ::LuxCPUDevice)
    Base.depwarn("`Tracker.jl` often produces incorrect gradients for `SimpleChains.jl` \
                  models. In future versions of Lux.jl you will need to load `Zygote.jl` \
                  to use `Tracker.jl` for your model.",
        :apply_simple_chain)
    @warn "`Tracker.jl` often produces incorrect gradients for `SimpleChains.jl` models. \
           As such please test your model with `FiniteDiff.jl` or `Zygote.jl` before using \
           `Tracker.jl` for your model." maxlog=1
    y, pb_f = CRC.rrule(layer, Tracker.data(x), Tracker.data(ps))
    __∇apply_simple_chain = let pb_f = pb_f
        Δ -> begin
            _, ∂x, ∂ps = pb_f(convert(Array, Tracker.data(Δ)))
            return Tracker.nobacksies(:apply_simple_chain, (nothing, ∂x, ∂ps, nothing))
        end
    end
    # Tracker is not great at handling arbitrary types, so we convert to Array
    return Array(y), __∇apply_simple_chain
end

# DynamicExpressions.jl
for T1 in (:TrackedArray, :AbstractArray), T2 in (:TrackedArray, :AbstractArray)
    T1 === :AbstractArray && T2 === :AbstractArray && continue

    @eval @grad_from_chainrules Lux.apply_dynamic_expression(
        de::Lux.DynamicExpressionsLayer, expr, operator_enum,
        x::$(T1), ps::$(T2), dev::LuxCPUDevice)
end
