macro compact(exs...)
    Base.depwarn(
        "Lux.Experimental.@compact` has been promoted out of `Lux.Experimental` and is now \
         available in `Lux`. In other words this has been deprecated and will be removed \
         in v1. Use `Lux.@compact` instead.",
        Symbol("@compact"))
    return Lux.CompactMacroImpl.compact_macro_impl(exs...)
end

Base.@deprecate StatefulLuxLayer(args...; kwargs...) Lux.StatefulLuxLayer(
    args...; kwargs...) false

for f in (:TrainState, :TrainingBackendCache, :single_train_step, :single_train_step!,
    :apply_gradients, :apply_gradients!, :compute_gradients)
    @eval Base.@deprecate $f(args...; kwargs...) Training.$f(args...; kwargs...) false
end
