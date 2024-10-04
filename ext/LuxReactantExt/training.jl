# TODO: We want to define directly on `single_train_step!` to compile the update function
#       as well.
function Lux.Training.compute_gradients_impl(
        backend::ReactantBackend, objective_function::F,
        data, ts::Training.TrainState) where {F}
    dps = Lux.recursive_make_zero(ts.parameters)

    compiled_gradient_function = @compile compute_gradients_internal(
        objective_function, ts.model, data, ts.parameters, dps, ts.states)

    grads, loss, stats, st = compiled_gradient_function(
        objective_function, ts.model, data, ts.parameters, dps, ts.states)

    cache = TrainingBackendCache(backend, False(), dps, (; compiled_gradient_function))
    @set! ts.cache = cache
    @set! ts.objective_function = objective_function
    @set! ts.states = st
    return grads, loss, stats, ts
end

function Lux.Training.compute_gradients_impl(::ReactantBackend, obj_fn::F, data,
        ts::Training.TrainState{<:TrainingBackendCache{ReactantBackend}, F}) where {F}
    dps = Lux.recursive_make_zero!!(ts.cache.dparameters)

    grads, loss, stats, st = ts.cache.extras.compiled_gradient_function(
        obj_fn, ts.model, data, ts.parameters, dps, ts.states)

    @set! ts.states = st
    return grads, loss, stats, ts
end

function compute_gradients_internal(
        objective_function::F, model, data, ps, dps, st) where {F}
    _, (loss, stₙ, stats) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, Const(objective_function), Active, Const(model),
        Duplicated(ps, dps), Const(st), Const(data))
    return dps, loss, stats, stₙ
end
