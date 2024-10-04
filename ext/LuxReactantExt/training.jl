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

for inplace in ("!", "")
    fname = Symbol(:single_train_step_impl, inplace)
    internal_fn = Symbol(:compute_gradients_internal_and_step, inplace)

    @eval function Lux.Training.$(fname)(backend::ReactantBackend, objective_function::F,
            data, ts::Training.TrainState) where {F}
        dps = Lux.recursive_make_zero(ts.parameters)

        compiled_grad_and_step_function = @compile $(internal_fn)(
            objective_function, ts.model, data, ts.parameters, dps, ts.states,
            ts.optimizer_state)

        grads, ps, loss, stats, st, opt_state = compiled_grad_and_step_function(
            objective_function, ts.model, data, ts.parameters, dps, ts.states,
            ts.optimizer_state)

        cache = TrainingBackendCache(
            backend, False(), dps, (; compiled_grad_and_step_function))
        @set! ts.cache = cache
        @set! ts.objective_function = objective_function
        @set! ts.states = st
        @set! ts.parameters = ps
        @set! ts.optimizer_state = opt_state
        @set! ts.step = ts.step + 1

        return grads, loss, stats, ts
    end

    @eval function Lux.Training.$(fname)(::ReactantBackend, obj_fn::F, data,
            ts::Training.TrainState{<:TrainingBackendCache{ReactantBackend}, F}) where {F}
        dps = Lux.recursive_make_zero!!(ts.cache.dparameters)

        grads, ps, loss, stats, st, opt_state = ts.cache.extras.compiled_grad_and_step_function(
            obj_fn, ts.model, data, ts.parameters, dps, ts.states, ts.optimizer_state)

        @set! ts.states = st
        @set! ts.parameters = ps
        @set! ts.optimizer_state = opt_state
        @set! ts.step = ts.step + 1

        return grads, loss, stats, ts
    end
end

for inplace in ("!", "")
    fname = Symbol(:compute_gradients_internal_and_step, inplace)
    update_fn = Symbol(:update, inplace)
    @eval function $(fname)(objective_function::F, model, data, ps, dps,
            st, opt_state) where {F}
        _, (loss, stₙ, stats) = Enzyme.autodiff(
            Enzyme.ReverseWithPrimal, Const(objective_function), Active, Const(model),
            Duplicated(ps, dps), Const(st), Const(data))
        opt_state, ps = Optimisers.$(update_fn)(opt_state, ps, dps)
        return dps, ps, loss, stats, stₙ, opt_state
    end
end
