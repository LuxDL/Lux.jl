function objective_function_wrapper(objective_function::F, model, ps, st, data) where {F}
    loss, stₙ, stats = objective_function(model, ps, st, data)
    return loss, Reactant.ignore_derivatives(stₙ), Reactant.ignore_derivatives(stats)
end

function compute_gradients_internal(objective_function::F, model, data, ps, st) where {F}
    (_, _, dps, _, _), (loss, stₙ, stats) = Enzyme.gradient(
        Enzyme.set_abi(Enzyme.ReverseWithPrimal, Reactant.ReactantABI),
        Const(objective_function_wrapper),
        Const(objective_function),
        Const(model),
        ps,
        Const(st),
        Const(data),
    )
    return dps, loss, stats, stₙ
end

Profiler.@annotate "Compile Compute Gradients" function Lux.Training.compute_gradients_impl(
    backend::ReactantBackend, objective_function::F, data, ts::Training.TrainState
) where {F}
    compiled_gradient_function = Reactant.with_config(;
        dot_general_precision=PrecisionConfig.HIGH,
        convolution_precision=PrecisionConfig.HIGH,
    ) do
        @compile compute_gradients_internal(
            objective_function, ts.model, data, ts.parameters, ts.states
        )
    end

    grads, loss, stats, st = compiled_gradient_function(
        objective_function, ts.model, data, ts.parameters, ts.states
    )

    cache = TrainingBackendCache(backend, False(), nothing, (; compiled_gradient_function))
    @set! ts.cache = cache
    @set! ts.objective_function = objective_function
    @set! ts.states = st
    return grads, loss, stats, ts
end

Profiler.@annotate "Compute Gradients" function Lux.Training.compute_gradients_impl(
    ::ReactantBackend,
    obj_fn::F,
    data,
    ts::Training.TrainState{<:TrainingBackendCache{<:ReactantBackend},F},
) where {F}
    grads, loss, stats, st = ts.cache.extras.compiled_gradient_function(
        obj_fn, ts.model, data, ts.parameters, ts.states
    )
    @set! ts.states = st
    return grads, loss, stats, ts
end

for inplace in ("!", "")
    fname = Symbol(:single_train_step_impl, inplace)
    internal_fn = Symbol(:compute_gradients_internal_and_step, inplace)
    apply_gradients_fn = Symbol(:apply_gradients, inplace)
    update_fn = Symbol(:update, inplace)

    # Ideally users never hit this dispatch but it is still good to have as a fallback
    @eval Profiler.@annotate "Optimisers Apply Gradients" function Lux.Training.$(
        apply_gradients_fn
    )(
        ts::Training.TrainState{<:TrainingBackendCache{<:ReactantBackend}}, grads
    )
        if hasfield(typeof(ts.cache.extras), :update_function)
            update_function = ts.cache.extras.update_function
        else
            update_function = Reactant.with_config(;
                dot_general_precision=PrecisionConfig.HIGH,
                convolution_precision=PrecisionConfig.HIGH,
            ) do
                @compile Optimisers.$(update_fn)(ts.optimizer_state, ts.parameters, grads)
            end

            @set! ts.cache.extras = merge(ts.cache.extras, (; update_function))
        end

        opt_state, ps = update_function(ts.optimizer_state, ts.parameters, grads)
        @set! ts.parameters = ps
        @set! ts.optimizer_state = opt_state
        @set! ts.step = ts.step + 1
        return ts
    end

    # XXX: recompile with a warning if new input types are used
    @eval Profiler.@annotate "Compile Train Step" function Lux.Training.$(fname)(
        backend::ReactantBackend, objective_function::F, data, ts::Training.TrainState
    ) where {F}
        compiled_grad_and_step_function = Reactant.with_config(;
            dot_general_precision=PrecisionConfig.HIGH,
            convolution_precision=PrecisionConfig.HIGH,
        ) do
            @compile $(internal_fn)(
                objective_function,
                ts.model,
                data,
                ts.parameters,
                ts.states,
                ts.optimizer_state,
                backend.return_gradients,
            )
        end

        grads, ps, loss, stats, st, opt_state = compiled_grad_and_step_function(
            objective_function,
            ts.model,
            data,
            ts.parameters,
            ts.states,
            ts.optimizer_state,
            backend.return_gradients,
        )

        cache = TrainingBackendCache(
            backend, False(), nothing, (; compiled_grad_and_step_function)
        )
        @set! ts.cache = cache
        @set! ts.objective_function = objective_function
        @set! ts.states = st
        @set! ts.parameters = ps
        @set! ts.optimizer_state = opt_state
        @set! ts.step = ts.step + 1

        return grads, loss, stats, ts
    end

    @eval Profiler.@annotate "Train Step" function Lux.Training.$(fname)(
        backend::ReactantBackend,
        obj_fn::F,
        data,
        ts::Training.TrainState{<:TrainingBackendCache{<:ReactantBackend},F},
    ) where {F}
        grads, ps, loss, stats, st, opt_state = ts.cache.extras.compiled_grad_and_step_function(
            obj_fn,
            ts.model,
            data,
            ts.parameters,
            ts.states,
            ts.optimizer_state,
            backend.return_gradients,
        )

        @set! ts.states = st
        @set! ts.parameters = ps
        @set! ts.optimizer_state = opt_state
        @set! ts.step = ts.step + 1

        return grads, loss, stats, ts
    end

    # XXX: Inplace version not actually inplace
    @eval function $(internal_fn)(
        objective_function::F, model, data, ps, st, opt_state, ::False
    ) where {F}
        dps, loss, stats, stₙ = compute_gradients_internal(
            objective_function, model, data, ps, st
        )
        opt_state, ps = Optimisers.$(update_fn)(opt_state, ps, dps)
        return nothing, ps, loss, stats, stₙ, opt_state
    end

    @eval function $(internal_fn)(
        objective_function::F, model, data, ps, st, opt_state, ::True
    ) where {F}
        dps, loss, stats, stₙ = compute_gradients_internal(
            objective_function, model, data, ps, st
        )
        opt_state, ps = Optimisers.$(update_fn)(opt_state, ps, dps)
        return dps, ps, loss, stats, stₙ, opt_state
    end
end
