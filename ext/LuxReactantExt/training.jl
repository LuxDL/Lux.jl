# Common mistake that users make is passing in a compiled function
function Lux.Training.TrainState(
    ::Reactant.Compiler.Thunk, ps, st, optimizer::Optimisers.AbstractRule
)
    throw(
        ArgumentError(
            """
Invalid TrainState construction using a compiled function.

`TrainState` is being constructed with a reactant compiled function, i.e. a
`Reactant.Compiler.Thunk`. This is likely a mistake as the model should be
passed in directly without being compiled first.

This is likely originating from the following style of usage:

```julia
using Lux, Reactant, Random, Optimisers

rdev = reactant_device()

model = Dense(10, 10)
ps, st = Lux.setup(Random.default_rng(), model) |> rdev
x = rand(10) |> rdev

model_compiled = @compile model(x, ps, st)

train_state = Training.TrainState(model_compiled, ps, st, Adam())
```

Instead avoid compiling the model and pass it directly to `TrainState`. When
`single_train_step` or other functions are called on the `TrainState`, the
model will be compiled automatically.

```julia
train_state = Training.TrainState(model, ps, st, Adam())
```

For end-to-end usage example refer to the documentation:
<https://lux.csail.mit.edu/stable/manual/compiling_lux_models#compile_lux_model_trainstate>
"""
        ),
    )
end

function objective_function_wrapper(objective_function::F, model, ps, st, data) where {F}
    loss, stₙ, stats = objective_function(model, ps, st, data)
    return loss, Reactant.ignore_derivatives(stₙ), Reactant.ignore_derivatives(stats)
end

function compute_gradients_internal!(
    dps, objective_function::F, model, data, ps, st, zeroed_grads::Bool=false
) where {F}
    zeroed_grads || Enzyme.make_zero!(dps)

    _, (loss, stₙ, stats) = Enzyme.autodiff(
        Enzyme.set_abi(Enzyme.ReverseWithPrimal, Reactant.ReactantABI),
        Const(objective_function_wrapper),
        Const(objective_function),
        Const(model),
        Duplicated(ps, dps),
        Const(st),
        Const(data),
    )
    return dps, loss, stats, stₙ
end

function compute_gradients_internal(objective_function::F, model, data, ps, st) where {F}
    return compute_gradients_internal!(
        Enzyme.make_zero(ps), objective_function, model, data, ps, st, true
    )
end

Profiler.@annotate "Compile Compute Gradients" function Lux.Training.compute_gradients_impl(
    backend::ReactantBackend, objective_function::F, data, ts::Training.TrainState
) where {F}
    compiled_gradient_function = with_default_precision_config(ts.parameters) do
        @compile sync = backend.sync compute_gradients_internal(
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
            update_function = with_default_precision_config(ts.parameters) do
                @compile sync = ts.cache.backend.sync Optimisers.$(update_fn)(
                    ts.optimizer_state, ts.parameters, grads
                )
            end

            @set! ts.cache.extras = merge(ts.cache.extras, (; update_function))
        end

        opt_state, ps = update_function(ts.optimizer_state, ts.parameters, grads)
        @set! ts.parameters = ps
        @set! ts.optimizer_state = opt_state
        @set! ts.step = ts.step + 1
        return ts
    end

    ps_expr = if inplace == "!"
        :(ps = ts.parameters)
    else
        :(ps = Functors.fmap(copy, ts.parameters; exclude=MLDataDevices.isleaf))
    end

    # XXX: recompile with a warning if new input types are used
    @eval Profiler.@annotate "Compile Train Step" function Lux.Training.$(fname)(
        backend::ReactantBackend, objective_function::F, data, ts::Training.TrainState
    ) where {F}
        device = get_device((ts.parameters, ts.states, ts.optimizer_state, data))
        @assert device isa ReactantDevice
        is_sharded = device.device === nothing

        dps = if backend.return_gradients isa True
            Functors.fmap(Utils.zero, ts.parameters; exclude=MLDataDevices.isleaf)
        else
            nothing
        end

        $(ps_expr)

        compiled_grad_and_step_function = with_default_precision_config(ts.parameters) do
            @compile sync = backend.sync $(internal_fn)(
                objective_function,
                ts.model,
                data,
                ps,
                ts.states,
                ts.optimizer_state,
                dps,
                is_sharded,
            )
        end

        grads, ps, loss, stats, st, opt_state = compiled_grad_and_step_function(
            objective_function,
            ts.model,
            data,
            ps,
            ts.states,
            ts.optimizer_state,
            dps,
            is_sharded,
        )

        cache = TrainingBackendCache(
            backend, False(), dps, (; compiled_grad_and_step_function, is_sharded)
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
        ::ReactantBackend,
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
            ts.cache.dparameters,
            ts.cache.extras.is_sharded,
        )

        @set! ts.states = st
        @set! ts.parameters = ps
        @set! ts.optimizer_state = opt_state
        @set! ts.step = ts.step + 1

        return grads, loss, stats, ts
    end

    @eval function $(internal_fn)(
        objective_function::F, model, data, ps, st, opt_state, ::Nothing, is_sharded::Bool
    ) where {F}
        dps, loss, stats, stₙ = compute_gradients_internal(
            objective_function, model, data, ps, st
        )

        opt_state, psₙ = Optimisers.update!(opt_state, ps, dps)
        # Ensure sharding of input and output states are consistent
        is_sharded && mark_same_sharding_group(st, stₙ)

        return nothing, psₙ, loss, stats, stₙ, opt_state
    end

    @eval function $(internal_fn)(
        objective_function::F, model, data, ps, st, opt_state, dps, is_sharded::Bool
    ) where {F}
        dps, loss, stats, stₙ = compute_gradients_internal!(
            dps, objective_function, model, data, ps, st
        )

        opt_state, psₙ = Optimisers.update!(opt_state, ps, dps)
        # Ensure sharding of input and output states are consistent
        is_sharded && mark_same_sharding_group(st, stₙ)

        return dps, psₙ, loss, stats, stₙ, opt_state
    end
end

mark_same_sharding_group(args...) = Functors.fmap(mark_same_sharding_group_inner, args...)

function mark_same_sharding_group_inner(arg1::Union{TracedRArray,TracedRNumber}, args...)
    return @opcall sharding_group(arg1, args...)
end
mark_same_sharding_group_inner(arg1, args...) = nothing
