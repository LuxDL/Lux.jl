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
passed in directly without being compiled first. When `single_train_step` or other
functions are called on the `TrainState`, the model will be compiled automatically.

The correct usage is:

```julia
using Lux, Reactant, Random, Optimisers

rdev = reactant_device()

model = Dense(10, 10)
ps, st = Lux.setup(Random.default_rng(), model) |> rdev
x = rand(10) |> rdev

train_state = TrainState(model, ps, st, Adam())
```

The error originates because the model is being compiled first, which is not
supported. **The following is the incorrect way, which potentially causes this
error.**

```julia
using Lux, Reactant, Random, Optimisers

rdev = reactant_device()

model = Dense(10, 10)
ps, st = Lux.setup(Random.default_rng(), model)
x = rand(10) |> rdev

model_compiled = @compile model(x, ps, st)

train_state = Training.TrainState(model_compiled, ps, st, Adam())
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

function annotate_compile(f::F, name::String) where {F}
    id = Profiler.profiler_activity_start(
        "Compile $(name)", Profiler.TRACE_ME_LEVEL_CRITICAL
    )
    res = f()
    Profiler.profiler_activity_end(id)
    return res
end

function annotate_execution(f::F, name::String, step::Int) where {F}
    id = Profiler.profiler_activity_start(
        name, Profiler.TRACE_ME_LEVEL_CRITICAL, "step_num" => step, "_r" => 1
    )
    res = f()
    Profiler.profiler_activity_end(id)
    return res
end

function Lux.Training.compute_gradients_impl(
    backend::ReactantBackend, objective_function::F, data, ts::Training.TrainState
) where {F}
    if (
        ts.cache isa TrainingBackendCache &&
        hasfield(typeof(ts.cache.extras), :compiled_gradient_function)
    )
        compiled_gradient_function = ts.cache.extras.compiled_gradient_function
    else
        compiled_gradient_function = annotate_compile("Compute Gradients") do
            with_default_precision_config(ts.parameters) do
                @compile sync = backend.sync compute_gradients_internal(
                    objective_function, ts.model, data, ts.parameters, ts.states
                )
            end
        end

        if ts.cache isa TrainingBackendCache
            @set! ts.cache.extras = merge(ts.cache.extras, (; compiled_gradient_function))
        else
            cache = TrainingBackendCache(
                backend, False(), nothing, (; compiled_gradient_function)
            )
            @set! ts.cache = cache
        end
        @set! ts.objective_function = objective_function
    end

    grads, loss, stats, st = annotate_execution("Compute Gradients", ts.step) do
        compiled_gradient_function(objective_function, ts.model, data, ts.parameters, ts.states)
    end

    @set! ts.states = st
    return grads, loss, stats, ts
end

for inplace in ("!", "")
    fname = Symbol(:single_train_step_impl, inplace)
    apply_gradients_fn = Symbol(:apply_gradients_reactant, inplace)
    update_fn = Symbol(:update, inplace)

    # Ideally users never hit this dispatch but it is still good to have as a fallback
    @eval function Lux.Training.$(apply_gradients_fn)(
        ts::Training.TrainState{<:TrainingBackendCache{<:ReactantBackend}}, grads
    )
        if (
            ts.cache isa TrainingBackendCache &&
            hasfield(typeof(ts.cache.extras), :update_function)
        )
            update_function = ts.cache.extras.update_function
        else
            update_function = annotate_compile("Apply Gradients") do
                with_default_precision_config(ts.parameters) do
                    @compile sync = ts.cache.backend.sync Optimisers.$(update_fn)(
                        ts.optimizer_state, ts.parameters, grads
                    )
                end
            end

            if ts.cache isa TrainingBackendCache
                @set! ts.cache.extras = merge(ts.cache.extras, (; update_function))
            else
                cache = TrainingBackendCache(backend, False(), nothing, (; update_function))
                @set! ts.cache = cache
            end
        end

        opt_state, ps = annotate_execution("Apply Gradients", ts.step) do
            update_function(ts.optimizer_state, ts.parameters, grads)
        end

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
    @eval function Lux.Training.$(fname)(
        backend::ReactantBackend, objective_function::F, data, ts::Training.TrainState
    ) where {F}
        if (
            ts.cache isa TrainingBackendCache &&
            hasfield(typeof(ts.cache.extras), :compiled_grad_and_step_function)
        )
            (; compiled_grad_and_step_function, is_sharded) = ts.cache.extras
            ps = ts.parameters
            dparameters = ts.cache.dparameters
        else
            device = get_device((ts.parameters, ts.states, ts.optimizer_state, data))
            @assert device isa ReactantDevice
            is_sharded = device.device === nothing

            dparameters = if backend.return_gradients isa True
                Functors.fmap(Utils.zero, ts.parameters; exclude=MLDataDevices.isleaf)
            else
                nothing
            end

            $(ps_expr)

            compiled_grad_and_step_function = annotate_compile("Train Step") do
                with_default_precision_config(ts.parameters) do
                    @compile sync = backend.sync compute_gradients_internal_and_step!(
                        objective_function,
                        ts.model,
                        data,
                        ps,
                        ts.states,
                        ts.optimizer_state,
                        dparameters,
                        is_sharded,
                    )
                end
            end

            if ts.cache isa TrainingBackendCache
                @set! ts.cache.dparameters = dparameters
                @set! ts.cache.extras = merge(
                    ts.cache.extras, (; compiled_grad_and_step_function, is_sharded)
                )
            else
                cache = TrainingBackendCache(
                    backend,
                    False(),
                    dparameters,
                    (; compiled_grad_and_step_function, is_sharded),
                )
                @set! ts.cache = cache
            end
            @set! ts.objective_function = objective_function
        end

        grads, ps, loss, stats, st, opt_state = annotate_execution("Train Step", ts.step) do
            compiled_grad_and_step_function(
                objective_function,
                ts.model,
                data,
                ps,
                ts.states,
                ts.optimizer_state,
                dparameters,
                is_sharded,
            )
        end

        @set! ts.states = st
        @set! ts.parameters = ps
        @set! ts.optimizer_state = opt_state
        @set! ts.step = ts.step + 1

        return grads, loss, stats, ts
    end
end

@eval function compute_gradients_internal_and_step!(
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

@eval function compute_gradients_internal_and_step!(
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

mark_same_sharding_group(args...) = Functors.fmap(mark_same_sharding_group_inner, args...)

function mark_same_sharding_group_inner(arg1::Union{TracedRArray,TracedRNumber}, args...)
    return @opcall sharding_group(arg1, args...)
end
mark_same_sharding_group_inner(arg1, args...) = nothing
