module Training

using ADTypes: AbstractADType, AutoEnzyme, AutoReverseDiff, AutoTracker, AutoZygote
using Compat: @compat
using ConcreteStructs: @concrete
using FastClosures: @closure
using Functors: Functors, fmap
using Optimisers: Optimisers
using Setfield: @set!
using Static: StaticBool, Static, False, True

using ..Lux: Lux, Utils, ReactantCompatibleOptimisers
using LuxCore: LuxCore, AbstractLuxLayer
using MLDataDevices: MLDataDevices, ReactantDevice, get_device_type, cpu_device

"""
    TrainState

Training State containing:

  - `model`: `Lux` model.
  - `parameters`: Trainable Variables of the `model`.
  - `states`: Non-trainable Variables of the `model`.
  - `optimizer`: Optimizer from `Optimisers.jl`.
  - `optimizer_state`: Optimizer State.
  - `step`: Number of updates of the parameters made.

Internal fields:

  - `cache`: Cached values. Implementations are free to use this for whatever they want.
  - `objective_function`: Objective function might be cached.

!!! warning

    Constructing this object directly shouldn't be considered a stable API. Use the
    version with the Optimisers API.
"""
@concrete struct TrainState
    cache
    objective_function
    model
    parameters
    states
    optimizer
    optimizer_state
    step::Int
end

"""
    TrainState(model::Lux.AbstractLuxLayer, ps, st, optimizer::Optimisers.AbstractRule)

Constructor for [`TrainState`](@ref).

## Arguments

  - `ps`: Parameters of the model.
  - `st`: States of the model.
  - `model`: `Lux` model.
  - `optimizer`: Optimizer from `Optimisers.jl`.

## Returns

[`TrainState`](@ref) object.
"""
function TrainState(model::AbstractLuxLayer, ps, st, optimizer::Optimisers.AbstractRule)
    st_opt = if get_device_type(ps) <: ReactantDevice
        Optimisers.setup(
            ReactantCompatibleOptimisers.make_reactant_compatible(optimizer), ps
        )
    else
        Optimisers.setup(optimizer, ps)
    end
    return TrainState(nothing, nothing, model, ps, st, optimizer, st_opt, 0)
end

@concrete struct TrainingBackendCache
    backend
    first_try <: StaticBool
    dparameters
    extras
end

dparameters(cache::TrainingBackendCache) = dparameters(cache, cache.first_try)
function dparameters(cache::TrainingBackendCache, ::False)
    return fmap(Utils.zero!!, cache.dparameters; exclude=MLDataDevices.isleaf)
end
dparameters(cache::TrainingBackendCache, ::True) = cache.dparameters

function Base.show(io::IO, ::MIME"text/plain", ts::TrainState)
    println(io, "TrainState")
    println(io, "    model: ", ts.model)
    println(io, "    # of parameters: ", LuxCore.parameterlength(ts.parameters))
    println(io, "    # of states: ", LuxCore.statelength(ts.states))
    println(io, "    optimizer: ", ts.optimizer)
    print(io, "    step: ", ts.step)
    if ts.cache !== nothing
        if ts.cache isa TrainingBackendCache
            print(io, "\n    cache: $(nameof(typeof(ts.cache)))($(ts.cache.backend))")
        else
            print(io, "\n    cache: $(nameof(typeof(ts.cache)))")
        end
    end
    ts.objective_function !== nothing &&
        print(io, "\n    objective_function: ", nameof(typeof(ts.objective_function)))
end

struct ReactantBackend end

const APPLY_GRAD_DOCSTRING = """
## Arguments

  - `ts`: [`TrainState`](@ref) object.
  - `grads`: Gradients of the loss function wrt `ts.params`.

## Returns

Updated [`TrainState`](@ref) object.
"""

"""
    apply_gradients(ts::TrainState, grads)

Update the parameters stored in `ts` using the gradients `grads`.

$(APPLY_GRAD_DOCSTRING)
"""
function apply_gradients(ts::TrainState, grads)
    optimizer_state, ps = Optimisers.update(ts.optimizer_state, ts.parameters, grads)
    @set! ts.parameters = ps
    @set! ts.optimizer_state = optimizer_state
    @set! ts.step = ts.step + 1
    return ts
end

"""
    apply_gradients!(ts::TrainState, grads)

Update the parameters stored in `ts` using the gradients `grads`. This is an inplace version
of [`apply_gradients`](@ref).

$(APPLY_GRAD_DOCSTRING)
"""
function apply_gradients!(ts::TrainState, grads)
    Optimisers.update!(ts.optimizer_state, ts.parameters, grads)
    @set! ts.step = ts.step + 1
    return ts
end

"""
    compute_gradients(ad::AbstractADType, objective_function::Function, data,
        ts::TrainState)

Compute the gradients of the objective function wrt parameters stored in `ts`.

## Backends & AD Packages

| Supported Backends           | Packages Needed  |
|:---------------------------- |:---------------- |
| `AutoZygote`                 | `Zygote.jl`      |
| `AutoReverseDiff(; compile)` | `ReverseDiff.jl` |
| `AutoTracker`                | `Tracker.jl`     |
| `AutoEnzyme`                 | `Enzyme.jl`      |

## Arguments

  - `ad`: Backend (from [ADTypes.jl](https://github.com/SciML/ADTypes.jl)) used to compute
    the gradients.
  - `objective_function`: Objective function. The function must take 4 inputs -- model,
    parameters, states and data. The function must return 3 values -- loss, updated_state,
    and any computed statistics.
  - `data`: Data used to compute the gradients.
  - `ts`: Current Training State. See [`TrainState`](@ref).

## Return

A 4-Tuple containing:

  - `grads`: Computed Gradients.
  - `loss`: Loss from the objective function.
  - `stats`: Any computed statistics from the objective function.
  - `ts`: Updated Training State.

## Known Limitations

  - `AutoReverseDiff(; compile=true)` is not supported for Lux models with non-empty state
    `st`. Additionally the returned stats must be empty (`NamedTuple()`). We catch these
    issues in most cases and throw an error.

!!! danger "Aliased Gradients"

    `grads` returned by this function might be aliased by the implementation of the gradient
    backend. For example, if you cache the `grads` from step `i`, the new gradients
    returned in step `i + 1` might be aliased by the old gradients. If you want to prevent
    this, simply use `copy(grads)` or `deepcopy(grads)` to make a copy of the gradients.
"""
function compute_gradients(ad, obj_fn::F, data, ts::TrainState) where {F}
    dev_type = get_device_type((ts.parameters, ts.states))
    return compute_gradients_impl(maybe_wrap_adtype(ad, dev_type), obj_fn, data, ts)
end

maybe_wrap_adtype(backend::ReactantBackend, _) = backend
maybe_wrap_adtype(ad::AbstractADType, _) = ad
function maybe_wrap_adtype(ad::AbstractADType, ::Type{ReactantDevice})
    ad isa AutoEnzyme && return ReactantBackend()
    throw(ArgumentError("Computing gradients for models on XLA is supported only with \
                         Enzyme.jl (`AutoEnzyme`)."))
end

function compute_gradients_impl(ad, ::F, _, ts::TrainState) where {F}
    return check_if_compute_gradients_implemented(ad)
end

function check_if_compute_gradients_implemented(::T) where {T <: AbstractADType}
    throw(ArgumentError("Support for AD backend $(nameof(T)) has not been implemented \
                         yet!"))
end

function check_if_compute_gradients_implemented(::ReactantBackend)
    throw(ArgumentError("Load `Reactant` with `using Reactant` before using this function!"))
end

for package in (:Zygote, :Tracker, :ReverseDiff, :Enzyme)
    adtype = Symbol(:Auto, package)
    msg = "Load `$(package)` with `using $(package)`/`import $(package)` before using this \
           function!"
    @eval function check_if_compute_gradients_implemented(::$(adtype))
        throw(ArgumentError($msg))
    end
end

function generate_wrappers(::F, m, ps, st, data, ::False) where {F}
    @warn "Detected function wrapper generation with function being updated between calls. \
           This will generate type-unstable code. A possible reason for this is \
           `TrainState` was compiled (first call to `compute_gradients`) with function \
           `foo` and is being called with `bar`. A common pattern for this would be \
           passing an anonymous function as `objective_function` inside a loop." maxlog=1
    return Ref{Any}(), Ref{NamedTuple}()
end

# Run the code when trying to compile the function for the first time.
function generate_wrappers(objective_function::F, m, ps, st, data, ::True) where {F}
    _, stₙ, statsₙ = objective_function(m, ps, st, data)
    return Ref{typeof(stₙ)}(stₙ), Ref{typeof(statsₙ)}(statsₙ)
end

function wrap_objective_function(
        objective_function::F, m, ps, st, data, first_try::StaticBool) where {F}
    st_updated, stats = generate_wrappers(objective_function, m, ps, st, data, first_try)

    wrapped_objective_function = @closure (model, ps, st, data) -> begin
        loss, st_, stats_ = objective_function(model, ps, st, data)
        Lux.Utils.set_refval!(st_updated, st_)
        Lux.Utils.set_refval!(stats, stats_)
        return loss
    end

    return wrapped_objective_function, st_updated, stats
end

"""
    single_train_step!(backend, obj_fn::F, data, ts::TrainState)

Perform a single training step. Computes the gradients using [`compute_gradients`](@ref) and
updates the parameters using [`apply_gradients!`](@ref). All backends supported via
[`compute_gradients`](@ref) are supported here.

## Return

Returned values are the same as [`compute_gradients`](@ref). Note that despite the `!`,
only the parameters in `ts` are updated inplace. Users should be using the returned `ts`
object for further training steps, else there is no caching and performance will be
suboptimal (and absolutely terrible for backends like `AutoReactant`).
"""
function single_train_step!(backend, obj_fn::F, data, ts::TrainState) where {F}
    backend = maybe_wrap_adtype(backend, get_device_type((ts.parameters, ts.states)))
    return single_train_step_impl!(backend, obj_fn, data, ts)
end

"""
    single_train_step(backend, obj_fn::F, data, ts::TrainState)

Perform a single training step. Computes the gradients using [`compute_gradients`](@ref) and
updates the parameters using [`apply_gradients`](@ref). All backends supported via
[`compute_gradients`](@ref) are supported here.

In most cases you should use [`single_train_step!`](@ref) instead of this function.

## Return

Returned values are the same as [`compute_gradients`](@ref).
"""
function single_train_step(backend, obj_fn::F, data, ts::TrainState) where {F}
    backend = maybe_wrap_adtype(backend, get_device_type((ts.parameters, ts.states)))
    return single_train_step_impl(backend, obj_fn, data, ts)
end

for inplace in ("!", "")
    step = Symbol(:single_train_step_impl, inplace)
    apply_fn = Symbol(:apply_gradients, inplace)
    @eval function $(step)(backend, obj_fn::F, data, ts::TrainState) where {F}
        grads, loss, stats, ts = compute_gradients(backend, obj_fn, data, ts)
        ts = $(apply_fn)(ts, grads)
        return grads, loss, stats, ts
    end
end

# Simple extension to the `adjust!` API
function Optimisers.adjust!(ts::TrainState, eta::Real)
    Optimisers.adjust!(ts.optimizer_state, eta)
    @set! ts.optimizer = Optimisers.adjust(ts.optimizer, eta)
    return ts
end

function Optimisers.adjust!(ts::TrainState; kwargs...)
    Optimisers.adjust!(ts.optimizer_state; kwargs...)
    @set! ts.optimizer = Optimisers.adjust(ts.optimizer; kwargs...)
    return ts
end

function Optimisers.adjust(ts::TrainState, eta::Real)
    @set! ts.optimizer_state = Optimisers.adjust(ts.optimizer_state, eta)
    @set! ts.optimizer = Optimisers.adjust(ts.optimizer, eta)
    return ts
end

function Optimisers.adjust(ts::TrainState; kwargs...)
    @set! ts.optimizer_state = Optimisers.adjust(ts.optimizer_state; kwargs...)
    @set! ts.optimizer = Optimisers.adjust(ts.optimizer; kwargs...)
    return ts
end

@compat(public,
    (TrainState, apply_gradients, apply_gradients!,
        compute_gradients, single_train_step, single_train_step!))

export AutoEnzyme, AutoReverseDiff, AutoTracker, AutoZygote

end
