module Training

using ADTypes: ADTypes, AutoEnzyme, AutoReverseDiff, AutoTracker, AutoZygote
using Compat: @compat
using ConcreteStructs: @concrete
using FastClosures: @closure
using ..Lux: Lux
using LuxCore: LuxCore, AbstractExplicitLayer
using LuxDeviceUtils: AbstractLuxDevice, gpu_device
using Optimisers: Optimisers
using Random: AbstractRNG

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
    TrainState(rng::Random.AbstractRNG, model::LuxCore.AbstractExplicitLayer,
        optimizer::Optimisers.AbstractRule;
        transform_variables::Union{Function, AbstractLuxDevice}=gpu_device())
    TrainState(model::LuxCore.AbstractExplicitLayer, ps, st,
        optimizer::Optimisers.AbstractRule)

Constructor for [`TrainState`](@ref).

## Arguments

  - `rng`: Random Number Generator.
  - `ps`: Parameters of the model.
  - `st`: States of the model.
  - `model`: `Lux` model.
  - `optimizer`: Optimizer from `Optimisers.jl`.
  - `transform_variables`: Function to transform the variables of the model. Typically used
    to transfer variables to GPU / CPU.

## Returns

[`TrainState`](@ref) object.
"""
function TrainState(
        rng::AbstractRNG, model::AbstractExplicitLayer, optimizer::Optimisers.AbstractRule;
        transform_variables::Union{Function, AbstractLuxDevice}=gpu_device())
    Base.depwarn(
        "`TrainState(rng::AbstractRNG, model::AbstractExplicitLayer, \
         optimizer::Optimisers.AbstractRule; transform_variables::Union{Function, \
         AbstractLuxDevice}=gpu_device())` has been deprecated in favor of \
         `TrainState(model::AbstractExplicitLayer, ps, st, \
         optimizer::Optimisers.AbstractRule)`",
        :TrainState)
    ps, st = LuxCore.setup(rng, model) .|> transform_variables
    return TrainState(model, ps, st, optimizer)
end

function TrainState(
        model::AbstractExplicitLayer, ps, st, optimizer::Optimisers.AbstractRule)
    st_opt = Optimisers.setup(optimizer, ps)
    return TrainState(nothing, nothing, model, ps, st, optimizer, st_opt, 0)
end

@concrete struct TrainingBackendCache{backend, first_try}
    dparameters
    extras
end

training_backend(::TrainingBackendCache{backend}) where {backend} = backend

function Base.show(io::IO, ::MIME"text/plain", ts::TrainState)
    println(io, "TrainState")
    println(io, "    model: ", ts.model)
    println(io, "    # of parameters: ", LuxCore.parameterlength(ts.parameters))
    println(io, "    # of states: ", LuxCore.statelength(ts.states))
    println(io, "    optimizer: ", ts.optimizer)
    print(io, "    step: ", ts.step)
    if ts.cache !== nothing
        if ts.cache isa TrainingBackendCache
            print(io,
                "\n    cache: $(nameof(typeof(ts.cache))){$(training_backend(ts.cache))}")
        else
            print(io, "\n    cache: $(nameof(typeof(ts.cache)))")
        end
    end
    ts.objective_function !== nothing &&
        print(io, "\n    objective_function: ", nameof(typeof(ts.objective_function)))
end

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
    return TrainState(ts.cache, ts.objective_function, ts.model, ps,
        ts.states, ts.optimizer, optimizer_state, ts.step + 1)
end

"""
    apply_gradients!(ts::TrainState, grads)

Update the parameters stored in `ts` using the gradients `grads`. This is an inplace version
of [`apply_gradients`](@ref).

$(APPLY_GRAD_DOCSTRING)
"""
function apply_gradients!(ts::TrainState, grads)
    Optimisers.update!(ts.optimizer_state, ts.parameters, grads)
    return TrainState(ts.cache, ts.objective_function, ts.model, ts.parameters,
        ts.states, ts.optimizer, ts.optimizer_state, ts.step + 1)
end

"""
    compute_gradients(ad::ADTypes.AbstractADType, objective_function::Function, data,
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
function compute_gradients(ad::ADTypes.AbstractADType, ::F, _, ::TrainState) where {F}
    return check_if_compute_gradients_implemented(ad)
end

function check_if_compute_gradients_implemented(::T) where {T <: ADTypes.AbstractADType}
    throw(ArgumentError("Support for AD backend $(nameof(T)) has not been implemented \
                         yet!"))
end

for package in (:Zygote, :Tracker, :ReverseDiff, :Enzyme)
    adtype = Symbol(:Auto, package)
    msg = "Load `$(package)` with `using $(package)`/`import $(package)` before using this \
           function!"
    @eval function check_if_compute_gradients_implemented(::$(adtype))
        throw(ArgumentError($msg))
    end
end

function generate_wrappers(::F, m, ps, st, data, ::Val{false}) where {F}
    @warn "Detected function wrapper generation with function being updated between calls. \
           This will generate type-unstable code. A possible reason for this is \
           `TrainState` was compiled (first call to `compute_gradients`) with function \
           `foo` and is being called with `bar`. A common pattern for this would be \
           passing an anonymous function as `objective_function` inside a loop." maxlog=1
    return Ref{Any}(), Ref{NamedTuple}()
end

# Run the code when trying to compile the function for the first time.
function generate_wrappers(objective_function::F, m, ps, st, data, ::Val{true}) where {F}
    _, stₙ, statsₙ = objective_function(m, ps, st, data)
    return Ref{typeof(stₙ)}(stₙ), Ref{typeof(statsₙ)}(statsₙ)
end

function wrap_objective_function(
        objective_function::F, m, ps, st, data, first_try::Val) where {F}
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
function single_train_step! end

"""
    single_train_step(backend, obj_fn::F, data, ts::TrainState)

Perform a single training step. Computes the gradients using [`compute_gradients`](@ref) and
updates the parameters using [`apply_gradients`](@ref). All backends supported via
[`compute_gradients`](@ref) are supported here.

In most cases you should use [`single_train_step!`](@ref) instead of this function.

## Return

Returned values are the same as [`compute_gradients`](@ref).
"""
function single_train_step end

for inplace in ("!", "")
    step, apply_fn = Symbol(:single_train_step, inplace), Symbol(:apply_gradients, inplace)
    @eval function $(step)(backend, obj_fn::F, data, ts::TrainState) where {F}
        grads, loss, stats, ts = compute_gradients(backend, obj_fn, data, ts)
        ts = $(apply_fn)(ts, grads)
        return grads, loss, stats, ts
    end
end

# Simple extension to the `adjust!` API
function Optimisers.adjust!(ts::TrainState, eta::Real)
    st_opt = ts.optimizer_state
    Optimisers.adjust!(st_opt, eta)
    optimizer = Optimisers.adjust(ts.optimizer, eta)
    return TrainState(ts.cache, ts.objective_function, ts.model,
        ts.parameters, ts.states, optimizer, st_opt, ts.step)
end

function Optimisers.adjust!(ts::TrainState; kwargs...)
    st_opt = ts.optimizer_state
    Optimisers.adjust!(st_opt; kwargs...)
    optimizer = Optimisers.adjust(ts.optimizer; kwargs...)
    return TrainState(ts.cache, ts.objective_function, ts.model,
        ts.parameters, ts.states, optimizer, st_opt, ts.step)
end

function Optimisers.adjust(ts::TrainState, eta::Real)
    st_opt = Optimisers.adjust(ts.optimizer_state, eta)
    optimizer = Optimisers.adjust(ts.optimizer, eta)
    return TrainState(ts.cache, ts.objective_function, ts.model,
        ts.parameters, ts.states, optimizer, st_opt, ts.step)
end

function Optimisers.adjust(ts::TrainState; kwargs...)
    st_opt = Optimisers.adjust(ts.optimizer_state; kwargs...)
    optimizer = Optimisers.adjust(ts.optimizer; kwargs...)
    return TrainState(ts.cache, ts.objective_function, ts.model,
        ts.parameters, ts.states, optimizer, st_opt, ts.step)
end

@compat(public,
    (TrainState, apply_gradients, apply_gradients!,
        compute_gradients, single_train_step, single_train_step!))

export AutoEnzyme, AutoReverseDiff, AutoTracker, AutoZygote

end
