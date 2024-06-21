"""
    TrainState

Training State containing:

  - `model`: `Lux` model.
  - `parameters`: Trainable Variables of the `model`.
  - `states`: Non-trainable Variables of the `model`.
  - `optimizer_state`: Optimizer State.
  - `step`: Number of updates of the parameters made.

Internal fields:

  - `cache`: Cached values. Implementations are free to use this for whatever they want.
  - `objective_function`: Objective function might be cached.

!!! warning

    Constructing this object directly shouldn't be considered a stable API. Use the
    version with the Optimisers API.
"""
@concrete struct TrainState{C, F}
    cache::C
    objective_function::F
    model
    parameters
    states
    optimizer_state
    step::Int
end

function Base.show(io::IO, ts::TrainState)
    println(io, "TrainState")
    println(io, "    model: ", ts.model)
    println(io, "    # of parameters: ", Lux.parameterlength(ts.parameters))
    println(io, "    # of states: ", Lux.statelength(ts.states))
    println(io, "    optimizer_state: ", ts.optimizer_state)
    print(io, "    step: ", ts.step)
    ts.cache !== nothing && print(io, "\n    cache: ", nameof(typeof(ts.cache)))
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
function apply_gradients end

"""
    apply_gradients!(ts::TrainState, grads)

Update the parameters stored in `ts` using the gradients `grads`. This is an inplace version
of [`apply_gradients`](@ref).

$(APPLY_GRAD_DOCSTRING)
"""
function apply_gradients! end

"""
    compute_gradients(ad::ADTypes.AbstractADType, objective_function::Function, data,
        ts::TrainState)

Compute the gradients of the objective function wrt parameters stored in `ts`.

## Backends & AD Packages

| Supported Backends | Packages Needed  |
|:------------------ |:---------------- |
| `AutoZygote`       | `Zygote.jl`      |
| `AutoReverseDiff`  | `ReverseDiff.jl` |
| `AutoTracker`      | `Tracker.jl`     |
| `AutoEnzyme`       | `Enzyme.jl`      |

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

## Special Notes on Backends

  - `AutoEnzyme`: `mode` is always ignored and Enzyme ReverseMode is used. The first call
    to `compute_gradients` will be type-unstable. It is recommended to call this function
    once outside of the training loop and use the returned train_state for type stability.
  - `AutoReverseDiff`: `compile` is always ignored and the gradient tape is never compiled.

!!! danger

    `grads` returned by this function might be aliased by the implementation of the gradient
    backend. For example, if you cache the `grads` from step `i`, the new gradients
    returned in step `i + 1` might be aliased by the old gradients. If you want to prevent
    this, simply use `copy(grads)` or `deepcopy(grads)` to make a copy of the gradients.
"""
function compute_gradients(ad::ADTypes.AbstractADType, ::F, _, ::TrainState) where {F}
    return __maybe_implemented_compute_gradients(ad)
end

function __maybe_implemented_compute_gradients(::T) where {T <: ADTypes.AbstractADType}
    throw(ArgumentError(lazy"Support for AD backend $(nameof(T)) has not been implemented yet!!!"))
end

for package in (:Zygote, :Tracker, :ReverseDiff, :Enzyme)
    adtype = Symbol(:Auto, package)
    msg = "Load `$(package)` with `using $(package)`/`import $(package)` before using this \
           function!"
    @eval function __maybe_implemented_compute_gradients(::ADTypes.$(adtype))
        throw(ArgumentError($msg))
    end
end

@inline function __generate_wrappers(
        objective_function::F, m, ps, st, data, ::Val{false}) where {F}
    @warn "Detected function wrapper generation with function being updated between calls. \
           This will generate type-unstable code. A possible reason for this is \
           `TrainState` was compiled (first call to `compute_gradients`) with function \
           `foo` and is being called with `bar`. A common pattern for this would be \
           passing an anonymous function as `objective_function` inside a loop." maxlog=1
    return Ref{Any}(), Ref{NamedTuple}()
end

# Run the code when trying to compile the function for the first time.
@inline function __generate_wrappers(
        objective_function::F, m, ps, st, data, ::Val{true}) where {F}
    _, st_, stats_ = objective_function(m, ps, st, data)
    return Ref{typeof(st_)}(st_), Ref{typeof(stats_)}(stats_)
end

@inline __set_refval!(x, y) = (x[] = y)
CRC.@non_differentiable __set_refval!(::Any...)
EnzymeRules.inactive(::typeof(__set_refval!), ::Any...) = nothing

@inline function __wrap_objective_function(
        objective_function::F, m, ps, st, data, first_try::Val) where {F}
    st_updated, stats = __generate_wrappers(objective_function, m, ps, st, data, first_try)

    wrapped_objective_function = @closure (model, ps, st, data) -> begin
        loss, st_, stats_ = objective_function(model, ps, st, data)
        __set_refval!(st_updated, st_)
        __set_refval!(stats, stats_)
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
