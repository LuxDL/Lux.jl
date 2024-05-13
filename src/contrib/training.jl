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
    println(io, "    parameters: ", Lux.parameterlength(ts.parameters))
    println(io, "    states: ", Lux.statelength(ts.states))
    println(io, "    optimizer_state: ", ts.optimizer_state)
    print(io, "    step: ", ts.step)
    ts.cache !== nothing && print(io, "\n    cache: ", nameof(typeof(ts.cache)))
    ts.objective_function !== nothing &&
        print(io, "\n    objective_function: ", nameof(typeof(ts.objective_function)))
end

"""
    apply_gradients(ts::TrainState, grads, update_inplace::Bool=false)

Update the parameters stored in `ts` using the gradients `grads`.

## Arguments

  - `ts`: [`TrainState`](@ref) object.
  - `grads`: Gradients of the loss function wrt `ts.params`.
  - `update_inplace`: Whether to update the parameters inplace or not.

## Returns

Updated [`TrainState`](@ref) object.
"""
function apply_gradients end

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

@inline function __get_st_stat_refs(objective_function::F, model, ps, st, data) where {F}
    ref_types = Core.Compiler._return_type(
        objective_function, Base.typesof(model, ps, st, data))
    ref_types <: Tuple &&
        return Ref{ref_types.parameters[2]}(), Ref{ref_types.parameters[3]}()
    return Ref{Any}(), Ref{Any}()
end

@inline function __wrap_objective_function(
        objective_function::F, model, ps, st, data) where {F}
    st_ref, stats_ref = __get_st_stat_refs(objective_function, model, ps, st, data)

    wrapped_objective_function = let objective_function = objective_function,
        st_ref = st_ref,
        stats_ref = stats_ref

        (model, ps, st, data) -> begin
            y, st, stats = objective_function(model, ps, st, data)
            st_ref[] = st
            stats_ref[] = stats
            return y
        end
    end

    return wrapped_objective_function, st_ref, stats_ref
end
