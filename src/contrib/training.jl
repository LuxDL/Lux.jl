"""
    TrainState

Training State containing:

  - `model`: `Lux` model.
  - `parameters`: Trainable Variables of the `model`.
  - `states`: Non-trainable Variables of the `model`.
  - `optimizer_state`: Optimizer State.
  - `step`: Number of updates of the parameters made.
"""
@concrete struct TrainState
    model
    parameters
    states
    optimizer_state
    step::Int
end

"""
    apply_gradients(ts::TrainState, grads)

Update the parameters stored in `ts` using the gradients `grads`.

## Arguments

  - `ts`: [`TrainState`](@ref) object.
  - `grads`: Gradients of the loss function wrt `ts.params`.

## Returns

Updated [`TrainState`](@ref) object.
"""
function apply_gradients end

"""
    compute_gradients(ad::ADTypes.AbstractADType, objective_function::Function, data,
        ts::TrainState)

Compute the gradients of the objective function wrt parameters stored in `ts`.

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
"""
function compute_gradients(ad::ADTypes.AbstractADType, ::F, _, ::TrainState) where {F}
    return __maybe_implemented_compute_gradients(ad)
end

function __maybe_implemented_compute_gradients(::T) where {T <: ADTypes.AbstractADType}
    throw(ArgumentError(lazy"Support for AD backend $(nameof(T)) has not been implemented yet!!!"))
end

for package in (:Zygote, :Tracker, :ReverseDiff)
    adtype = Symbol(:Auto, package)
    @eval function __maybe_implemented_compute_gradients(::ADTypes.$(adtype))
        throw(ArgumentError(lazy"Load `$(package)` with `using $(package)`/`import $(package)` before using this function!"))
    end
end
