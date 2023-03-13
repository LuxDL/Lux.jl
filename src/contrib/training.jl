module Training

# NOTE(@avik-pal): In the long term this will be pulled out into its own package but
# currently all the dependencies are met by Lux itself.
using ..Lux
using Optimisers, Random, Setfield

"""
    TrainState

Training State containing:

  - `model`: `Lux` model.
  - `parameters`: Trainable Variables of the `model`.
  - `states`: Non-trainable Variables of the `model`.
  - `optimizer_state`: Optimizer State.
  - `step`: Number of updates of the parameters made.
"""
struct TrainState{Ps, St, Ost, M}
    model::M
    parameters::Ps
    states::St
    optimizer_state::Ost
    step::Int
end

"""
    TrainState(rng::Random.AbstractRNG, model::Lux.AbstractExplicitLayer,
               optimizer::Optimisers.AbstractRule; transform_variables::Function=Lux.cpu)

Constructor for `TrainState`.

## Arguments

  - `rng`: Random Number Generator.
  - `model`: `Lux` model.
  - `optimizer`: Optimizer from `Optimisers.jl`.
  - `transform_variables`: Function to transform the variables of the model. Typically used
    to transfer variables to `gpu`/`cpu`.

## Returns

`TrainState` object.
"""
function TrainState(rng::Random.AbstractRNG, model::Lux.AbstractExplicitLayer,
                    optimizer::Optimisers.AbstractRule;
                    transform_variables::Function=Lux.cpu)
    ps, st = Lux.setup(rng, model) .|> transform_variables
    st_opt = Optimisers.setup(optimizer, ps)
    return TrainState(model, ps, st, st_opt, 0)
end

"""
    apply_gradients(ts::TrainState, grads)

Update the parameters stored in `ts` using the gradients `grads`.

## Arguments

  - `ts`: `TrainState` object.
  - `grads`: Gradients of the loss function wrt `ts.params`.

## Returns

Updated `TrainState` object.
"""
function apply_gradients(ts::TrainState, grads)
    optimizer_state, parameters = Optimisers.update(ts.optimizer_state, ts.parameters,
                                                    grads)
    return TrainState(ts.model, parameters, ts.states, optimizer_state, ts.step + 1)
end

# VJPs
"""
    AbstractVJP

Base Type for all Vector-Jacobian Product Backends.
"""
abstract type AbstractVJP end

"""
    backend(::AbstractVJP)

Package used to compute the VJP.
"""
function backend(::T) where {T <: AbstractVJP}
    throw(ArgumentError("`backend` function must be defined for type $T"))
end

"""
    compute_gradients(vjp::AbstractVJP, objective_function::Function, data, ts::TrainState)

Compute the gradients of the objective function wrt parameters stored in `ts`.

## Arguments

  - `vjp`: Backend used to compute the gradients. See [`AbstractVJP`](@ref).
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
function compute_gradients(t::T, objective_function::Function, data,
                           ts::TrainState) where {T <: AbstractVJP}
    throw(ArgumentError("Support for AD backend $(backend(t)) has not been implemented
                         yet!!!"))
end

"""
    ZygoteVJP <: AbstractVJP

Vector-Jacobian Product using Zygote.
"""
struct ZygoteVJP <: AbstractVJP end

backend(::ZygoteVJP) = :Zygote

"""
    EnzymeVJP <: AbstractVJP

Vector-Jacobian Product using Enzyme.
"""
struct EnzymeVJP <: AbstractVJP end

backend(::EnzymeVJP) = :Enzyme

"""
    YotaVJP <: AbstractVJP

Vector-Jacobian Product using Yota.
"""
struct YotaVJP <: AbstractVJP end

backend(::YotaVJP) = :Yota

"""
    TrackerVJP <: AbstractVJP

Vector-Jacobian Product using Tracker.
"""
struct TrackerVJP <: AbstractVJP end

backend(::TrackerVJP) = :Tracker

end
