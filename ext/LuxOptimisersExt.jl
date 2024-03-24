module LuxOptimisersExt

using Lux: Lux, DistributedUtils
using LuxDeviceUtils: AbstractLuxDevice, gpu_device
using Optimisers: Optimisers
using Random: Random
using Setfield: @set!

"""
    TrainState(rng::Random.AbstractRNG, model::Lux.AbstractExplicitLayer,
        optimizer::Optimisers.AbstractRule;
        transform_variables::Union{Function, AbstractLuxDevice}=gpu_device())

Constructor for [`TrainState`](@ref).

## Arguments

  - `rng`: Random Number Generator.
  - `model`: `Lux` model.
  - `optimizer`: Optimizer from `Optimisers.jl`.
  - `transform_variables`: Function to transform the variables of the model. Typically used
    to transfer variables to GPU / CPU.

## Returns

[`TrainState`](@ref) object.
"""
function Lux.Experimental.TrainState(
        rng::Random.AbstractRNG, model::Lux.AbstractExplicitLayer,
        optimizer::Optimisers.AbstractRule;
        transform_variables::Union{Function, AbstractLuxDevice}=gpu_device())
    ps, st = Lux.setup(rng, model) .|> transform_variables
    st_opt = Optimisers.setup(optimizer, ps)
    return Lux.Experimental.TrainState(model, ps, st, st_opt, 0)
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
function Lux.Experimental.apply_gradients(ts::Lux.Experimental.TrainState, grads)
    optimizer_state, ps = Optimisers.update(ts.optimizer_state, ts.parameters, grads)
    return Lux.Experimental.TrainState(
        ts.model, ps, ts.states, optimizer_state, ts.step + 1)
end

# Distributed Utilities
struct DistributedOptimizer{
    B <: DistributedUtils.AbstractLuxDistributedBackend, O <: Optimisers.AbstractRule} <:
       Optimisers.AbstractRule
    backend::B
    opt::O
end

function Optimisers.apply!(opt::DistributedOptimizer, state, x, y)
    y_avg = allreduce!(opt.backend, y, avg)
    return Optimisers.apply!(opt.opt, state, x, y_avg)
end

Optimisers.init(opt::DistributedOptimizer, x::AbstractArray) = Optimisers.init(opt.opt, x)

function Optimisers._adjust(opt::DistributedOptimizer, nt::NamedTuple)
    return DistributedOptimizer(opt.backend, Optimisers._adjust(opt.opt, nt))
end

function DistributedUtils.synchronize!!(
        backend::DistributedUtils.AbstractLuxDistributedBackend,
        ps::Optimisers.Leaf; root::Int=0)
    @set! ps.state = DistributedUtils.synchronize!!(backend, ps.state; root)
    return ps
end

end
