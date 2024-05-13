module LuxOptimisersExt

using ConcreteStructs: @concrete
using Lux: Lux, DistributedUtils
using .DistributedUtils: AbstractLuxDistributedBackend
using LuxDeviceUtils: AbstractLuxDevice, gpu_device
using Optimisers: Optimisers, AbstractRule, Leaf
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
    return Lux.Experimental.TrainState(nothing, nothing, model, ps, st, st_opt, 0)
end

function Lux.Experimental.apply_gradients(
        ts::Lux.Experimental.TrainState, grads, update_inplace=false)
    if update_inplace
        optimizer_state, ps = Optimisers.update(ts.optimizer_state, ts.parameters, grads)
        return Lux.Experimental.TrainState(ts.cache, ts.objective_function, ts.model,
            ps, ts.states, optimizer_state, ts.step + 1)
    else
        Optimisers.update!(ts.optimizer_state, ts.parameters, grads)
        return Lux.Experimental.TrainState(
            ts.cache, ts.objective_function, ts.model, ts.parameters,
            ts.states, ts.optimizer_state, ts.step + 1)
    end
end

# DistributedUtils
@concrete struct DistributedOptimizer{B <: AbstractLuxDistributedBackend} <: AbstractRule
    backend::B
    opt
end

function Optimisers.apply!(opt::DistributedOptimizer, state, x, y)
    y_avg = DistributedUtils.allreduce!(opt.backend, y, DistributedUtils.avg)
    return Optimisers.apply!(opt.opt, state, x, y_avg)
end

Optimisers.init(opt::DistributedOptimizer, x::AbstractArray) = Optimisers.init(opt.opt, x)

function Optimisers._adjust(opt::DistributedOptimizer, nt::NamedTuple)
    return DistributedOptimizer(opt.backend, Optimisers._adjust(opt.opt, nt))
end

function DistributedUtils.synchronize!!(
        backend::AbstractLuxDistributedBackend, ps::Leaf; root::Int=0)
    @set! ps.state = DistributedUtils.synchronize!!(backend, ps.state; root)
    return ps
end

end
