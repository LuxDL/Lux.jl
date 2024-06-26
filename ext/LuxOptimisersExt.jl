module LuxOptimisersExt

using ConcreteStructs: @concrete
using Lux: Lux, DistributedUtils
using .DistributedUtils: AbstractLuxDistributedBackend
using LuxDeviceUtils: AbstractLuxDevice, gpu_device
using Optimisers: Optimisers, AbstractRule, Leaf
using Random: Random
using Setfield: @set!

function Lux.FluxLayer(l)
    p, re = Optimisers.destructure(l)
    return Lux.FluxLayer(l, re, Returns(copy(p)))
end

"""
    TrainState(rng::Random.AbstractRNG, model::Lux.AbstractExplicitLayer,
        optimizer::Optimisers.AbstractRule;
        transform_variables::Union{Function, AbstractLuxDevice}=gpu_device())
    TrainState(model::Lux.AbstractExplicitLayer, ps, st, optimizer::Optimisers.AbstractRule)

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
function Lux.Experimental.TrainState(
        rng::Random.AbstractRNG, model::Lux.AbstractExplicitLayer,
        optimizer::Optimisers.AbstractRule;
        transform_variables::Union{Function, AbstractLuxDevice}=gpu_device())
    ps, st = Lux.setup(rng, model) .|> transform_variables
    return Lux.Experimental.TrainState(model, ps, st, optimizer)
end

function Lux.Experimental.TrainState(
        model::Lux.AbstractExplicitLayer, ps, st, optimizer::Optimisers.AbstractRule)
    st_opt = Optimisers.setup(optimizer, ps)
    return Lux.Experimental.TrainState(
        nothing, nothing, model, ps, st, optimizer, st_opt, 0)
end

function Lux.Experimental.apply_gradients(ts::Lux.Experimental.TrainState, grads)
    optimizer_state, ps = Optimisers.update(ts.optimizer_state, ts.parameters, grads)
    return Lux.Experimental.TrainState(ts.cache, ts.objective_function, ts.model, ps,
        ts.states, ts.optimizer, optimizer_state, ts.step + 1)
end

function Lux.Experimental.apply_gradients!(ts::Lux.Experimental.TrainState, grads)
    Optimisers.update!(ts.optimizer_state, ts.parameters, grads)
    return Lux.Experimental.TrainState(
        ts.cache, ts.objective_function, ts.model, ts.parameters,
        ts.states, ts.optimizer, ts.optimizer_state, ts.step + 1)
end

# Simple extension to the `adjust!` API
function Optimisers.adjust!(ts::Lux.Experimental.TrainState, eta::Real)
    st_opt = ts.optimizer_state
    Optimisers.adjust!(st_opt, eta)
    optimizer = Optimisers.adjust(ts.optimizer, eta)
    return Lux.Experimental.TrainState(ts.cache, ts.objective_function, ts.model,
        ts.parameters, ts.states, optimizer, st_opt, ts.step)
end

function Optimisers.adjust!(ts::Lux.Experimental.TrainState; kwargs...)
    st_opt = ts.optimizer_state
    Optimisers.adjust!(st_opt; kwargs...)
    optimizer = Optimisers.adjust(ts.optimizer; kwargs...)
    return Lux.Experimental.TrainState(ts.cache, ts.objective_function, ts.model,
        ts.parameters, ts.states, optimizer, st_opt, ts.step)
end

function Optimisers.adjust(ts::Lux.Experimental.TrainState, eta::Real)
    st_opt = Optimisers.adjust(ts.optimizer_state, eta)
    optimizer = Optimisers.adjust(ts.optimizer, eta)
    return Lux.Experimental.TrainState(ts.cache, ts.objective_function, ts.model,
        ts.parameters, ts.states, optimizer, st_opt, ts.step)
end

function Optimisers.adjust(ts::Lux.Experimental.TrainState; kwargs...)
    st_opt = Optimisers.adjust(ts.optimizer_state; kwargs...)
    optimizer = Optimisers.adjust(ts.optimizer; kwargs...)
    return Lux.Experimental.TrainState(ts.cache, ts.objective_function, ts.model,
        ts.parameters, ts.states, optimizer, st_opt, ts.step)
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
