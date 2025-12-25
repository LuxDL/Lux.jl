module Training

using Adapt: Adapt
using ADTypes:
    AbstractADType, AutoEnzyme, AutoReverseDiff, AutoTracker, AutoZygote, AutoMooncake
using SciMLPublic: @public
using ConcreteStructs: @concrete
using FastClosures: @closure
using Functors: Functors, fmap
using Optimisers: Optimisers
using Setfield: @set!
using Static: StaticBool, Static, False, True, static

using ..Lux: Lux, Utils, ReactantCompatibleOptimisers
using LuxCore: LuxCore, AbstractLuxLayer
using MLDataDevices:
    MLDataDevices, AbstractDevice, ReactantDevice, get_device, get_device_type

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
  - `allocator_cache`: Used by GPUArrays compatible backends to cache memory allocations.
  - `objective_function`: Objective function might be cached.

!!! warning

    Constructing this object directly shouldn't be considered a stable API. Use the
    version with the Optimisers API.
"""
@concrete struct TrainState
    cache
    objective_function
    allocator_cache
    model
    parameters
    states
    optimizer
    optimizer_state
    step::Int
end

MLDataDevices.isleaf(::TrainState) = true

function Adapt.adapt_structure(to::AbstractDevice, ts::TrainState)
    return TrainState(
        nothing,
        nothing,
        get_allocator_cache(to),
        ts.model,
        to(ts.parameters),
        to(ts.states),
        ts.optimizer,
        to(ts.optimizer_state),
        ts.step,
    )
end

function Adapt.adapt_structure(to::ReactantDevice, ts::TrainState)
    @warn """
    Moving `TrainState` to `ReactantDevice` might lead to unwanted behaviour.

    Move the `ps` and `st` to the device before constructing the `TrainState`.
    This ensures the optimizer state and other internal states are on the device on
    construction. Prefer using the following style:

    ```julia
    rdev = reactant_device()

    ps, st = Lux.setup(rng, model) |> rdev
    train_state = TrainState(model, ps, st, opt)
    ```

    This warning potentially originates from having `ps` and `st` on the host when
    constructing the `TrainState`, and later moving the `TrainState` to the device.
    **The following is the incorrect way, which potentially causes this warning to
    appear.**

    ```julia
    rdev = reactant_device()

    ps, st = Lux.setup(rng, model)
    train_state = TrainState(model, ps, st, opt)
    train_state = train_state |> rdev
    ```
    """
    return @invoke Adapt.adapt_structure(to::AbstractDevice, ts::TrainState)
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
    dev = get_device(ps)
    if dev isa ReactantDevice
        optimizer = ReactantCompatibleOptimisers.make_reactant_compatible(optimizer, dev)
    end
    st_opt = Optimisers.setup(optimizer, ps)
    return TrainState(
        nothing, nothing, get_allocator_cache(dev), model, ps, st, optimizer, st_opt, 0
    )
end

get_allocator_cache(_) = nothing

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
    println(io, "TrainState(")
    Lux.PrettyPrinting.big_show(io, ts.model, 4)
    println(io, "    number of parameters: ", LuxCore.parameterlength(ts.parameters))
    println(io, "    number of states: ", LuxCore.statelength(ts.states))
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
    return println(io, "\n)")
end

@concrete struct ReactantBackend
    return_gradients <: StaticBool
    sync::Bool
    ad <: AutoEnzyme
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
    if (
        (ts.cache isa TrainingBackendCache && ts.cache.backend isa ReactantBackend) ||
        get_device_type(grads) <: ReactantDevice
    )
        return apply_gradients_reactant(ts, grads)
    end
    return apply_gradients_impl(ts, grads)
end

# apply_gradients -> apply_gradients_reactant (for ReactantBackend)
#                 -> apply_gradients_impl

function apply_gradients_impl(ts::TrainState, grads)
    optimizer_state, ps = Optimisers.update(ts.optimizer_state, ts.parameters, grads)
    @set! ts.parameters = ps
    @set! ts.optimizer_state = optimizer_state
    @set! ts.step = ts.step + 1
    return ts
end

function apply_gradients_reactant end # updated in ReactantExt

"""
    apply_gradients!(ts::TrainState, grads)

Update the parameters stored in `ts` using the gradients `grads`. This is an inplace version
of [`apply_gradients`](@ref).

$(APPLY_GRAD_DOCSTRING)
"""
function apply_gradients!(ts::TrainState, grads)
    if (
        (ts.cache isa TrainingBackendCache && ts.cache.backend isa ReactantBackend) ||
        get_device_type(grads) <: ReactantDevice
    )
        return apply_gradients_reactant!(ts, grads)
    end
    return apply_gradients_with_allocator_cache!(ts.allocator_cache, ts, grads)
end

# apply_gradients! -> apply_gradients_reactant! (for ReactantBackend)
#                  -> apply_gradients_with_allocator_cache! -> apply_gradients_impl!

function apply_gradients_with_allocator_cache!(::Nothing, ts::TrainState, grads)
    return apply_gradients_impl!(ts, grads)
end

function apply_gradients_impl!(ts::TrainState, grads)
    Optimisers.update!(ts.optimizer_state, ts.parameters, grads)
    @set! ts.step = ts.step + 1
    return ts
end

function apply_gradients_reactant! end # updated in ReactantExt

const SYNC_DOCSTRING = """
  - `sync`: If `true`, then the compiled reactant function is compiled with `sync=true`.
    Typically reactant functions are asynchronous, which means if used with profiling or
    for timing, the timing will be inaccurate. Setting `sync=true` will ensure that the
    function will finish execution before this function returns. This is only used for
    Reactant Backend.
"""

"""
    compute_gradients(
        ad::AbstractADType, objective_function::Function, data, ts::TrainState;
        sync::Bool=false
    )

Compute the gradients of the objective function wrt parameters stored in `ts`.

## Backends & AD Packages

| Supported Backends           | Packages Needed  |
|:---------------------------- |:---------------- |
| `AutoZygote`                 | `Zygote.jl`      |
| `AutoReverseDiff(; compile)` | `ReverseDiff.jl` |
| `AutoTracker`                | `Tracker.jl`     |
| `AutoEnzyme`                 | `Enzyme.jl`      |
| `AutoForwardDiff`            |                  |
| `AutoMooncake`               | `Mooncake.jl`    |

## Arguments

  - `ad`: Backend (from [ADTypes.jl](https://github.com/SciML/ADTypes.jl)) used to compute
    the gradients.
  - `objective_function`: Objective function. The function must take 4 inputs -- model,
    parameters, states and data. The function must return 3 values -- loss, updated_state,
    and any computed statistics.
  - `data`: Data used to compute the gradients.
  - `ts`: Current Training State. See [`TrainState`](@ref).

## Keyword Arguments

$(SYNC_DOCSTRING)

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
  - AutoForwardDiff only works with parameters that are AbstractArrays
    (e.g. ps=ComponentVector(ps))

!!! danger "Aliased Gradients"

    `grads` returned by this function might be aliased by the implementation of the gradient
    backend. For example, if you cache the `grads` from step `i`, the new gradients
    returned in step `i + 1` might be aliased by the old gradients. If you want to prevent
    this, simply use `copy(grads)` or `deepcopy(grads)` to make a copy of the gradients.
"""
function compute_gradients(ad, obj_fn::F, data, ts::TrainState; sync::Bool=false) where {F}
    dev_type = get_device_type((ts.parameters, ts.states))
    return compute_gradients_impl_with_allocator_cache(
        maybe_wrap_adtype(ad, dev_type; sync), ts.allocator_cache, obj_fn, data, ts
    )
end

# compute_gradients -> compute_gradients_impl_with_allocator_cache -> compute_gradients_impl

function compute_gradients_impl_with_allocator_cache(
    backend, ::Nothing, obj_fn::F, data, ts::TrainState
) where {F}
    return compute_gradients_impl(backend, obj_fn, data, ts)
end

function compute_gradients_impl(ad, ::F, _, ts::TrainState) where {F}
    return check_if_compute_gradients_implemented(ad)
end

function check_if_compute_gradients_implemented(::T) where {T<:AbstractADType}
    throw(ArgumentError("Support for AD backend $(nameof(T)) has not been implemented \
                         yet!"))
end

function check_if_compute_gradients_implemented(::ReactantBackend)
    throw(
        ArgumentError("Load `Reactant` with `using Reactant` before using this function!")
    )
end

for package in (:Zygote, :Tracker, :ReverseDiff, :Enzyme, :Mooncake)
    adtype = Symbol(:Auto, package)
    msg = "Load `$(package)` with `using $(package)`/`import $(package)` before using this \
           function!"
    @eval function check_if_compute_gradients_implemented(::$(adtype))
        throw(ArgumentError($msg))
    end
end

maybe_wrap_adtype(backend::ReactantBackend, ::Any; kwargs...) = backend
maybe_wrap_adtype(ad::AbstractADType, ::Any; kwargs...) = ad
function maybe_wrap_adtype(
    ad::AbstractADType,
    ::Type{ReactantDevice};
    return_gradients::Utils.BoolType=True(),
    sync::Bool=false,
)
    ad isa AutoEnzyme && return ReactantBackend(static(return_gradients), sync, ad)
    throw(ArgumentError("Computing gradients for models on XLA is supported only with \
                         Enzyme.jl (`AutoEnzyme`)."))
end

function generate_wrappers(::F, m, ps, st, data, ::False, ::StaticBool) where {F}
    @warn "Detected function wrapper generation with function being updated between calls. \
           This will generate type-unstable code. A possible reason for this is \
           `TrainState` was compiled (first call to `compute_gradients`) with function \
           `foo` and is being called with `bar`. A common pattern for this would be \
           passing an anonymous function as `objective_function` inside a loop." maxlog = 1
    return Ref{Any}(), Ref{NamedTuple}()
end

function generate_wrappers(
    objective_function::F, m, ps, st, data, ::True, ::False
) where {F}
    _, stₙ, statsₙ = objective_function(m, ps, st, data)
    return Ref{typeof(stₙ)}(stₙ), Ref{NamedTuple}()  # State type is not preserved
end

# Run the code when trying to compile the function for the first time.
function generate_wrappers(objective_function::F, m, ps, st, data, ::True, ::True) where {F}
    _, stₙ, statsₙ = objective_function(m, ps, st, data)
    return Ref{typeof(stₙ)}(stₙ), Ref{typeof(statsₙ)}(statsₙ)
end

function wrap_objective_function(
    objective_function::F, m, ps, st, data, first_try::StaticBool
) where {F}
    st_updated, stats = generate_wrappers(
        objective_function,
        m,
        ps,
        st,
        data,
        first_try,
        static(LuxCore.preserves_state_type(m)),
    )

    wrapped_objective_function = @closure (model, ps, st, data) -> begin
        loss, st_, stats_ = objective_function(model, ps, st, data)
        Lux.Utils.set_refval!(st_updated, st_)
        Lux.Utils.set_refval!(stats, stats_)
        return loss
    end

    return wrapped_objective_function, st_updated, stats
end

const RETURN_GRADIENTS_DOCSTRING = """
  - `return_gradients`: If `True()`, the gradients are returned. If `False()`, the returned
    gradients are `nothing`. Defaults to `True()`. This is only used for Reactant Backend.
"""

"""
    single_train_step!(
        backend, obj_fn::F, data, ts::TrainState; return_gradients=True(), sync::Bool=false
    )

Perform a single training step. Computes the gradients using [`compute_gradients`](@ref) and
updates the parameters using [`apply_gradients!`](@ref). All backends supported via
[`compute_gradients`](@ref) are supported here.

## Keyword Arguments

$(RETURN_GRADIENTS_DOCSTRING)
$(SYNC_DOCSTRING)

## Return

Returned values are the same as [`compute_gradients`](@ref). Note that despite the `!`,
only the parameters in `ts` are updated inplace. Users should be using the returned `ts`
object for further training steps, else there is no caching and performance will be
suboptimal (and absolutely terrible for backends like `AutoReactant`).
"""
function single_train_step!(
    backend,
    obj_fn::F,
    data,
    ts::TrainState;
    return_gradients::Utils.BoolType=True(),
    sync::Bool=false,
) where {F}
    backend = maybe_wrap_adtype(
        backend, get_device_type((ts.parameters, ts.states)); return_gradients, sync
    )
    return single_train_step_impl_with_allocator_cache!(
        backend, ts.allocator_cache, obj_fn, data, ts
    )
end

"""
    single_train_step(
        backend, obj_fn::F, data, ts::TrainState; return_gradients=True(), sync::Bool=false
    )

Perform a single training step. Computes the gradients using [`compute_gradients`](@ref) and
updates the parameters using [`apply_gradients`](@ref). All backends supported via
[`compute_gradients`](@ref) are supported here.

In most cases you should use [`single_train_step!`](@ref) instead of this function.

## Keyword Arguments

$(RETURN_GRADIENTS_DOCSTRING)
$(SYNC_DOCSTRING)

## Return

Returned values are the same as [`single_train_step!`](@ref).
"""
function single_train_step(
    backend,
    obj_fn::F,
    data,
    ts::TrainState;
    return_gradients::Utils.BoolType=True(),
    sync::Bool=false,
) where {F}
    backend = maybe_wrap_adtype(
        backend, get_device_type((ts.parameters, ts.states)); return_gradients, sync
    )
    return single_train_step_impl(backend, obj_fn, data, ts)
end

# single_train_step -> single_train_step_impl_with_allocator_cache -> single_train_step_impl

function single_train_step_impl_with_allocator_cache!(
    backend, ::Nothing, obj_fn::F, data, ts::TrainState
) where {F}
    return single_train_step_impl!(backend, obj_fn, data, ts)
end

for inplace in ("!", "")
    step = Symbol(:single_train_step_impl, inplace)
    apply_fn = Symbol(:apply_gradients, inplace)

    @eval begin
        function $(step)(backend, obj_fn::F, data, ts::TrainState) where {F}
            grads, loss, stats, ts = compute_gradients(backend, obj_fn, data, ts)
            ts = $(apply_fn)(ts, grads)
            return grads, loss, stats, ts
        end
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

@public (
    TrainState,
    apply_gradients,
    apply_gradients!,
    compute_gradients,
    single_train_step,
    single_train_step!,
)

export AutoEnzyme, AutoReverseDiff, AutoTracker, AutoZygote, AutoMooncake

end
