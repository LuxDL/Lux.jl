# Zygote.jl
function gradient(f::F, ::AutoZygote, args...) where {F}
    return map((xᵢ, dxᵢ) -> dxᵢ === nothing || xᵢ isa Number ? CRC.NoTangent() : dxᵢ,
        args, Zygote.gradient(f, args...))
end

# FiniteDiff.jl
function gradient(f::F, ::AutoFiniteDiff, args...) where {F}
    return gradient(f, FD.finite_difference_gradient, args...)
end

# Enzyme.jl
function gradient(f::F, ::AutoEnzyme{Nothing}, args...) where {F}
    return gradient(f, AutoEnzyme(Enzyme.Reverse), args...)
end

function gradient(f::F, ad::AutoEnzyme{<:Enzyme.ReverseMode}, args...) where {F}
    !ENZYME_TESTING_ENABLED &&
        return ntuple(Returns(GradientComputationSkipped()), length(args))

    args_activity = map(args) do x
        needs_gradient(x) && return Enzyme.Duplicated(x, Enzyme.make_zero(x))
        return Enzyme.Const(x)
    end
    Enzyme.autodiff(ad.mode, f, Enzyme.Active, args_activity...)
    return Tuple(map(enumerate(args)) do (i, x)
        needs_gradient(x) && return args_activity[i].dval
        return CRC.NoTangent()
    end)
end

function gradient(::F, ::AutoEnzyme{<:Enzyme.ForwardMode}, args...) where {F}
    return error("AutoEnzyme{ForwardMode} is not supported yet.")
end

# Tracker.jl
function gradient(f::F, ::AutoTracker, args...) where {F}
    counter = 0
    tracked_args = map(args) do x
        if needs_gradient(x)
            counter += 1
            return Functors.fmap(Tracker.param, x; exclude=_tracker_leaf)
        end
        return x
    end

    @assert counter>0 "No tracked arguments found in `gradient(f, AutoTracker, args...)`"
    Tracker.back!(f(tracked_args...))

    return Tuple(map(tracked_args) do x
        needs_gradient(x) && return Functors.fmap(__tracker_grad, x; exclude=_tracker_leaf)
        return CRC.NoTangent()
    end)
end

_tracker_leaf(x) = Functors.isleaf(x)
_tracker_leaf(::AbstractArray) = true

__tracker_grad(x) = Tracker.grad(x)
__tracker_grad(x::ComponentArray) = ComponentArray(__tracker_grad(getdata(x)), getaxes(x))

# ReverseDiff.jl
function gradient(f::F, ::AutoReverseDiff, args...) where {F}
    return gradient(f, ReverseDiff.gradient, args...)
end

# ForwardDiff.jl
function gradient(f::F, ::AutoForwardDiff, args...) where {F}
    return gradient(f, ForwardDiff.gradient, args...)
end

function gradient(f::F, grad_fn::GFN, args...) where {F, GFN <: Function}
    gs = Vector{Any}(undef, length(args))
    for i in 1:length(args)
        _f, x = partial_function(f, i, args...)
        if x isa AbstractArray
            gs[i] = grad_fn(_f, x)
        elseif x isa NamedTuple
            __f, x_flat = flatten_gradient_computable(_f, x)
            gs[i] = x_flat === nothing ? CRC.NoTangent() : NamedTuple(grad_fn(__f, x_flat))
        else
            gs[i] = CRC.NoTangent()
        end
    end
    return Tuple(gs)
end

# Main Functionality to Test Gradient Correctness
"""
    test_gradients(f, args...; skip_backends=[], broken_backends=[], kwargs...)

Test the gradients of `f` with respect to `args` using the specified backends.

| Backend        | ADType              | CPU | GPU | Notes             |
|:-------------- |:------------------- |:--- |:--- |:----------------- |
| Zygote.jl      | `AutoZygote()`      | ✔   | ✔   |                   |
| Tracker.jl     | `AutoTracker()`     | ✔   | ✔   |                   |
| ReverseDiff.jl | `AutoReverseDiff()` | ✔   | ✖   |                   |
| ForwardDiff.jl | `AutoForwardDiff()` | ✔   | ✖   | `len ≤ 100`       |
| FiniteDiff.jl  | `AutoFiniteDiff()`  | ✔   | ✖   | `len ≤ 100`       |
| Enzyme.jl      | `AutoEnzyme()`      | ✔   | ✖   | Only Reverse Mode |

## Arguments

  - `f`: The function to test the gradients of.
  - `args`: The arguments to test the gradients of. Only `AbstractArray`s are considered
    for gradient computation. Gradients wrt all other arguments are assumed to be
    `NoTangent()`.

## Keyword Arguments

  - `skip_backends`: A list of backends to skip.
  - `broken_backends`: A list of backends to treat as broken.
  - `soft_fail`: If `true`, then the test will be recorded as a `soft_fail` test. This
    overrides any `broken` kwargs. Alternatively, a list of backends can be passed to
    `soft_fail` to allow soft_fail tests for only those backends.
  - `kwargs`: Additional keyword arguments to pass to `check_approx`.

## Example

```julia
julia> f(x, y, z) = x .+ sum(abs2, y.t) + sum(y.x.z)

julia> x = (; t=rand(10), x=(z=[2.0],))

julia> test_gradients(f, 1.0, x, nothing)

```
"""
function test_gradients(f, args...; skip_backends=[], broken_backends=[],
        soft_fail::Union{Bool, Vector}=false, kwargs...)
    on_gpu = get_device_type(args) <: AbstractGPUDevice
    total_length = mapreduce(__length, +, Functors.fleaves(args); init=0)

    # Choose the backends to test
    backends = []
    push!(backends, AutoZygote())
    if !on_gpu
        push!(backends, AutoReverseDiff())
        total_length ≤ 100 && push!(backends, AutoForwardDiff())
        total_length ≤ 100 && push!(backends, AutoFiniteDiff())
        # TODO: Move Enzyme out of here once it supports GPUs
        ENZYME_TESTING_ENABLED && push!(backends, AutoEnzyme())
    end
    push!(backends, AutoTracker())

    intersect_backends = intersect(broken_backends, skip_backends)
    if !isempty(intersect_backends)
        throw(ArgumentError("`broken_backends` and `skip_backends` cannot contain the same \
                             backends -- $(intersect_backends)."))
    end

    # Test the gradients
    ∂args_gt = gradient(f, backends[1], args...)  # Should be Zygote in most cases

    @assert (backends[1] ∉ broken_backends)&&(backends[1] ∉ skip_backends) "first backend cannot be broken or skipped"

    @testset "gradtest($(f))" begin
        @testset "$(nameof(typeof(backends[1])))() vs $(nameof(typeof(backend)))()" for backend in backends[2:end]
            if backend in skip_backends
                @test_skip begin
                    ∂args = allow_unstable() do
                        return gradient(f, backend, args...)
                    end
                    check_approx(∂args, ∂args_gt; kwargs...)
                end
            elseif (soft_fail isa Bool && soft_fail) ||
                   (soft_fail isa Vector && backend in soft_fail)
                @test_softfail begin
                    ∂args = allow_unstable() do
                        return gradient(f, backend, args...)
                    end
                    check_approx(∂args, ∂args_gt; kwargs...)
                end
            elseif backend in broken_backends
                @test_broken begin
                    ∂args = allow_unstable() do
                        return gradient(f, backend, args...)
                    end
                    check_approx(∂args, ∂args_gt; kwargs...)
                end
            else
                @test begin
                    ∂args = allow_unstable() do
                        return gradient(f, backend, args...)
                    end
                    check_approx(∂args, ∂args_gt; kwargs...)
                end
            end
        end
    end
end
