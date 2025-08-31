struct Constant{T}
    val::T
end

# Zygote.jl on CPU
function ground_truth_gradient(f, args...)
    cdev = cpu_device()
    f_cpu = try
        cdev(f)
    catch err
        @error "Encountered error while moving $(f) to CPU. Skipping movement... This can \
                be fixed by defining overloads using ConstructionBase.jl" err
        f
    end
    return gradient(f_cpu, AutoZygote(), map(cdev, args)...)
end

# Zygote.jl
function gradient(f::F, ::AutoZygote, args...) where {F}
    return gradient(f, only ∘ Zygote.gradient, args...)
end

# FiniteDiff.jl
function gradient(f::F, ::AutoFiniteDiff, args...) where {F}
    return gradient(f, FD.finite_difference_gradient, args...)
end

# Enzyme.jl
function gradient(f::F, ::AutoEnzyme{Nothing}, args...) where {F}
    return gradient(f, AutoEnzyme(; mode=Enzyme.Reverse), args...)
end

function gradient(f::F, ad::AutoEnzyme{<:Enzyme.ReverseMode}, args...) where {F}
    !ENZYME_TESTING_ENABLED &&
        return ntuple(Returns(GradientComputationSkipped()), length(args))

    args_activity = map(args) do x
        needs_gradient(x) && return Enzyme.Duplicated(x, Enzyme.make_zero(x))
        x isa Constant && return Enzyme.Const(x.val)
        return Enzyme.Const(x)
    end
    Enzyme.autodiff(ad.mode, Enzyme.Const(f), Enzyme.Active, args_activity...)
    return Tuple(
        map(enumerate(args)) do (i, x)
            needs_gradient(x) && return args_activity[i].dval
            return CRC.NoTangent()
        end,
    )
end

function gradient(::F, ::AutoEnzyme{<:Enzyme.ForwardMode}, args...) where {F}
    return error("AutoEnzyme{ForwardMode} is not supported yet.")
end

# ForwardDiff.jl
function gradient(f::F, ::AutoForwardDiff, args...) where {F}
    return gradient(f, ForwardDiff.gradient, args...)
end

function gradient(f::F, grad_fn::GFN, args...) where {F,GFN<:Function}
    gs = Vector{Any}(undef, length(args))
    for i in 1:length(args)
        _f, x = partial_function(f, i, args...)
        if x isa AbstractArray{<:AbstractFloat}
            gs[i] = grad_fn(_f, x)
        elseif x isa Constant
            gs[i] = CRC.NoTangent()
        elseif x isa NamedTuple || x isa Tuple
            __f, x_flat, re = flatten_gradient_computable(_f, x)
            gs[i] = x_flat === nothing ? CRC.NoTangent() : re(grad_fn(__f, x_flat))
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
| ForwardDiff.jl | `AutoForwardDiff()` | ✔   | ✖   | `len ≤ 32`        |
| FiniteDiff.jl  | `AutoFiniteDiff()`  | ✔   | ✖   | `len ≤ 32`        |
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
  - `enzyme_set_runtime_activity`: If `true`, then activate runtime activity for Enzyme.
  - `enable_enzyme_reverse_mode`: If `true`, then enable reverse mode for Enzyme.
  - `kwargs`: Additional keyword arguments to pass to `check_approx`.

## Example

```julia
julia> f(x, y, z) = x .+ sum(abs2, y.t) + sum(y.x.z)

julia> x = (; t=rand(10), x=(z=[2.0],))

julia> test_gradients(f, 1.0, x, nothing)

```
"""
function test_gradients(
    f,
    args...;
    skip_backends=[],
    broken_backends=[],
    soft_fail::Union{Bool,Vector}=false,
    enzyme_set_runtime_activity::Bool=false,
    enable_enzyme_reverse_mode::Bool=false,
    # Internal kwargs start
    source::LineNumberNode=LineNumberNode(0, nothing),
    test_expr::Expr=:(check_approx(∂args, ∂args_gt; kwargs...)),
    # Internal kwargs end
    kwargs...,
)
    bad_skip_backend = findfirst(removed_backend, skip_backends)
    bad_skip_backend === nothing || throw(
        ArgumentError("skip_backends cannot contain $(skip_backends[bad_skip_backend])")
    )
    bad_broken_backend = findfirst(removed_backend, broken_backends)
    bad_broken_backend === nothing || throw(
        ArgumentError(
            "broken_backends cannot contain $(broken_backends[bad_broken_backend])"
        ),
    )

    if !(soft_fail isa Bool)
        bad_soft_fail = findfirst(removed_backend, soft_fail)
        bad_soft_fail === nothing ||
            throw(ArgumentError("soft_fail cannot contain $(soft_fail[bad_soft_fail])"))
    end

    on_gpu = get_device_type(args) <: AbstractGPUDevice
    total_length = mapreduce(__length, +, Functors.fleaves(args); init=0)

    # Choose the backends to test
    backends = []
    push!(backends, AutoZygote())
    if !on_gpu
        total_length ≤ 32 && push!(backends, AutoForwardDiff())
        total_length ≤ 32 && push!(backends, AutoFiniteDiff())
        # TODO: Move Enzyme out of here once it supports GPUs
        if enable_enzyme_reverse_mode || ENZYME_TESTING_ENABLED
            mode = if enzyme_set_runtime_activity
                Enzyme.set_runtime_activity(Enzyme.Reverse)
            else
                Enzyme.Reverse
            end
            push!(backends, AutoEnzyme(; mode))
        end
    end

    intersect_backends = intersect(broken_backends, skip_backends)
    if !isempty(intersect_backends)
        throw(
            ArgumentError("`broken_backends` and `skip_backends` cannot contain the same \
                           backends -- $(intersect_backends).")
        )
    end

    # Test the gradients
    ∂args_gt = ground_truth_gradient(f, args...)  # Should be Zygote in most cases

    @assert (backends[1] ∉ broken_backends) && (backends[1] ∉ skip_backends) "first backend cannot be broken or skipped"

    return @testset "gradtest($(f))" begin
        @testset "$(nameof(typeof(backend)))()" for backend in backends
            local_test_expr = :([$(nameof(typeof(backend)))] - $(test_expr))

            result = if check_ad_backend_in(backend, skip_backends)
                Broken(:skipped, local_test_expr)
            elseif (soft_fail isa Bool && soft_fail) ||
                (soft_fail isa Vector && check_ad_backend_in(backend, soft_fail))
                try
                    ∂args = allow_unstable() do
                        return gradient(f, backend, args...)
                    end
                    matched = check_approx(∂args, ∂args_gt; kwargs...)
                    if matched
                        Pass(:test, local_test_expr, nothing, nothing, source)
                    else
                        Broken(:test, local_test_expr)
                    end
                catch
                    Broken(:test, local_test_expr)
                end
            elseif check_ad_backend_in(backend, broken_backends)
                try
                    ∂args = allow_unstable() do
                        return gradient(f, backend, args...)
                    end
                    matched = check_approx(∂args, ∂args_gt; kwargs...)
                    if matched
                        Error(:test_unbroken, local_test_expr, matched, nothing, source)
                    else
                        Broken(:test, local_test_expr)
                    end
                catch
                    Broken(:test, local_test_expr)
                end
            else
                try
                    ∂args = allow_unstable() do
                        return gradient(f, backend, args...)
                    end
                    matched = check_approx(∂args, ∂args_gt; kwargs...)
                    if matched
                        Pass(:test, local_test_expr, nothing, nothing, source)
                    else
                        context = "\n   ∂args: $(∂args)\n∂args_gt: $(∂args_gt)"
                        Fail(
                            :test,
                            local_test_expr,
                            matched,
                            nothing,
                            context,
                            source,
                            false,
                        )
                    end
                catch err
                    err isa InterruptException && rethrow()
                    Error(
                        :test_error,
                        local_test_expr,
                        err,
                        Base.current_exceptions(),
                        source,
                    )
                end
            end
            Test.record(get_testset(), result)
        end
    end
end

removed_backend(::AutoTracker) = true
removed_backend(::AutoReverseDiff) = true
removed_backend(x) = false

"""
    @test_gradients(f, args...; kwargs...)

See the documentation of [`test_gradients`](@ref) for more details. This macro provides
correct line information for the failing tests.
"""
macro test_gradients(exprs...)
    exs = reorder_macro_kw_params(exprs)
    kwarg_idx = findfirst(ex -> Meta.isexpr(ex, :kw), exs)
    if kwarg_idx === nothing
        args = [exs...]
        kwargs = []
    else
        args = [exs[1:(kwarg_idx - 1)]...]
        kwargs = [exs[kwarg_idx:end]...]
    end
    push!(kwargs, Expr(:kw, :source, QuoteNode(__source__)))
    push!(kwargs, Expr(:kw, :test_expr, QuoteNode(:(test_gradients($(exs...))))))
    return esc(:($(test_gradients)($(args...); $(kwargs...))))
end
