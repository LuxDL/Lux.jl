module LuxTestUtils

using ComponentArrays, Optimisers, Preferences, LuxCore, LuxDeviceUtils, Test
using ForwardDiff, ReverseDiff, Tracker, Zygote, FiniteDifferences
# TODO: Yota, Enzyme

const JET_TARGET_MODULES = @load_preference("target_modules", nothing)

# JET Testing
try
    using JET
    global JET_TESTING_ENABLED = true

    import JET: JETTestFailure, get_reports
catch
    @warn "JET not not precompiling. All JET tests will be skipped!!" maxlog=1
    global JET_TESTING_ENABLED = false
end

import Test: Error, Broken, Pass, Fail, get_testset

"""
    @jet f(args...) call_broken=false opt_broken=false

Run JET tests on the function `f` with the arguments `args...`. If `JET` fails to compile
or julia version is < 1.7, then the macro will be a no-op.

## Keyword Arguments

  - `call_broken`: Marks the test_call as broken.
  - `opt_broken`: Marks the test_opt as broken.

All additional arguments will be forwarded to `@JET.test_call` and `@JET.test_opt`.

:::tip

Instead of specifying `target_modules` with every call, you can set preferences for
`target_modules` using `Preferences.jl`. For example, to set `target_modules` to
`(Lux, LuxLib)` we can run:

```julia
using Preferences

set_preferences!(Base.UUID("ac9de150-d08f-4546-94fb-7472b5760531"),
    "target_modules" => ["Lux", "LuxLib"])
```

:::

## Example

```julia
using LuxTestUtils

@testset "Showcase JET Testing" begin
    @jet sum([1, 2, 3]) target_modules=(Base, Core)

    @jet sum(1, 1) target_modules=(Base, Core) opt_broken=true
end
```
"""
macro jet(expr, args...)
    if JET_TESTING_ENABLED
        all_args, call_extras, opt_extras = [], [], []
        target_modules_set = false
        for kwexpr in args
            if Meta.isexpr(kwexpr, :(=))
                if kwexpr.args[1] == :call_broken
                    push!(call_extras, :(broken = $(kwexpr.args[2])))
                elseif kwexpr.args[1] == :opt_broken
                    push!(opt_extras, :(broken = $(kwexpr.args[2])))
                elseif kwexpr.args[1] == :broken
                    throw(ArgumentError("`broken` keyword argument is ambiguous. Use `call_broken` or `opt_broken` instead."))
                else
                    kwexpr.args[1] == :target_modules && (target_modules_set = true)
                    push!(all_args, kwexpr)
                end
            else
                push!(all_args, kwexpr)
            end
        end

        if !target_modules_set && JET_TARGET_MODULES !== nothing
            target_modules = getproperty.((__module__,), Tuple(Symbol.(JET_TARGET_MODULES)))
            push!(all_args, :(target_modules = $target_modules))
        end

        push!(all_args, expr)

        ex_call = JET.call_test_ex(:report_call, Symbol("@test_call"),
            vcat(call_extras, all_args), __module__, __source__)
        ex_opt = JET.call_test_ex(:report_opt, Symbol("@test_opt"),
            vcat(opt_extras, all_args), __module__, __source__)

        return Expr(:block, ex_call, ex_opt)
    end
    return :()
end

# Approximate Equality
struct GradientComputationSkipped end

@generated function check_approx(x::X, y::Y; kwargs...) where {X, Y}
    device = cpu_device()
    (X == GradientComputationSkipped || Y == GradientComputationSkipped) && return :(true)
    hasmethod(isapprox, (X, Y)) && return :(isapprox($(device)(x), $(device)(y); kwargs...))
    return quote
        @warn "No `isapprox` method found for types $(X) and $(Y). Using `==` instead."
        return $(device)(x) == $(device)(y)
    end
end

function check_approx(x::LuxCore.AbstractExplicitLayer, y::LuxCore.AbstractExplicitLayer;
        kwargs...)
    return x == y
end
check_approx(x::Tuple, y::Tuple; kwargs...) = all(check_approx.(x, y; kwargs...))

function check_approx(x::Optimisers.Leaf, y::Optimisers.Leaf; kwargs...)
    return check_approx(x.rule, y.rule; kwargs...) &&
           check_approx(x.state, y.state; kwargs...)
end

function check_approx(nt1::NamedTuple{fields}, nt2::NamedTuple{fields};
        kwargs...) where {fields}
    _check_approx(xy) = check_approx(xy[1], xy[2]; kwargs...)
    _check_approx(t::Tuple{Nothing, Nothing}) = true
    return all(_check_approx, zip(values(nt1), values(nt2)))
end

function check_approx(t1::NTuple{N, T}, t2::NTuple{N, T}; kwargs...) where {N, T}
    _check_approx(xy) = check_approx(xy[1], xy[2]; kwargs...)
    _check_approx(t::Tuple{Nothing, Nothing}) = true
    return all(_check_approx, zip(t1, t2))
end

function check_approx(ca::ComponentArray, nt::NamedTuple; kwargs...)
    return check_approx(NamedTuple(ca), nt; kwargs...)
end
function check_approx(nt::NamedTuple, ca::ComponentArray; kwargs...)
    return check_approx(nt, NamedTuple(ca); kwargs...)
end

check_approx(::Nothing, v::AbstractArray; kwargs...) = length(v) == 0
check_approx(v::AbstractArray, ::Nothing; kwargs...) = length(v) == 0
check_approx(v::NamedTuple, ::Nothing; kwargs...) = length(v) == 0
check_approx(::Nothing, v::NamedTuple; kwargs...) = length(v) == 0
check_approx(v::Tuple, ::Nothing; kwargs...) = length(v) == 0
check_approx(::Nothing, v::Tuple; kwargs...) = length(v) == 0
check_approx(x::AbstractArray, y::NamedTuple; kwargs...) = length(x) == 0 && length(y) == 0
check_approx(x::NamedTuple, y::AbstractArray; kwargs...) = length(x) == 0 && length(y) == 0
check_approx(x::AbstractArray, y::Tuple; kwargs...) = length(x) == 0 && length(y) == 0
check_approx(x::Tuple, y::AbstractArray; kwargs...) = length(x) == 0 && length(y) == 0

# Test Gradients across ADs and FiniteDifferences
"""
    @test_gradients f args... [kwargs...]

Compare the gradients computed by Zygote.jl (Reverse Mode AD) against:

  - Tracker.jl (Reverse Mode AD)
  - ReverseDiff.jl (Reverse Mode AD)
  - ForwardDiff.jl (Forward Mode AD)
  - FiniteDifferences.jl (Finite Differences)

:::tip

This function is completely compatible with Test.jl

:::

## Arguments

  - `f`: The function to test.
  - `args...`: Inputs to `f` wrt which the gradients are computed.

## Keyword Arguments

  - `gpu_testing`: Disables ForwardDiff, ReverseDiff and FiniteDifferences tests. (Default:
    `false`)
  - `soft_fail`: If `true`, the test will not fail if any of the gradients are incorrect,
    instead it will show up as broken. (Default: `false`)
  - `skip_(tracker|reverse_diff|forward_diff|finite_differences)`: Skip the
    corresponding gradient computation and check. (Default: `false`)
  - `large_arrays_skip_(forward_diff|finite_differences)`: Skip the corresponding gradient
    computation and check for large arrays. (Forward Mode and Finite Differences are not
    efficient for large arrays.) (Default: `true`)
  - `large_array_length`: The length of the array above which the gradient computation is
    considered large. (Default: 25)
  - `max_total_array_size`: Treat as large array if the total size of all arrays is greater
    than this value. (Default: 100)
  - `(tracker|reverse_diff|forward_diff|finite_differences)_broken`: Mark the corresponding
    gradient test as broken. (Default: `false`)

## Keyword Arguments for `check_approx`

  - `atol`: Absolute tolerance for gradient comparisons. (Default: `0.0`)
  - `rtol`: Relative tolerance for gradient comparisons.
    (Default: `atol > 0 ? 0.0 : √eps(typeof(atol))`)
  - `nans`: Whether or not NaNs are considered equal. (Default: `false`)

## Example

```julia
using LuxTestUtils

x = randn(10)

@testset "Showcase Gradient Testing" begin
    @test_gradients sum abs2 x

    @test_gradients prod x
end
```
"""
macro test_gradients(all_args...)
    args, kwargs = [], Pair{Symbol, Any}[]

    for kwexpr in all_args
        if Meta.isexpr(kwexpr, :(=))
            push!(kwargs, kwexpr.args[1] => kwexpr.args[2])
        else
            push!(args, kwexpr)
        end
    end

    return test_gradients_expr(__module__, __source__, args...; kwargs...)
end

function test_gradients_expr(__module__, __source__, f, args...;
        gpu_testing::Bool=false,
        soft_fail::Bool=false,
        # Skip Gradient Computation
        skip_finite_differences::Bool=false,
        skip_forward_diff::Bool=false,
        skip_zygote::Bool=false,
        skip_tracker::Bool=false,
        skip_reverse_diff::Bool=false,
        # Skip Large Arrays
        large_arrays_skip_finite_differences::Bool=true,
        large_arrays_skip_forward_diff::Bool=true,
        large_array_length::Int=25,
        max_total_array_size::Int=100,
        # Broken Tests
        finite_differences_broken::Bool=false,
        tracker_broken::Bool=false,
        reverse_diff_broken::Bool=false,
        forward_diff_broken::Bool=false,
        # Others passed to `check_approx`
        atol::Real=0.0,
        rtol::Real=atol > 0 ? 0.0 : √eps(typeof(atol)),
        nans::Bool=false,
        kwargs...)
    orig_exprs = map(x -> QuoteNode(Expr(:macrocall,
            GlobalRef(@__MODULE__, Symbol("@test_gradients{$x}")), __source__, f, args...)),
        ("Tracker", "ReverseDiff", "ForwardDiff", "FiniteDifferences"))
    len = length(args)
    __source__ = QuoteNode(__source__)
    return quote
        gs_zygote = __gradient(Zygote.gradient, $(esc(f)), $(esc.(args)...);
            skip=$skip_zygote)

        gs_tracker = __gradient(Base.Fix1(broadcast, Tracker.data) ∘ Tracker.gradient,
            $(esc(f)), $(esc.(args)...); skip=$skip_tracker)
        tracker_broken = $(tracker_broken && !skip_tracker)

        skip_reverse_diff = $(skip_reverse_diff || gpu_testing)
        gs_rdiff = __gradient(_rdiff_gradient, $(esc(f)), $(esc.(args)...);
            skip=skip_reverse_diff)
        reverse_diff_broken = $reverse_diff_broken && !skip_reverse_diff

        arr_len = length.(filter(Base.Fix2(isa, AbstractArray) ∘
                                 Base.Fix1(__correct_arguments, identity),
            tuple($(esc.(args)...))))
        large_arrays = any(x -> x ≥ $large_array_length, arr_len) ||
                       sum(arr_len) ≥ $max_total_array_size
        if large_arrays
            @debug "Large arrays detected. Skipping some tests based on keyword arguments."
        end

        skip_forward_diff = $skip_forward_diff || $gpu_testing ||
                            (large_arrays && $large_arrays_skip_forward_diff)
        gs_fdiff = __gradient(_fdiff_gradient, $(esc(f)), $(esc.(args)...);
            skip=skip_forward_diff)
        forward_diff_broken = $forward_diff_broken && !skip_forward_diff

        skip_finite_differences = $skip_finite_differences || $gpu_testing ||
                                  (large_arrays && $large_arrays_skip_finite_differences)
        gs_finite_diff = __gradient(_finitedifferences_gradient, $(esc(f)),
            $(esc.(args)...); skip=skip_finite_differences)
        finite_differences_broken = $finite_differences_broken && !skip_finite_differences

        for idx in 1:($len)
            __test_gradient_pair_check($__source__, $(orig_exprs[1]), gs_zygote[idx],
                gs_tracker[idx], "Zygote", "Tracker"; broken=tracker_broken,
                soft_fail=$soft_fail, atol=$atol, rtol=$rtol, nans=$nans)
            __test_gradient_pair_check($__source__, $(orig_exprs[2]), gs_zygote[idx],
                gs_rdiff[idx], "Zygote", "ReverseDiff"; broken=reverse_diff_broken,
                soft_fail=$soft_fail, atol=$atol, rtol=$rtol, nans=$nans)
            __test_gradient_pair_check($__source__, $(orig_exprs[3]), gs_zygote[idx],
                gs_fdiff[idx], "Zygote", "ForwardDiff"; broken=forward_diff_broken,
                soft_fail=$soft_fail, atol=$atol, rtol=$rtol, nans=$nans)
            __test_gradient_pair_check($__source__, $(orig_exprs[4]), gs_zygote[idx],
                gs_finite_diff[idx], "Zygote", "FiniteDifferences";
                broken=finite_differences_broken, soft_fail=$soft_fail, atol=$atol,
                rtol=$rtol, nans=$nans)
        end
    end
end

function __test_gradient_pair_check(__source__, orig_expr, v1, v2, name1, name2;
        broken::Bool=false, soft_fail::Bool=false, kwargs...)
    match = check_approx(v1, v2; kwargs...)
    test_type = Symbol("@test_gradients{$name1, $name2}")

    test_func = soft_fail ? (match ? __test_pass : __test_broken) :
                (broken ? (match ? __test_error : __test_broken) :
                 (match ? __test_pass : __test_fail))

    return Test.record(Test.get_testset(), test_func(test_type, orig_expr, __source__))
end

function __test_pass(test_type, orig_expr, source)
    return Test.Pass(test_type, orig_expr, nothing, nothing, source)
end

function __test_fail(test_type, orig_expr, source)
    return Test.Fail(test_type, orig_expr, nothing, nothing, nothing, source, false)
end

function __test_error(test_type, orig_expr, source)
    return Test.Error(test_type, orig_expr, nothing, nothing, source)
end

__test_broken(test_type, orig_expr, source) = Test.Broken(test_type, orig_expr)

__correct_arguments(f::F, x::AbstractArray) where {F} = x
function __correct_arguments(f::F, x::NamedTuple) where {F}
    cpu_dev = cpu_device()
    gpu_dev = gpu_device()
    xc = cpu_dev(x)
    ca = ComponentArray(xc)
    # Hacky check to see if there are any non-CPU arrays in the NamedTuple
    typeof(xc) == typeof(x) && return ca
    return gpu_dev(ca)
end
__correct_arguments(f::F, x) where {F} = x

__uncorrect_arguments(x::ComponentArray, ::NamedTuple, z::ComponentArray) = NamedTuple(x)
function __uncorrect_arguments(x::AbstractArray, nt::NamedTuple, z::ComponentArray)
    return __uncorrect_arguments(ComponentArray(vec(x), getaxes(z)), nt, z)
end
__uncorrect_arguments(x, y, z) = x

function __gradient(gradient_function::F, f, args...; skip::Bool) where {F}
    if skip
        return ntuple(_ -> GradientComputationSkipped(), length(args))
    else
        corrected_args = map(Base.Fix1(__correct_arguments, gradient_function), args)
        aa_inputs = [map(Base.Fix2(isa, AbstractArray), corrected_args)...]
        __aa_input_idx = cumsum(aa_inputs)
        if sum(aa_inputs) == length(args)
            gs = gradient_function(f, corrected_args...)
            return ntuple(i -> __uncorrect_arguments(gs[i], args[i], corrected_args[i]),
                length(args))
        end
        function __f(inputs...)
            updated_inputs = ntuple(i -> aa_inputs[i] ? inputs[__aa_input_idx[i]] : args[i],
                length(args))
            return f(updated_inputs...)
        end
        gs = gradient_function(__f, [corrected_args...][aa_inputs]...)
        return ntuple(i -> aa_inputs[i] ?
                           __uncorrect_arguments(gs[__aa_input_idx[i]],
                args[__aa_input_idx[i]],
                corrected_args[__aa_input_idx[i]]) : GradientComputationSkipped(),
            length(args))
    end
end

_rdiff_gradient(f, args...) = _named_tuple.(ReverseDiff.gradient(f, args))

function _fdiff_gradient(f, args...)
    length(args) == 1 && return (ForwardDiff.gradient(f, args[1]),)
    N = length(args)
    __f(x::ComponentArray) = f([getproperty(x, Symbol("input_$i")) for i in 1:N]...)
    ca = ComponentArray(NamedTuple{ntuple(i -> Symbol("input_$i"), N)}(args))
    return values(NamedTuple(ForwardDiff.gradient(__f, ca)))
end

function _finitedifferences_gradient(f, args...)
    return _named_tuple.(FiniteDifferences.grad(FiniteDifferences.central_fdm(3, 1), f,
        args...))
end

function __correct_arguments(::typeof(_finitedifferences_gradient), x::NamedTuple)
    cpu_dev = cpu_device()
    gpu_dev = gpu_device()
    xc = cpu_dev(x)
    ca = ComponentArray(xc)
    # Hacky check to see if there are any non-CPU arrays in the NamedTuple
    typeof(xc) == typeof(x) && return x
    return gpu_dev(x)
end

function __fdiff_compatible_function(f, ::Val{N}) where {N}
    N == 1 && return f
    inputs = ntuple(i -> Symbol("x.input_$i"), N)
    function __fdiff_compatible_function_closure(x::ComponentArray)
        return f([getproperty(x, Symbol("input_$i")) for i in 1:N]...)
    end
end

_named_tuple(x::ComponentArray) = NamedTuple(x)
_named_tuple(x) = x

# Exports
export @jet, @test_gradients

end
