module LuxTestUtils

using ComponentArrays, Optimisers, Preferences, Test
using ForwardDiff, ReverseDiff, Tracker, Zygote, FiniteDifferences
# TODO: Yota, Enzyme

const JET_TARGET_MODULES = @load_preference("target_modules", nothing)

# JET Testing
try
    using JET
    global JET_TESTING_ENABLED = true
catch
    @warn "JET not not precompiling. All JET tests will be skipped!!" maxlog=1
    global JET_TESTING_ENABLED = false
end

"""
    @jet f(args...) call_broken=false opt_broken=false

Run JET tests on the function `f` with the arguments `args...`. If `JET` fails to compile
or julia version is < 1.7, then the macro will be a no-op.

## Keyword Arguments

  - `call_broken`: Marks the test_call as broken.
  - `opt_broken`: Marks the test_opt as broken.

All additional arguments will be forwarded to `@JET.test_call` and `@JET.test_opt`.

!!! note

    Instead of specifying `target_modules` with every call, you can set preferences for
    `target_modules` using `Preferences.jl`. For example, to set `target_modules` to
    `(Lux, LuxLib)` we can run:

    ```julia
    using Preferences

    set_preferences!(Base.UUID("ac9de150-d08f-4546-94fb-7472b5760531"),
                     "target_modules" => ["Lux", "LuxLib"])
    ```

## Example

```julia
@jet sum([1, 2, 3]) target_modules=(Base, Core)

@jet sum(1, 1) target_modules=(Base, Core) opt_broken=true
```
"""
macro jet(expr, args...)
    @static if VERSION >= v"1.7" && JET_TESTING_ENABLED
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
    X == GradientComputationSkipped || Y == GradientComputationSkipped && return :(true)
    hasmethod(isapprox, (X, Y)) && return :(isapprox(x, y; kwargs...))
    return quote
        @warn "No `isapprox` method found for types $(X) and $(Y). Using `==` instead."
        return x == y
    end
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
    return all(_checkapprox, zip(values(nt1), values(nt2)))
end

function check_approx(t1::NTuple{N, T}, t2::NTuple{N, T}; kwargs...) where {N, T}
    _check_approx(xy) = check_approx(xy[1], xy[2]; kwargs...)
    _check_approx(t::Tuple{Nothing, Nothing}) = true
    return all(_checkapprox, zip(t1, t2))
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

TODO: Write docs
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

function test_gradients_expr(__module__, __source__, f, args...; gpu_testing::Bool=false,
                             soft_fail::Bool=false,
                             # Skip Gradient Computation
                             skip_finite_differences::Bool=false,
                             skip_forward_diff::Bool=false, skip_zygote::Bool=false,
                             skip_tracker::Bool=false, skip_reverse_diff::Bool=false,
                             # Skip Large Arrays
                             large_arrays_skip_finite_differences::Bool=true,
                             large_arrays_skip_forward_diff::Bool=true,
                             large_array_length::Int=25, max_total_array_size::Int=100,
                             # Broken Tests
                             finite_differences_broken::Bool=false,
                             tracker_broken::Bool=false, reverse_diff_broken::Bool=false,
                             forward_diff_broken::Bool=false,
                             # Others passed to `check_approx`
                             kwargs...)
    orig_expr = QuoteNode(Expr(:macrocall,
                               GlobalRef(@__MODULE__, Symbol("@test_gradients")),
                               __source__, f, args...))
    len = length(args)
    __source__ = QuoteNode(__source__)
    return quote
        gs_zygote = __gradient(Zygote.gradient, $(esc(f)), $(esc.(args)...);
                               skip=$skip_zygote)

        any_non_array_input = any(!Base.Fix2(isa, AbstractArray), tuple($(esc.(args)...)))

        gs_tracker = __gradient(Base.Fix1(broadcast, Tracker.data) ∘ Tracker.gradient,
                                $(esc(f)), $(esc.(args)...);
                                skip=$skip_tracker || any_non_array_input)

        gs_rdiff = __gradient(_rdiff_gradient, $(esc(f)), $(esc.(args)...);
                              skip=$skip_reverse_diff ||
                                   any_non_array_input ||
                                   $gpu_testing)

        arr_len = length.(filter(Base.Fix2(isa, AbstractArray), tuple($(esc.(args)...))))
        large_arrays = any(x -> x >= $large_array_length, arr_len) ||
                       sum(arr_len) >= $max_total_array_size
        # if large_arrays
        #     @debug "Large arrays detected. Skipping some tests based on keyword arguments."
        # end

        gs_fdiff = __gradient(_fdiff_gradient, $(esc(f)), $(esc.(args)...);
                              skip=$skip_forward_diff ||
                                   (large_arrays && $large_arrays_skip_forward_diff) ||
                                   any_non_array_input ||
                                   $gpu_testing)

        gs_finite_diff = __gradient(_finitedifferences_gradient, $(esc(f)),
                                    $(esc.(args)...);
                                    skip=$skip_finite_differences ||
                                         (large_arrays &&
                                          $large_arrays_skip_finite_differences) ||
                                         any_non_array_input ||
                                         $gpu_testing)

        for idx in 1:($len)
            __test_gradient_pair_check($__source__, $orig_expr, gs_zygote[idx],
                                       gs_tracker[idx], "Zygote", "Tracker";
                                       broken=$tracker_broken, soft_fail=$soft_fail,
                                       $(kwargs...))
            __test_gradient_pair_check($__source__, $orig_expr, gs_zygote[idx],
                                       gs_rdiff[idx], "Zygote", "ReverseDiff";
                                       broken=$reverse_diff_broken, soft_fail=$soft_fail,
                                       $(kwargs...))
            __test_gradient_pair_check($__source__, $orig_expr, gs_zygote[idx],
                                       gs_fdiff[idx], "Zygote", "ForwardDiff";
                                       broken=$forward_diff_broken, soft_fail=$soft_fail,
                                       $(kwargs...))
            __test_gradient_pair_check($__source__, $orig_expr, gs_zygote[idx],
                                       gs_finite_diff[idx], "Zygote", "FiniteDifferences";
                                       broken=$finite_differences_broken,
                                       soft_fail=$soft_fail, $(kwargs...))
        end

        return nothing
    end
end

function __test_gradient_pair_check(__source__, orig_expr, v1, v2, name1, name2;
                                    broken::Bool=false, soft_fail::Bool=false, kwargs...)
    match = check_approx(v1, v2; kwargs...)
    test_type = Symbol("@test_gradients{$name1, $name2}")

    if !soft_fail
        if broken
            if !match
                test_res = Test.Broken(test_type, orig_expr)
            else
                test_res = Test.Error(test_type, orig_expr, nothing, nothing, __source__)
            end
        else
            if match
                test_res = Test.Pass(test_type, orig_expr, nothing, nothing, __source__)
            else
                test_res = Test.Fail(test_type, orig_expr, nothing, nothing, nothing,
                                     __source__)
            end
        end
    else
        if match
            test_res = Test.Pass(test_type, orig_expr, nothing, nothing, __source__)
        else
            test_res = Test.Broken(test_type, orig_expr)
        end
    end

    return Test.record(Test.get_testset(), test_res)
end

function __gradient(gradient_function, f, args...; skip::Bool)
    return skip ? ntuple(_ -> GradientComputationSkipped(), length(args)) :
           gradient_function(f, args...)
end

_rdiff_gradient(f, args...) = _named_tuple.(ReverseDiff.gradient(f, ComponentArray.(args)))

function _fdiff_gradient(f, args...)
    length(args) == 1 && return ForwardDiff.gradient(f, args[1])
    N = length(args)
    __f(x::ComponentArray) = f([getproperty(x, Symbol("input_$i")) for i in 1:N]...)
    ca = ComponentArray(NamedTuple{ntuple(i -> Symbol("input_$i"), N)}(args))
    return values(NamedTuple(ForwardDiff.gradient(__f, ca)))
end

function _finitedifferences_gradient(f, args...)
    return _named_tuple.(FiniteDifferences.grad(FiniteDifferences.central_fdm(8, 1), f,
                                                ComponentArray.(args)...))
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
