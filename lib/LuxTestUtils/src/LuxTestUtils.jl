module LuxTestUtils

using ComponentArrays, Optimisers

# Type Inference: JET Tests
try
    using JET
catch
    @warn "JET not not precompiling. All JET tests will be skipped." maxlog=1
    global test_call(args...; kwargs...) = nothing
    global test_opt(args...; kwargs...) = nothing
end

function run_JET_tests(f, args...; call_broken=false, opt_broken=false, kwargs...)
    @static if VERSION >= v"1.7"
        test_call(f, typeof.(args); broken=call_broken, kwargs...)
        test_opt(f, typeof.(args); broken=opt_broken, kwargs...)
    end
end

# Approx Tests
function check_approx(x, y; kwargs...)
    hasmethod(isapprox, (typeof(x), typeof(y))) && return isapprox(x, y; kwargs...)
    @warn "`check_approx` is not defined for ($(typeof(x)), $(typeof(y))). Using `==` instead."
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

# Gradient Correctness Tests
_named_tuple(x::ComponentArray) = NamedTuple(x)
_named_tuple(x) = x

# Configuration

# Exports
export check_approx, run_JET_tests

end
