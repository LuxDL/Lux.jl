using CUDA, FiniteDifferences, LuxLib, Test
using ReverseDiff, Tracker, Zygote  # AD Packages

const LUXLIB_TESTING_MODE = get(ENV, "LUXLIB_TESTING_MODE", :all)

try
    using JET
catch
    @warn "JET not not precompiling. All JET tests will be skipped." maxlog=1
    global test_call(args...; kwargs...) = nothing
    global test_opt(args...; kwargs...) = nothing
end

function cpu_testing()
    return LUXLIB_TESTING_MODE == :all || LUXLIB_TESTING_MODE == :cpu
end

function gpu_testing()
    return (LUXLIB_TESTING_MODE == :all || LUXLIB_TESTING_MODE == :gpu) && has_cuda()
end

function Base.isapprox(x, y; kwargs...)
    @warn "`isapprox` is not defined for ($(typeof(x)), $(typeof(y))). Using `==` instead."
    return x == y
end

function Base.isapprox(x::Tuple, y::Tuple; kwargs...)
    return all(isapprox.(x, y; kwargs...))
end

function Base.isapprox(nt1::NamedTuple{fields}, nt2::NamedTuple{fields};
                       kwargs...) where {fields}
    checkapprox(xy) = isapprox(xy[1], xy[2]; kwargs...)
    checkapprox(t::Tuple{Nothing, Nothing}) = true
    return all(checkapprox, zip(values(nt1), values(nt2)))
end

function Base.isapprox(t1::NTuple{N, T}, t2::NTuple{N, T}; kwargs...) where {N, T}
    checkapprox(xy) = isapprox(xy[1], xy[2]; kwargs...)
    checkapprox(t::Tuple{Nothing, Nothing}) = true
    return all(checkapprox, zip(t1, t2))
end

Base.isapprox(::Nothing, v::AbstractArray; kwargs...) = length(v) == 0
Base.isapprox(v::AbstractArray, ::Nothing; kwargs...) = length(v) == 0
Base.isapprox(v::NamedTuple, ::Nothing; kwargs...) = length(v) == 0
Base.isapprox(::Nothing, v::NamedTuple; kwargs...) = length(v) == 0
Base.isapprox(v::Tuple, ::Nothing; kwargs...) = length(v) == 0
Base.isapprox(::Nothing, v::Tuple; kwargs...) = length(v) == 0
Base.isapprox(x::AbstractArray, y::NamedTuple; kwargs...) = length(x) == 0 && length(y) == 0
Base.isapprox(x::NamedTuple, y::AbstractArray; kwargs...) = length(x) == 0 && length(y) == 0
Base.isapprox(x::AbstractArray, y::Tuple; kwargs...) = length(x) == 0 && length(y) == 0
Base.isapprox(x::Tuple, y::AbstractArray; kwargs...) = length(x) == 0 && length(y) == 0

# JET Tests
function run_JET_tests(f, args...; call_broken=false, opt_broken=false, kwargs...)
    @static if VERSION >= v"1.7"
        test_call(f, typeof.(args); broken=call_broken, target_modules=(LuxLib,))
        test_opt(f, typeof.(args); broken=opt_broken, target_modules=(LuxLib,))
    end
end

# Test the gradients generated using AD against the gradients generated using Finite
# Differences
# Currently this is called exclusively on CPU. So we can simply use ReverseDiff.
# However this function has evolved to be more general and can be used to test GPU autodiff.
function test_gradient_correctness_fdm(f::Function, args...; kwargs...)
    gs_ad_zygote = Zygote.gradient(f, args...)
    gs_ad_tracker = Tracker.gradient(f, args...)
    gs_ad_reversediff = ReverseDiff.gradient(f, args)
    gs_fdm = FiniteDifferences.grad(FiniteDifferences.central_fdm(8, 1), f, args...)
    for (g_ad_zygote, g_ad_tracker, g_ad_reverse_diff, g_fdm) in zip(gs_ad_zygote,
                                                                     gs_ad_tracker,
                                                                     gs_ad_reversediff,
                                                                     gs_fdm)
        @test isapprox(g_ad_zygote, g_fdm; kwargs...)
        @test isapprox(Tracker.data(g_ad_tracker), g_ad_zygote; kwargs...)
        @test isapprox(ReverseDiff.value(g_ad_reverse_diff), g_ad_zygote; kwargs...)
    end
end
