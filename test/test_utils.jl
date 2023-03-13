using ComponentArrays, FiniteDifferences, Functors, Lux, Optimisers, Random, Test
import ReverseDiff, Tracker, Zygote

try
    using JET
catch
    @warn "JET not not precompiling. All JET tests will be skipped." maxlog=1
    global test_call(args...; kwargs...) = nothing
    global test_opt(args...; kwargs...) = nothing
end

function Base.isapprox(x, y; kwargs...)
    @warn "`isapprox` is not defined for ($(typeof(x)), $(typeof(y))). Using `==` instead."
    return x == y
end

function Base.isapprox(x::Tuple, y::Tuple; kwargs...)
    return all(isapprox.(x, y; kwargs...))
end

function Base.isapprox(x::Optimisers.Leaf, y::Optimisers.Leaf; kwargs...)
    return isapprox(x.rule, y.rule; kwargs...) && isapprox(x.state, y.state; kwargs...)
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

_named_tuple(x::ComponentArray) = NamedTuple(x)
_named_tuple(x) = x

# Test the gradients generated using AD against the gradients generated using Finite Differences
function test_gradient_correctness_fdm(f::Function, args...; reversediff_broken=false,
                                       kwargs...)
    gs_ad_zygote = Zygote.gradient(f, args...)

    gs_ad_tracker = Tracker.gradient(f, args...)

    # ReverseDiff requires AbstractArray inputs
    if any(!Base.Fix2(isa, AbstractArray), args)
        rdiff_skipped = true
        gs_ad_rdiff = fmap(zero, args)
    else
        rdiff_skipped = false
        gs_ad_rdiff = _named_tuple.(ReverseDiff.gradient(f, ComponentArray.(args)))
    end

    gs_fdm = _named_tuple.(FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), f,
                                                  ComponentArray.(args)...))

    for (g_ad_zygote, g_ad_tracker, g_ad_rdiff, g_fdm) in zip(gs_ad_zygote, gs_ad_tracker,
                                                              gs_ad_rdiff, gs_fdm)
        @test isapprox(g_ad_zygote, g_fdm; kwargs...)
        @test isapprox(Tracker.data(g_ad_tracker), g_ad_zygote; kwargs...)
        if !rdiff_skipped
            if reversediff_broken
                @test_broken isapprox(g_ad_rdiff, g_ad_zygote; kwargs...)
            else
                @test isapprox(g_ad_rdiff, g_ad_zygote; kwargs...)
            end
        end
    end
end

# Some Helper Functions
function run_fwd_and_bwd(model, input, ps, st)
    y, pb = Zygote.pullback(p -> model(input, p, st)[1], ps)
    gs = pb(ones(eltype(y), size(y)))
    # if we make it to here with no error, success!
    return true
end

function run_model(m::Lux.AbstractExplicitLayer, x, mode=:test)
    ps, st = Lux.setup(Random.default_rng(), m)
    if mode == :test
        st = Lux.testmode(st)
    end
    return Lux.apply(m, x, ps, st)[1]
end

# JET Tests
function run_JET_tests(f, args...; call_broken=false, opt_broken=false, kwargs...)
    @static if VERSION >= v"1.7"
        test_call(f, typeof.(args); broken=call_broken, target_modules=(Lux,))
        test_opt(f, typeof.(args); broken=opt_broken, target_modules=(Lux,))
    end
end
