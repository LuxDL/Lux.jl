using FiniteDifferences
using Lux
using Random
using Zygote

function Base.isapprox(nt1::NamedTuple{fields}, nt2::NamedTuple{fields}; kwargs...) where {fields}
    checkapprox(xy) = isapprox(xy[1], xy[2]; kwargs...)
    checkapprox(t::Tuple{Nothing,Nothing}) = true
    return all(checkapprox, zip(values(nt1), values(nt2)))
end

# Test the gradients generated using AD against the gradients generated using Finite Differences
function test_gradient_correctness_fdm(f::Function, args...; kwargs...)
    gs_ad = Zygote.gradient(f, args...)
    gs_fdm = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), f, args...)
    for (g_ad, g_fdm) in zip(gs_ad, gs_fdm)
        @test isapprox(g_ad, g_fdm; kwargs...)
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