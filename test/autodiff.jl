using Lux, ComponentArrays, ReverseDiff, Random, Zygote, Test

include("utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

# Test Dense Layers
@testset "Gradient Correctness: Dense Chain" begin
    c = Chain(Dense(3, 4), Dense(4, 1))

    x = randn(Float32, 3, 2)
    ps, st = Lux.setup(rng, c)

    ps = ps |> Lux.ComponentArray

    gs_r = ReverseDiff.gradient(ps -> sum(first(Lux.apply(c, x, ps, st))), ps)
    gs_z = Zygote.gradient(ps -> sum(first(Lux.apply(c, x, ps, st))), ps)[1]
    gs_fdm = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1),
                                    ps -> sum(first(Lux.apply(c, x, ps, st))), ps)[1]

    @test gs_r == gs_z
    @test gs_r ≈ gs_fdm
    @test gs_z ≈ gs_fdm
end
