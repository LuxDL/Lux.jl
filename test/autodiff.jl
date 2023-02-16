using Lux, ComponentArrays, ReverseDiff, Random, Zygote, Test

include("test_utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "Gradient Correctness: Dense Chain" begin
    c = Chain(Dense(3, 4), Dense(4, 1))

    x = randn(rng, Float32, 3, 2)
    ps, st = Lux.setup(rng, c)

    ps = ps |> ComponentArray

    gs_r = ReverseDiff.gradient(ps -> sum(first(Lux.apply(c, x, ps, st))), ps)
    gs_z = Zygote.gradient(ps -> sum(first(Lux.apply(c, x, ps, st))), ps)[1]
    gs_fdm = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1),
                                    ps -> sum(first(Lux.apply(c, x, ps, st))), ps)[1]

    @test gs_r == gs_z
    @test gs_r ≈ gs_fdm
    @test gs_z ≈ gs_fdm
end

@testset "Broadcasting identity custom rrule" begin
    x = randn(rng, Float32, 3, 2)
    gs_x_1 = Zygote.gradient(x -> sum(identity.(x)), x)[1]
    gs_x_2 = Zygote.gradient(sum, x)[1]

    @test gs_x_1 == gs_x_2
end
