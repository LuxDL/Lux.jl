using Lux, ComponentArrays, ReverseDiff, Random, Zygote, Test

include("test_utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "Gradient Correctness: Dense Chain" begin
    c = Chain(Dense(3, 4), Dense(4, 1))

    x = randn(rng, Float32, 3, 2)
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

@testset "Broadcasting identity custom rrule" begin
    x = randn(rng, Float32, 3, 2)
    gs_x_1 = Zygote.gradient(x -> sum(identity.(x)), x)[1]
    gs_x_2 = Zygote.gradient(sum, x)[1]

    @test gs_x_1 == gs_x_2
end

@testset "_reshape_into_proper_shape" begin
    x = randn(rng, Float32, 3, 2)
    y = randn(rng, Float32, 2, 2, 6, 2)

    @test size(Lux._reshape_into_proper_shape(x, y)) == (1, 1, 6, 1)
    @inferred Lux._reshape_into_proper_shape(x, y)

    gs_1 = Zygote.gradient(x -> sum(Lux._reshape_into_proper_shape(x, y)), x)[1]
    gs_2 = Zygote.gradient(x -> sum(reshape(x, (1, 1, 6, 1))), x)[1]

    @test gs_1 == gs_2
end
