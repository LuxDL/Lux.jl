using ComponentArrays, Lux, Random, Test, Zygote

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "LuxComponentArraysExt" begin
    # Ref: https://github.com/avik-pal/Lux.jl/issues/243
    nn = Chain(Dense(4, 3), Dense(3, 2))
    ps, st = Lux.setup(rng, nn)

    l2reg(p) = sum(abs2, ComponentArray(p))
    @test_nowarn gradient(l2reg, ps)
end
