using Lux, Random, Test

include("../test_utils.jl")

rng = Random.default_rng()
Random.seed!(rng, 0)

@testset "All Parameters Freezing" begin
    d = Dense(5 => 5)
    psd, std = Lux.setup(rng, d)

    fd, ps, st = Lux.freeze(d, psd, std, nothing)
    @test length(keys(ps)) == 0
    @test length(keys(st)) == 2
    @test sort([keys(st)...]) == [:frozen_params, :states]
    @test sort([keys(st.frozen_params)...]) == [:bias, :weight]

    x = randn(rng, Float32, 5, 1)

    @test d(x, psd, std)[1] == fd(x, ps, st)[1]

    run_JET_tests(fd, x, ps, st)
    test_gradient_correctness_fdm(x -> sum(fd(x, ps, st)[1]), x; atol=1.0f-3, rtol=1.0f-3)
end

@testset "Partial Freezing" begin
    d = Dense(5 => 5)
    psd, std = Lux.setup(rng, d)

    fd, ps, st = Lux.freeze(d, psd, std, (:weight,))
    @test length(keys(ps)) == 1
    @test length(keys(st)) == 2
    @test sort([keys(st)...]) == [:frozen_params, :states]
    @test sort([keys(st.frozen_params)...]) == [:weight]
    @test sort([keys(ps)...]) == [:bias]

    x = randn(rng, Float32, 5, 1)

    @test d(x, psd, std)[1] == fd(x, ps, st)[1]

    run_JET_tests(fd, x, ps, st)
    test_gradient_correctness_fdm((x, ps) -> sum(fd(x, ps, st)[1]), x, ps; atol=1.0f-3,
                                  rtol=1.0f-3)
end
