using Boltz, BruLux, Functors, Lux, Random
using Test

@testset "BruLux.jl" begin
    @testset "Dense" begin
        rng = Random.default_rng()
        Random.seed!(rng, 0)

        # Dense without bias
        d = Dense(2 => 4; bias=false)
        ps, st = Lux.setup(rng, d)
        ps_b, st_b = fmap(BruLuxArray, ps), fmap(BruLuxArray, st)
        x = randn(Float32, 2, 3)
        x_b = BruLuxArray(x)

        @test isapprox(d(x, ps, st)[1], d(x_b, ps_b, st_b)[1])

        # Dense with bias
        d = Dense(2 => 4)
        ps, st = Lux.setup(rng, d)
        ps_b, st_b = fmap(BruLuxArray, ps), fmap(BruLuxArray, st)
        x = randn(Float32, 2, 3)
        x_b = BruLuxArray(x)

        @test isapprox(d(x, ps, st)[1], d(x_b, ps_b, st_b)[1])
    end

    @testset "Conv" begin
        rng = Random.default_rng()
        Random.seed!(rng, 0)

        c = Conv((3, 3), 3 => 16)
        ps, st = Lux.setup(rng, c)
        ps_b, st_b = fmap(BruLuxArray, ps), fmap(BruLuxArray, st)
        x = randn(Float32, 3, 3, 3, 1)
        x_b = BruLuxArray(x)

        @test isapprox(c(x, ps, st)[1], c(x_b, ps_b, st_b)[1])
    end

    @testset "Vision Transformer" begin
        rng = Random.default_rng()
        Random.seed!(rng, 0)

        make_brulux_array(x::AbstractArray) = BruLuxArray(x)
        make_brulux_array(x) = x

        vit, ps, st = vision_transformer(:tiny)
        ps_b, st_b = fmap(make_brulux_array, ps), fmap(make_brulux_array, st)
        x = randn(Float32, 256, 256, 3, 2)
        x_b = BruLuxArray(x)

        r = vit(x, ps, st)[1]
        r_b = vit(x_b, ps_b, st_b)[1]

        @test isapprox(r, r_b)
    end
end
