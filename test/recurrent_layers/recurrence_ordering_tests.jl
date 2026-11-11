using LuxTestUtils: check_approx

include("../shared_testsetup.jl")

@testset "Recurrence Ordering Check #302" begin
    rng = StableRNG(12345)
    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        encoder = Recurrence(
            RNNCell(
                1 => 1,
                identity;
                init_weight=(rng, args...; kwargs...) -> ones(args...; kwargs...),
                init_state=(rng, args...; kwargs...) -> zeros(args...; kwargs...),
                init_bias=(rng, args...; kwargs...) -> zeros(args...; kwargs...),
            );
            return_sequence=true,
        )
        ps, st = dev(Lux.setup(rng, encoder))
        m2 = aType(reshape([0.5, 0.0, 0.7, 0.8], 1, :, 1))
        res, _ = encoder(m2, ps, st)

        @test Array(vec(reduce(vcat, res))) ≈ [0.5, 0.5, 1.2, 2.0]
    end

    @testset "Recurrence Sequence Accumulation & Edge Cases" begin
        r = Recurrence(RNNCell(3 => 4); return_sequence=true)
        ps, st = Lux.setup(rng, r)

        empty_seq = Matrix{Float32}[]
        y_empty, _ = r(empty_seq, ps, st)
        @test isempty(y_empty)

        x_het = AbstractMatrix[ones(Float32, 3, 2), ones(Float64, 3, 2)]
        y_het, _ = r(x_het, ps, st)
        @test length(y_het) == 2
        @test eltype(y_het) == Matrix{Float64}

        x50 = [randn(rng, Float32, 3, 2) for _ in 1:50]
        x250 = [randn(rng, Float32, 3, 2) for _ in 1:250]
        r(x50, ps, st)
        r(x250, ps, st)
        alloc50 = @allocated r(x50, ps, st)
        alloc250 = @allocated r(x250, ps, st)
        @test alloc250 < 8 * alloc50

        g_first = Zygote.gradient(x -> sum(first(first(r(x, ps, st)))), x50[1:4])[1]
        @test length(g_first) == 4
        @test g_first[1] !== nothing
    end
end
