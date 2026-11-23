@testitem "@jet" begin
    LuxTestUtils.jet_target_modules!(["LuxTestUtils"])

    @jet sum([1, 2, 3]) target_modules = (Base, Core)
end

@testitem "test_gradients" begin
    using MetaTesting, ComponentArrays, Test

    f(x, y, z) = x .+ sum(abs2, y.t) + sum(y.x.z)

    x = (; t=rand(10), x=(z=[2.0],))

    test_gradients(f, 1.0, x, nothing)
    @test_gradients(f, 1.0, x, nothing)

    @test errors() do
        test_gradients(f, 1.0, x, nothing; broken_backends=[AutoForwardDiff()])
    end
    @test errors() do
        @test_gradients(f, 1.0, x, nothing; broken_backends=[AutoForwardDiff()])
    end

    @test_throws ArgumentError test_gradients(
        f,
        1.0,
        x,
        nothing;
        skip_backends=[AutoForwardDiff()],
        broken_backends=[AutoForwardDiff()],
    )
    @test_throws ArgumentError @test_gradients(
        f,
        1.0,
        x,
        nothing;
        skip_backends=[AutoForwardDiff()],
        broken_backends=[AutoForwardDiff()]
    )

    @test_throws ArgumentError test_gradients(
        f, 1.0, x, nothing; skip_backends=[AutoMooncake()]
    )

    test_gradients(f, 1.0, x, nothing; soft_fail=true)
    @test_gradients(f, 1.0, x, nothing; soft_fail=true)

    x_ca = ComponentArray(x)

    test_gradients(f, 1.0, x_ca, nothing)
    @test_gradients(f, 1.0, x_ca, nothing)

    x_2 = (; t=x.t', x=(z=x.x.z',))

    test_gradients(f, 1.0, x_2, nothing)
    @test_gradients(f, 1.0, x_2, nothing)

    # Test Enzyme skipped when ENZYME_TESTING_ENABLED[] == false
    old_enzyme_status = LuxTestUtils.ENZYME_TESTING_ENABLED[]
    try
        LuxTestUtils.ENZYME_TESTING_ENABLED[] = false
        f_simple(x) = sum(abs2, x)
        ts = test_gradients(f_simple, rand(Float32, 4); enable_enzyme_reverse_mode=true)
        enzyme_ts = only(
            filter(
                r -> r isa Test.DefaultTestSet && occursin("AutoEnzyme", r.description),
                ts.results,
            ),
        )
        @test any(r -> r isa Test.Broken && r.test_type === :skipped, enzyme_ts.results)
        @test !any(r -> r isa Test.Pass, enzyme_ts.results)
    finally
        LuxTestUtils.ENZYME_TESTING_ENABLED[] = old_enzyme_status
    end

    # Test needs_gradient and test_gradients with nothing alongside float arrays
    @test !LuxTestUtils.needs_gradient((;
        running_mean=ones(Float32, 3), training=Val(true)
    ))
    @test !LuxTestUtils.needs_gradient((; a=nothing))

    nt = (; weight=randn(Float32, 3, 2), bias=nothing)
    @test LuxTestUtils.needs_gradient(nt)

    f_nt(x) = sum(abs2, x.weight)
    if LuxTestUtils.ZYGOTE_TESTING_ENABLED[]
        @test LuxTestUtils.gradient(f_nt, AutoZygote(), nt)[1].weight ≈ 2 .* nt.weight
    end
    test_gradients(f_nt, nt)
    @test_gradients(f_nt, nt)

    tup = (randn(Float32, 3, 2), nothing)
    @test LuxTestUtils.needs_gradient(tup)
    f_tup(x) = sum(abs2, x[1])
    test_gradients(f_tup, tup)
    @test_gradients(f_tup, tup)
end

@testitem "test_gradients (CUDA.jl)" skip = :(using CUDA; !CUDA.functional()) begin
    using CUDA

    f(x, y, z) = x .+ sum(abs2, y.t) + sum(y.x.z)

    x = (; t=cu(rand(10)), x=(z=cu([2.0]),))

    test_gradients(f, 1.0, x, nothing)

    test_gradients(f, 1.0, x, nothing)
end

@testitem "@test_softfail" begin
    using MetaTesting

    @test errors() do
        @test_softfail 1 + 1
    end
    @test_softfail 1 + 1 == 2
    @test_softfail 1 + 1 < 2
end
