@testitem "@jet" begin
    LuxTestUtils.jet_target_modules!(["LuxTestUtils"])

    @jet sum([1, 2, 3]) target_modules=(Base, Core)
end

@testitem "test_gradients" begin
    using MetaTesting, ComponentArrays

    f(x, y, z) = x .+ sum(abs2, y.t) + sum(y.x.z)

    x = (; t=rand(10), x=(z=[2.0],))

    test_gradients(f, 1.0, x, nothing)

    test_gradients(f, 1.0, x, nothing; skip_backends=[AutoTracker()])
    @test_gradients(f, 1.0, x, nothing; skip_backends=[AutoTracker()])

    @test errors() do
        test_gradients(f, 1.0, x, nothing; broken_backends=[AutoTracker()])
    end

    @test errors() do
        @test_gradients(f, 1.0, x, nothing; broken_backends=[AutoTracker()])
    end

    @test_throws ArgumentError test_gradients(
        f, 1.0, x, nothing; broken_backends=[AutoTracker()],
        skip_backends=[AutoTracker(), AutoEnzyme()])
    @test_throws ArgumentError @test_gradients(
        f, 1.0, x, nothing; broken_backends=[AutoTracker()],
        skip_backends=[AutoTracker(), AutoEnzyme()])

    test_gradients(f, 1.0, x, nothing; soft_fail=[AutoTracker()])
    @test_gradients(f, 1.0, x, nothing; soft_fail=[AutoTracker()])

    test_gradients(f, 1.0, x, nothing; soft_fail=true)
    @test_gradients(f, 1.0, x, nothing; soft_fail=true)

    x_ca = ComponentArray(x)

    test_gradients(f, 1.0, x_ca, nothing)
    @test_gradients(f, 1.0, x_ca, nothing)

    x_2 = (; t=x.t', x=(z=x.x.z',))

    test_gradients(f, 1.0, x_2, nothing)
    @test_gradients(f, 1.0, x_2, nothing)
end

@testitem "test_gradients (CUDA.jl)" skip=:(using CUDA; !CUDA.functional()) begin
    using CUDA

    f(x, y, z) = x .+ sum(abs2, y.t) + sum(y.x.z)

    x = (; t=cu(rand(10)), x=(z=cu([2.0]),))

    test_gradients(f, 1.0, x, nothing)

    test_gradients(f, 1.0, x, nothing; skip_backends=[AutoTracker()])
end

@testitem "@test_softfail" begin
    using MetaTesting

    @test errors() do
        @test_softfail 1 + 1
    end
    @test_softfail 1 + 1 == 2
    @test_softfail 1 + 1 < 2
end
