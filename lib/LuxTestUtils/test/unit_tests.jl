@testitem "@jet" begin
    LuxTestUtils.jet_target_modules!(["LuxTestUtils"])

    @jet sum([1, 2, 3]) target_modules=(Base, Core)
end

@testitem "test_gradients" begin
    f(x, y, z) = x .+ sum(abs2, y.t) + sum(y.x.z)

    x = (; t=rand(10), x=(z=[2.0],))

    test_gradients(f, 1.0, x, nothing)
end
