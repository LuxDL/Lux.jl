@testitem "Tracing AD: AoS to SoA" setup=[SharedTestSetup] tags=[:autodiff] begin
    using ReverseDiff, Tracker

    rng = StableRNG(1234)

    x = rand(rng, Float32, 1, 128)
    nn = Dense(1 => 1)
    ps, st = Lux.setup(rng, nn)

    x_t = Tracker.TrackedReal.(x)
    y_t = LuxCore.stateless_apply(nn, x_t, ps)
    @test y_t isa Tracker.TrackedArray

    y_t = first(nn(x_t, ps, st))
    @test y_t isa AbstractArray{<:Tracker.TrackedReal}

    x_t = ReverseDiff.TrackedReal.(x, zero(x))
    y_t = LuxCore.stateless_apply(nn, x_t, ps)
    @test y_t isa ReverseDiff.TrackedArray

    y_t = first(nn(x_t, ps, st))
    @test y_t isa AbstractArray{<:ReverseDiff.TrackedReal}
end

@testitem "Tracker.jl patches" setup=[SharedTestSetup] tags=[:autodiff] begin
    using Tracker

    nested_st = (; m=Dense(2 => 3), v=rand(2), d=(; x=(rand(2), 1)))
    tnested_st = Tracker.param(nested_st)

    @test tnested_st.m === nested_st.m
    @test tnested_st.v isa TrackedArray
    @test tnested_st.d.x[1] isa TrackedArray
    @test tnested_st.d.x[2] isa Tracker.TrackedReal

    @test_nowarn Tracker.zero_grad!(nested_st)
    @test_nowarn Tracker.zero_grad!(nested_st.m)

    @test_nowarn Tracker.extract_grad!(tnested_st)
    @test_nowarn Tracker.data(tnested_st)

    x = ones(10) |> Tracker.param
    @test Lux._gate(x, 1, 1) isa TrackedVector
end
