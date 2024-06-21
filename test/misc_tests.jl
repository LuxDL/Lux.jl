# Previously we had a few Zygote.gradient over Zygote.gradient tests but those are now 
# removed in favor of BatchedRoutines.jl

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
