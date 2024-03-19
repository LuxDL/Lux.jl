@testitem "Simple Zygote Second Order Derivative" setup=[SharedTestSetup] begin
    # Add tests for BatchedRoutines here
    rng = get_stable_rng()

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        x = rand(rng, Float32, 1, 128) |> aType
        nn = Dense(1 => 1)
        ps, st = Lux.setup(rng, nn) |> dev
        ps_ca = ComponentArray(ps)

        function test_f(x, ps)
            mz, back = Zygote.pullback(x -> first(nn(x, ps, st)), x)
            ep = only(back(one.(mz)))
            return sum(mz) + sum(ep)
        end

        @testset "Named Tuple Parameters" begin
            @test_nowarn test_f(x, ps)

            @test begin
                y, back = Zygote.pullback(test_f, x, ps)
                ∂x, ∂ps = back(one(y))
                ∂x !== nothing && ∂ps !== nothing
            end

            # Weird Zygote Quirks
            @test_broken begin
                ∂x, ∂ps = Zygote.jacobian(test_f, x, ps)
                ∂x !== nothing && ∂ps !== nothing
            end
        end

        @testset "Component Array Parameters" begin
            @test_nowarn test_f(x, ps_ca)

            @test begin
                y, back = Zygote.pullback(test_f, x, ps_ca)
                ∂x, ∂ps = back(one(y))
                ∂x !== nothing && ∂ps !== nothing
            end

            # Weird Zygote Quirks
            @test begin
                ∂x, ∂ps = Zygote.jacobian(test_f, x, ps_ca)
                ∂x !== nothing && ∂ps !== nothing
            end
        end
    end
end

@testitem "Tracing AD: AoS to SoA" setup=[SharedTestSetup] begin
    using ReverseDiff, Tracker

    rng = get_stable_rng()

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
