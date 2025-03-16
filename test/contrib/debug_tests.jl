@testitem "Debugging Tools: DimensionMismatch" setup = [SharedTestSetup] tags = [:misc] begin
    using Logging

    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = Chain(
            Dense(1 => 16, relu), Chain(Dense(16 => 3), Dense(1 => 1)), BatchNorm(1)
        )

        ps, st = dev(Lux.setup(rng, model))
        x = aType(randn(rng, Float32, 1, 5))

        @test_throws DimensionMismatch model(x, ps, st)

        model_debug = Lux.Experimental.@debug_mode model

        @test_throws DimensionMismatch model_debug(x, ps, st)
        # XXX this is a bit flaky in CI on 1.11+
        if VERSION < v"1.11-"
            @test_logs (:info,) (
                :error,
                "Layer Dense(1 => 1) failed!! This layer is present at location KeyPath(:model, :layers, :layer_2, :layers, :layer_2).",
            ) match_mode = :any try
                model_debug(x, ps, st)
            catch
            end
        end

        model_debug = Lux.Experimental.@debug_mode model error_check = false

        @test_throws DimensionMismatch model_debug(x, ps, st)
        # XXX this is a bit flaky in CI on 1.11+
        if VERSION < v"1.11-"
            @test_logs min_level = Logging.Error try
                model_debug(x, ps, st)
            catch
            end
        end

        model_fixed = Chain(
            Dense(1 => 16, relu), Chain(Dense(16 => 1), Dense(1 => 1)), BatchNorm(1)
        )

        ps, st = dev(Lux.setup(rng, model_fixed))

        @test model_fixed(x, ps, st) isa Any

        model_fixed_debug = Lux.Experimental.@debug_mode model_fixed

        # XXX this is a bit flaky in CI on 1.11+
        if VERSION < v"1.11-"
            @test_logs min_level = Logging.Error Zygote.gradient(
                sum ∘ first ∘ model_fixed_debug, x, ps, st
            )
        end
    end
end

@testitem "Debugging Tools: NaN" setup = [SharedTestSetup] tags = [:misc] begin
    using Logging, ChainRulesCore
    import ChainRulesCore as CRC

    rng = StableRNG(12345)

    offending_layer(x) = 2 .* x

    function CRC.rrule(::typeof(offending_layer), x)
        y = offending_layer(x)
        function ∇offending_layer(Δ)
            problematicΔ = CRC.@thunk begin
                Δ = CRC.unthunk(Δ)
                Δ[1:1] .= NaN
                return Δ
            end
            return NoTangent(), problematicΔ
        end
        return y, ∇offending_layer
    end

    @testset "$mode: NaN Debugging" for (mode, aType, dev, ongpu) in MODES
        model = Chain(
            Dense(1 => 16, relu), Chain(Dense(16 => 1), Dense(1 => 1)), BatchNorm(1)
        )

        x = aType(randn(rng, Float32, 1, 5))
        ps, st = dev(Lux.setup(rng, model))

        model_debug = Lux.Experimental.@debug_mode model nan_check = :both

        ps.layer_2.layer_2.weight .*= NaN32

        @test any(isnan, Array(first(model(x, ps, st))))

        @test_throws DomainError model_debug(x, ps, st)

        model_debug2 = Lux.Experimental.@debug_mode model nan_check = :forward

        @test_throws DomainError model_debug2(x, ps, st)

        model_debug3 = Lux.Experimental.@debug_mode model nan_check = :backward
        @test any(isnan, Array(first(model_debug3(x, ps, st))))

        model_debug4 = Lux.Experimental.@debug_mode model nan_check = :none
        @test any(isnan, Array(first(model_debug4(x, ps, st))))

        model = Chain(
            Dense(1 => 16, relu), Chain(Dense(16 => 1), offending_layer), BatchNorm(1)
        )

        ps, st = dev(Lux.setup(rng, model))

        @test !any(isnan, Array(first(model(x, ps, st))))

        gs = only(Zygote.gradient(ps -> sum(first(model(x, ps, st))), ps))
        @test any(isnan, gs.layer_1.weight)
        @test any(isnan, gs.layer_1.bias)
        @test any(isnan, gs.layer_2.layer_1.weight)
        @test any(isnan, gs.layer_2.layer_1.bias)
        @test gs.layer_2.layer_2 === nothing
        @test !any(isnan, gs.layer_3.scale)
        @test !any(isnan, gs.layer_3.bias)

        model_debug = Lux.Experimental.@debug_mode model nan_check = :both

        @test_logs min_level = Logging.Error model_debug(x, ps, st)

        @test_throws DomainError only(
            Zygote.gradient(ps -> sum(first(model_debug(x, ps, st))), ps)
        )

        model_debug2 = Lux.Experimental.@debug_mode model nan_check = :forward

        @test_logs min_level = Logging.Error model_debug2(x, ps, st)

        gs = only(Zygote.gradient(ps -> sum(first(model_debug2(x, ps, st))), ps))
        @test any(isnan, gs.layer_1.weight)
        @test any(isnan, gs.layer_1.bias)
        @test any(isnan, gs.layer_2.layer_1.weight)
        @test any(isnan, gs.layer_2.layer_1.bias)
        @test gs.layer_2.layer_2 === nothing
        @test !any(isnan, gs.layer_3.scale)
        @test !any(isnan, gs.layer_3.bias)

        model_debug3 = Lux.Experimental.@debug_mode model nan_check = :backward

        @test_logs min_level = Logging.Error model_debug3(x, ps, st)

        @test_throws DomainError only(
            Zygote.gradient(ps -> sum(first(model_debug3(x, ps, st))), ps)
        )

        model_debug4 = Lux.Experimental.@debug_mode model nan_check = :none

        @test_logs min_level = Logging.Error model_debug4(x, ps, st)

        gs = only(Zygote.gradient(ps -> sum(first(model_debug4(x, ps, st))), ps))
        @test any(isnan, gs.layer_1.weight)
        @test any(isnan, gs.layer_1.bias)
        @test any(isnan, gs.layer_2.layer_1.weight)
        @test any(isnan, gs.layer_2.layer_1.bias)
        @test gs.layer_2.layer_2 === nothing
        @test !any(isnan, gs.layer_3.scale)
        @test !any(isnan, gs.layer_3.bias)
    end
end

@testitem "Debugging Tools: Issue #1068" setup = [SharedTestSetup] tags = [:misc] begin
    model = Chain(
        Conv((3, 3), 3 => 16, relu; stride=2),
        MaxPool((2, 2)),
        AdaptiveMaxPool((2, 2)),
        GlobalMaxPool(),
    )

    model_debug = Lux.Experimental.@debug_mode model
    display(model_debug)

    @test model_debug[1] isa Lux.Experimental.DebugLayer
    @test model_debug[2] isa Lux.Experimental.DebugLayer
    @test model_debug[3] isa Lux.Experimental.DebugLayer
    @test model_debug[4] isa Lux.Experimental.DebugLayer
end
