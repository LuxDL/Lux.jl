@testitem "Debugging Tools: DimensionMismatch" setup=[SharedTestSetup] tags=[:contrib] begin
    using Logging

    rng=StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        model = Chain(
            Dense(1 => 16, relu), Chain(Dense(16 => 3), Dense(1 => 1)), BatchNorm(1))

        ps, st = Lux.setup(rng, model) |> dev
        x = randn(rng, Float32, 1, 5) |> aType

        @test_throws DimensionMismatch model(x, ps, st)

        model_debug = Lux.Experimental.@debug_mode model

        @test_throws DimensionMismatch model_debug(x, ps, st)
        @test_logs (:info,) (:error,
            "Layer Dense(1 => 1) failed!! This layer is present at location KeyPath(:model, :layers, :layer_2, :layers, :layer_2).") match_mode=:any try
            model_debug(x, ps, st)
        catch
        end

        model_debug = Lux.Experimental.@debug_mode model error_check=false

        @test_throws DimensionMismatch model_debug(x, ps, st)
        @test_logs min_level=Logging.Error try
            model_debug(x, ps, st)
        catch
        end

        model_fixed = Chain(
            Dense(1 => 16, relu), Chain(Dense(16 => 1), Dense(1 => 1)), BatchNorm(1))

        ps, st = Lux.setup(rng, model_fixed) |> dev

        @test model_fixed(x, ps, st) isa Any

        model_fixed_debug = Lux.Experimental.@debug_mode model_fixed

        @test_logs min_level=Logging.Error Zygote.gradient(
            sum ∘ first ∘ model_fixed_debug, x, ps, st)
    end
end

@testitem "Debugging Tools: NaN" setup=[SharedTestSetup] tags=[:contrib] begin
    using Logging, ChainRulesCore
    import ChainRulesCore as CRC

    rng=StableRNG(12345)

    offending_layer(x)=2 .* x

    function CRC.rrule(::typeof(offending_layer), x)
        y=offending_layer(x)
        function ∇offending_layer(Δ)
            Δ[1:1].=NaN
            return NoTangent(), Δ
        end
        return y, ∇offending_layer
    end

    @testset "$mode: NaN Debugging" for (mode, aType, dev, ongpu) in MODES
        model = Chain(
            Dense(1 => 16, relu), Chain(Dense(16 => 1), Dense(1 => 1)), BatchNorm(1))

        x = randn(rng, Float32, 1, 5) |> aType
        ps, st = Lux.setup(rng, model) |> dev

        model_debug = Lux.Experimental.@debug_mode model nan_check=:both

        ps.layer_2.layer_2.weight .*= NaN32

        @test any(isnan, first(model(x, ps, st)) |> Array)

        @test_throws DomainError model_debug(x, ps, st)

        model_debug2 = Lux.Experimental.@debug_mode model nan_check=:forward

        @test_throws DomainError model_debug2(x, ps, st)

        model_debug3 = Lux.Experimental.@debug_mode model nan_check=:backward
        @test any(isnan, first(model_debug3(x, ps, st)) |> Array)

        model_debug4 = Lux.Experimental.@debug_mode model nan_check=:none
        @test any(isnan, first(model_debug4(x, ps, st)) |> Array)

        model = Chain(
            Dense(1 => 16, relu), Chain(Dense(16 => 1), offending_layer), BatchNorm(1))

        ps, st = Lux.setup(rng, model) |> dev

        @test !any(isnan, first(model(x, ps, st)) |> Array)

        gs = only(Zygote.gradient(ps -> sum(first(model(x, ps, st))), ps))
        @test any(isnan, gs.layer_1.weight)
        @test any(isnan, gs.layer_1.bias)
        @test any(isnan, gs.layer_2.layer_1.weight)
        @test any(isnan, gs.layer_2.layer_1.bias)
        @test gs.layer_2.layer_2 === nothing
        @test !any(isnan, gs.layer_3.scale)
        @test !any(isnan, gs.layer_3.bias)

        model_debug = Lux.Experimental.@debug_mode model nan_check=:both

        @test_logs min_level=Logging.Error model_debug(x, ps, st)

        @test_throws DomainError only(Zygote.gradient(
            ps -> sum(first(model_debug(x, ps, st))), ps))

        model_debug2 = Lux.Experimental.@debug_mode model nan_check=:forward

        @test_logs min_level=Logging.Error model_debug2(x, ps, st)

        gs = only(Zygote.gradient(ps -> sum(first(model_debug2(x, ps, st))), ps))
        @test any(isnan, gs.layer_1.weight)
        @test any(isnan, gs.layer_1.bias)
        @test any(isnan, gs.layer_2.layer_1.weight)
        @test any(isnan, gs.layer_2.layer_1.bias)
        @test gs.layer_2.layer_2 === nothing
        @test !any(isnan, gs.layer_3.scale)
        @test !any(isnan, gs.layer_3.bias)

        model_debug3 = Lux.Experimental.@debug_mode model nan_check=:backward

        @test_logs min_level=Logging.Error model_debug3(x, ps, st)

        @test_throws DomainError only(Zygote.gradient(
            ps -> sum(first(model_debug3(x, ps, st))), ps))

        model_debug4 = Lux.Experimental.@debug_mode model nan_check=:none

        @test_logs min_level=Logging.Error model_debug4(x, ps, st)

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
