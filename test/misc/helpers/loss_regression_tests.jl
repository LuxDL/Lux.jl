using ForwardDiff, Zygote, Enzyme, OneHotArrays
using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent

const CRC = ChainRulesCore

include("../../shared_testsetup.jl")

@testset "Regression Loss" begin
    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        y = aType([1.0, 1.0, 0.0, 0.0])
        ŷ = aType([0.9, 0.1, 0.1, 0.9])

        loss_res_map = Dict(
            "MSE" => (0.1^2 + 0.9^2) / 2, "MAE" => (0.1 + 0.9) / 2, "Huber" => 0.205
        )

        @testset "$(loss)" for (loss, loss_res) in loss_res_map
            loss_mean = eval(Symbol(loss * "Loss"))()
            loss_sum = eval(Symbol(loss * "Loss"))(; agg=sum)
            loss_sum2 = eval(Symbol(loss * "Loss"))(; agg=(args...) -> sum(args...))

            @test loss_mean(ŷ, y) ≈ loss_res
            @test loss_sum(ŷ, y) ≈ loss_res * 4
            @test loss_sum2(ŷ, y) ≈ loss_res * 4

            @jet loss_mean(ŷ, y)
            @jet loss_sum(ŷ, y)

            @test_gradients(Base.Fix2(loss_mean, y), ŷ; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "MSLE" begin
            y = aType([123.0, 456.0, 789.0])
            ŷ = aType([345.0, 332.0, 789.0])

            @test MSLELoss()(ŷ, y) ≈ 0.38813985859136585

            @jet MSLELoss()(ŷ, y)

            @test_gradients(Base.Fix2(MSLELoss(), y), ŷ; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "GenericLossFunction (looped rrule)" begin
            custom_op = (ŷ, y) -> abs2(ŷ - y)
            @test !Lux.LossFunctionImpl.has_custom_derivative(custom_op)

            loss_fn = GenericLossFunction(custom_op; agg=sum)
            y = aType([1.0f0, 2.0f0, 3.0f0])
            ŷ = aType([2.0f0, 1.0f0, 4.0f0])

            @test loss_fn(ŷ, y) ≈ 3.0f0
            @jet loss_fn(ŷ, y)
            @test_gradients(Base.Fix2(loss_fn, y), ŷ; atol=1.0f-3, rtol=1.0f-3)
        end
    end
end
