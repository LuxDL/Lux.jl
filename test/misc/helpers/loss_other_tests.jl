using ForwardDiff, Zygote, Enzyme, OneHotArrays
using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent

const CRC = ChainRulesCore

include("../../shared_testsetup.jl")

@testset "LuxOps.xlogx & LuxOps.xlogy" begin
    @test iszero(LuxOps.xlogx(0))
    @test isnan(LuxOps.xlogx(NaN))
    @test LuxOps.xlogx(2) ≈ 2.0 * log(2.0)

    ∂x1 = ForwardDiff.derivative(LuxOps.xlogx, 2.0)
    ∂x2 = Zygote.gradient(LuxOps.xlogx, 2.0)[1]
    @test ∂x1 ≈ ∂x2

    @test @inferred(LuxOps.xlogx(2)) isa Number
    @test @inferred(LuxOps.xlogx(0)) isa Number
    @jet LuxOps.xlogx(2)

    @test iszero(LuxOps.xlogy(0, 1))
    @test isnan(LuxOps.xlogy(NaN, 1))
    @test isnan(LuxOps.xlogy(1, NaN))
    @test isnan(LuxOps.xlogy(NaN, NaN))
    @test LuxOps.xlogy(2, 3) ≈ 2.0 * log(3.0)

    ∂x1 = ForwardDiff.derivative(Base.Fix2(LuxOps.xlogy, 3.0), 2.0)
    ∂y1 = ForwardDiff.derivative(Base.Fix1(LuxOps.xlogy, 2.0), 3.0)
    ∂x2, ∂y2 = Zygote.gradient(LuxOps.xlogy, 2.0, 3.0)
    if LuxTestUtils.ENZYME_TESTING_ENABLED[]
        ((∂x3, ∂y3),) = Enzyme.autodiff(
            Enzyme.Reverse, LuxOps.xlogy, Active, Active(2.0), Active(3.0)
        )
        @test ∂x1 ≈ ∂x2 ≈ ∂x3
        @test ∂y1 ≈ ∂y2 ≈ ∂y3
    else
        @test ∂x1 ≈ ∂x2
        @test ∂y1 ≈ ∂y2
        @test_broken false
    end

    @test @inferred(LuxOps.xlogy(2, 3)) isa Number
    @test @inferred(LuxOps.xlogy(0, 1)) isa Number
    @jet LuxOps.xlogy(2, 3)

    if LuxTestUtils.ENZYME_TESTING_ENABLED[]
        @test @inferred(
            Enzyme.autodiff(Enzyme.Reverse, LuxOps.xlogy, Active, Active(2.0), Active(3.0))
        ) isa Any
    else
        @test_broken false
    end

    ∂x_zero, ∂y_zero = Zygote.gradient(LuxOps.xlogy, 0.0, 0.5)
    @test (∂x_zero === nothing || iszero(∂x_zero)) &&
        (∂y_zero === nothing || iszero(∂y_zero))
    @test something.(Zygote.gradient(LuxOps.xlogy, 0.0, 0.5), 0.0) == (0.0, 0.0)

    ∂x_zero0, ∂y_zero0 = Zygote.gradient(LuxOps.xlogy, 0.0, 0.0)
    @test (∂x_zero0 === nothing || iszero(∂x_zero0)) &&
        (∂y_zero0 === nothing || iszero(∂y_zero0))
    @test something.(Zygote.gradient(LuxOps.xlogy, 0.0, 0.0), 0.0) == (0.0, 0.0)

    val_xy, pb_xy = CRC.rrule(LuxOps.xlogy, 0.0, 0.5)
    @test iszero(val_xy) && val_xy isa Float64
    @test pb_xy(1.0) == (NoTangent(), ZeroTangent(), ZeroTangent())

    val_xy0, pb_xy0 = CRC.rrule(LuxOps.xlogy, 0.0, 0.0)
    @test iszero(val_xy0) && val_xy0 isa Float64
    @test pb_xy0(1.0) == (NoTangent(), ZeroTangent(), ZeroTangent())

    val_xy_int, pb_xy_int = CRC.rrule(LuxOps.xlogy, 0, 2.0)
    @test iszero(val_xy_int) && val_xy_int isa Float64
    @test pb_xy_int(1.0) == (NoTangent(), ZeroTangent(), ZeroTangent())

    val_xx, pb_xx = CRC.rrule(LuxOps.xlogx, 0.0)
    @test iszero(val_xx) && val_xx isa Float64
    @test pb_xx(1.0) == (NoTangent(), ZeroTangent())

    val_xx_int, pb_xx_int = CRC.rrule(LuxOps.xlogx, 0)
    @test iszero(val_xx_int) && val_xx_int isa Float64
    @test pb_xx_int(1.0) == (NoTangent(), ZeroTangent())

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        x = aType(rand(10))
        @test_gradients(
            sum ∘ Broadcast.BroadcastFunction(LuxOps.xlogx),
            x;
            atol=1.0f-3,
            rtol=1.0f-3,
            soft_fail=[AutoFiniteDiff()]
        )

        y = aType(rand(10))
        @test_gradients(
            sum ∘ Broadcast.BroadcastFunction(LuxOps.xlogy),
            x,
            y;
            atol=1.0f-3,
            rtol=1.0f-3,
            soft_fail=[AutoFiniteDiff()]
        )

        x_zeros = aType([0.0, 0.5, 1.0])
        f_x = LuxOps.xlogx.(x_zeros)
        @test all(!isnan, Array(f_x))
        @test all(isfinite, Array(f_x))
        @test Array(f_x) ≈ [LuxOps.xlogx(0.0), LuxOps.xlogx(0.5), LuxOps.xlogx(1.0)]

        (dx,) = Zygote.gradient(x -> sum(LuxOps.xlogx.(x)), x_zeros)
        @test all(!isnan, Array(dx))
        @test all(isfinite, Array(dx))
        @test Array(dx)[1] ≈ 0.0
        @test Array(dx)[2] ≈ log(0.5) + 1.0
        @test Array(dx)[3] ≈ log(1.0) + 1.0

        x_zeros_y = aType([0.0, 0.5])
        y_zeros = aType([0.0, 0.25])
        f_xy = LuxOps.xlogy.(x_zeros_y, y_zeros)
        @test all(!isnan, Array(f_xy))
        @test all(isfinite, Array(f_xy))
        @test Array(f_xy) ≈ [LuxOps.xlogy(0.0, 0.0), LuxOps.xlogy(0.5, 0.25)]

        dx_y, dy_x = Zygote.gradient((x, y) -> sum(LuxOps.xlogy.(x, y)), x_zeros_y, y_zeros)
        @test all(!isnan, Array(dx_y))
        @test all(isfinite, Array(dx_y))
        @test all(!isnan, Array(dy_x))
        @test all(isfinite, Array(dy_x))
        @test Array(dx_y)[1] ≈ 0.0
        @test Array(dx_y)[2] ≈ log(0.25)
        @test Array(dy_x)[1] ≈ 0.0
        @test Array(dy_x)[2] ≈ 0.5 / 0.25
    end
end

@testset "Other Losses" begin
    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "KLDivergenceLoss" begin
            y = aType([1 2 3])
            ŷ = aType([4.0 5.0 6.0])

            @test KLDivergenceLoss()(aType([0.1, 0.0, 0.9]), aType([0.1, 0.0, 0.9])) ≈
                KLDivergenceLoss()(aType([0.1, 0.9]), aType([0.1, 0.9]))
            @test KLDivergenceLoss()(ŷ, y) ≈ -1.7661057888493457
            @test KLDivergenceLoss()(y, y) ≈ 0

            @jet KLDivergenceLoss()(ŷ, y)

            @test_gradients(
                Base.Fix2(KLDivergenceLoss(), y),
                ŷ;
                atol=1.0f-3,
                rtol=1.0f-3,
                skip_backends=VERSION ≥ v"1.11-" ? [AutoEnzyme()] : []
            )

            y_zeros = aType([0.1, 0.0, 0.9])
            ŷ_zeros = aType([0.2, 0.1, 0.7])
            (d_ŷ,) = Zygote.gradient(ŷ -> KLDivergenceLoss()(ŷ, y_zeros), ŷ_zeros)
            @test all(!isnan, Array(d_ŷ))
            @test all(isfinite, Array(d_ŷ))
            (d_y,) = Zygote.gradient(y -> KLDivergenceLoss()(ŷ_zeros, y), y_zeros)
            @test all(!isnan, Array(d_y))
            @test all(isfinite, Array(d_y))
        end

        @testset "HingeLoss" begin
            y = aType([1, 2, 3, 4])
            ŷ = aType([5.0, 6.0, 7.0, 8.0])

            @test Lux.HingeLoss()(ŷ, y) ≈ 0
            @test Lux.HingeLoss()(y, 0.5 .* y) ≈ 0.125

            @jet Lux.HingeLoss()(ŷ, y)

            __f = Base.Fix2(Lux.HingeLoss(), y)
            @test_gradients(__f, ŷ; atol=1.0f-3, rtol=1.0f-3)

            y_bin = aType([1, -1, 1, -1])
            ŷ_bin = aType([0.2, 0.3, -0.4, -1.5])

            @test Lux.HingeLoss()(ŷ_bin, y_bin) ≈ 0.875
            @jet Lux.HingeLoss()(ŷ_bin, y_bin)

            __f_bin = Base.Fix2(Lux.HingeLoss(), y_bin)
            @test_gradients(__f_bin, ŷ_bin; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "SquaredHingeLoss" begin
            y = aType([1, 2, 3, 4])
            ŷ = aType([5.0, 6.0, 7.0, 8.0])

            @test SquaredHingeLoss()(ŷ, y) ≈ 0
            @test SquaredHingeLoss()(y, 0.5 .* y) ≈ 0.0625

            @jet SquaredHingeLoss()(ŷ, y)

            __f = Base.Fix2(SquaredHingeLoss(), y)
            @test_gradients(__f, ŷ; atol=1.0f-3, rtol=1.0f-3)

            y_bin = aType([1, -1, 1, -1])
            ŷ_bin = aType([0.2, 0.3, -0.4, -1.5])

            @test SquaredHingeLoss()(ŷ_bin, y_bin) ≈ 1.0725
            @jet SquaredHingeLoss()(ŷ_bin, y_bin)

            __f_bin = Base.Fix2(SquaredHingeLoss(), y_bin)
            @test_gradients(__f_bin, ŷ_bin; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "PoissonLoss" begin
            y = aType([0.1, 0.2, 0.3])
            ŷ = aType([0.4, 0.5, 0.6])

            @test Lux.PoissonLoss()(ŷ, y) ≈ 0.6278353988097339
            @test Lux.PoissonLoss()(y, y) ≈ 0.5044459776946685

            @jet Lux.PoissonLoss()(ŷ, y)

            __f = Base.Fix2(Lux.PoissonLoss(), y)
            @test_gradients(__f, ŷ; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "DiceCoeffLoss" begin
            y = aType([1.0, 0.5, 0.3, 2.4])
            ŷ = aType([0.0, 1.4, 0.5, 1.2])

            @test DiceCoeffLoss()(ŷ, y) ≈ 0.2799999999999999
            @test DiceCoeffLoss()(y, y) ≈ 0.0

            @jet DiceCoeffLoss()(ŷ, y)

            __f = Base.Fix2(DiceCoeffLoss(), y)
            @test_gradients(__f, ŷ; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "Siamese Contrastive Loss" begin
            y = aType([
                1 0
                0 0
                0 1
            ])
            ŷ = aType([
                0.4 0.2
                0.5 0.5
                0.1 0.3
            ])
            y1 = aType([
                1 0 0 0 1
                0 1 0 1 0
                0 0 1 0 0
            ])
            ŷ1 = aType(softmax(reshape(-7:7, 3, 5) .* 1.0f0))
            y2 = aType([
                1
                0
                0
                1
                1
            ])
            ŷ2 = aType([
                0.6
                0.4
                0.1
                0.2
                0.7
            ])

            @test SiameseContrastiveLoss()(ŷ, y) ≈ 0.2333333333333333
            @test SiameseContrastiveLoss(; margin=0.5f0)(ŷ, y) ≈ 0.10000000000000002
            @test SiameseContrastiveLoss(; margin=1.5f0)(ŷ, y) ≈ 0.5333333333333333
            @test SiameseContrastiveLoss()(ŷ1, y1) ≈ 0.32554644f0
            @test SiameseContrastiveLoss(; margin=0.5f0)(ŷ1, y1) ≈ 0.16271012f0
            @test SiameseContrastiveLoss(; margin=1.5f0)(ŷ1, y1) ≈ 0.6532292f0
            @test SiameseContrastiveLoss(; margin=1)(ŷ, y) ≈ SiameseContrastiveLoss()(ŷ, y)
            @test SiameseContrastiveLoss()(y, y) ≈ 0.0
            @test SiameseContrastiveLoss()(y1, y1) ≈ 0.0
            @test SiameseContrastiveLoss(; margin=0)(ŷ, y) ≈ 0.09166666666666667
            @test SiameseContrastiveLoss(; margin=0)(ŷ1, y1) ≈ 0.13161165f0
            @test SiameseContrastiveLoss()(ŷ2, y2) ≈ 0.21200000000000005
            @test SiameseContrastiveLoss()(ŷ2, ŷ2) ≈ 0.18800000000000003

            @jet SiameseContrastiveLoss()(ŷ, y)

            @test_throws AssertionError SiameseContrastiveLoss(; margin=-0.5)
            @test_throws AssertionError SiameseContrastiveLoss(; margin=-1)
        end
    end
end

@testset "Losses: Error Checks and Misc" begin
    @testset "Size Checks" begin
        @test_throws DimensionMismatch MSELoss()([1, 2], [1, 2, 3])
    end

    @testset "No Aggregation" begin
        @test MSELoss(; agg=nothing)([1, 3], [3, 1]) == [4, 4]
    end

    @testset "Scalar Loss" begin
        @test MSELoss(; agg=sum)(1.0, 1.0) == 0.0
        @test MSELoss(; agg=sum)(2.0, 0.0) == 4.0
        @test MSLELoss(; agg=sum)(2.0, 0.0) ≈
            Lux.LossFunctionImpl.msle_loss(2.0, 0.0, nothing)
    end

    @testset "LossFunctionsExt Enzyme & AD" begin
        if Base.find_package("LossFunctions") !== nothing
            @eval using LossFunctions: LossFunctions
            Base.invokelatest() do
                lfn = LossFunctions.L2DistLoss()
                loss_fn = GenericLossFunction(lfn; agg=sum)
                x = [1.0, 2.0, 3.0]
                y = [0.5, 1.5, 2.0]

                @test loss_fn(x, y) ≈ sum((x .- y) .^ 2)

                f = (x, y) -> loss_fn(x, y) + sum(2.0 .* x)
                expected_g = 2.0 .* (x .- y) .+ 2.0
                @test only(Zygote.gradient(x -> f(x, y), x)) ≈ expected_g

                # Direct EnzymeRules.reverse cotangent accumulation check
                cfg = Enzyme.EnzymeRules.RevConfig{
                    false,false,1,(false, false, false, false, false),false,false
                }()
                dx_accum = [10.0, 10.0, 10.0]
                Enzyme.EnzymeRules.reverse(
                    cfg,
                    Enzyme.Const(Lux.LossFunctionImpl.fused_agg),
                    Enzyme.Active(1.0),
                    (nothing, nothing),
                    Enzyme.Const(sum),
                    Enzyme.Const(lfn),
                    Enzyme.Duplicated(copy(x), dx_accum),
                    Enzyme.Const(y),
                )
                @test dx_accum ≈ 2.0 .* (x .- y) .+ 10.0
            end
        else
            @test_skip false
        end
    end
end

@testset "LossFunctionsExt" begin
    using LossFunctions: LossFunctions

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        y = aType([1.0, 1.0, 0.0, 0.0])
        ŷ = aType([0.9, 0.1, 0.1, 0.9])

        loss_sum = GenericLossFunction(LossFunctions.L2DistLoss(); agg=sum)
        loss_mean = GenericLossFunction(LossFunctions.L2DistLoss(); agg=mean)

        loss_res = (0.1^2 + 0.9^2) / 2

        @test loss_mean(ŷ, y) ≈ loss_res
        @test loss_sum(ŷ, y) ≈ loss_res * 4

        @jet loss_sum(ŷ, y)
        @jet loss_mean(ŷ, y)

        @test_gradients(Base.Fix2(loss_sum, y), ŷ; atol=1.0f-3, rtol=1.0f-3)
        @test_gradients(Base.Fix2(loss_mean, y), ŷ; atol=1.0f-3, rtol=1.0f-3)
    end
end
