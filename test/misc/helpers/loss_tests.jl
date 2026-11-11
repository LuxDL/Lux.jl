using ForwardDiff, Zygote, Enzyme, OneHotArrays

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
    end
end

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
    end
end

@testset "Classification Loss" begin
    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        y = dev(onehotbatch([1, 1, 0, 0], 0:1))
        y_smoothed = Lux.LossFunctionImpl.label_smoothing(0.1, y, Float32)

        ŷ = dev([0.1 0.9; 0.9 0.1; 0.9 0.1; 0.1 0.9]')
        v = log(0.1 / 0.9)
        logŷ = dev([v 0.0; 0.0 v; 0.0 v; v 0.0]')
        lossvalue = 1.203972804325936
        lossvalue_smoothed = 1.2039728043259348

        yl = dev(onehotbatch([1], 0:1))
        sf = 0.1
        yls = dev([sf (1 - sf)]')
        ylp = dev([0.9 0.1]')
        logylp = dev([0.0 v]')

        ya = dev(onehotbatch([1, 1, 1, 0, 0], 0:1))
        ya_smoothed = Lux.LossFunctionImpl.label_smoothing(2sf, ya, Float32)
        y_same = Float32.(ya)
        y_sim = y_same .* (1 - 2 * sf) .+ sf
        y_dis = copy(y_sim)
        y_dis[1, :], y_dis[2, :] = y_dis[2, :], y_dis[1, :]

        @testset "CrossEntropyLoss" begin
            celoss = CrossEntropyLoss()

            @test celoss(aType([0.1, 0.0, 0.9]), aType([0.1, 0.0, 0.9])) ≈
                celoss(aType([0.1, 0.9]), aType([0.1, 0.9]))

            @test celoss(ŷ, y) ≈ lossvalue
            @test celoss(ŷ, y_smoothed) ≈ lossvalue_smoothed

            celoss_smooth = CrossEntropyLoss(; label_smoothing=0.1)
            @test celoss_smooth(ŷ, y) ≈ lossvalue_smoothed

            celoss_smooth2 = CrossEntropyLoss(; label_smoothing=2sf)
            @test celoss_smooth2(ylp, yl) ≈ sum(-yls .* log.(ylp))

            @test celoss(ylp, yl) ≈ sum(-yl .* log.(ylp))

            @test iszero(CrossEntropyLoss(; epsilon=0)(y_same, ya))

            @test celoss(y_sim, ya) < celoss_smooth(y_sim, ya)
            @test celoss(y_dis, ya) > celoss_smooth(y_dis, ya)

            @jet celoss(ŷ, y)
            @jet celoss_smooth(ŷ, y)

            @test_gradients(
                Base.Fix2(celoss, y),
                ŷ;
                atol=1.0f-3,
                rtol=1.0f-3,
                skip_backends=VERSION ≥ v"1.11-" ? [AutoEnzyme()] : [],
            )

            # 1D vector inputs
            y_1d = dev([0.0f0, 1.0f0])
            ŷ_1d = dev([0.2f0, 0.8f0])
            y_1d_smooth = dev(y_1d .* 0.9f0 .+ 0.1f0 / 2)
            @test celoss_smooth(ŷ_1d, y_1d) ≈ celoss(ŷ_1d, y_1d_smooth)
            @jet celoss_smooth(ŷ_1d, y_1d)
            @test_gradients(
                Base.Fix2(celoss_smooth, y_1d),
                ŷ_1d;
                atol=1.0f-3,
                rtol=1.0f-3,
                skip_backends=VERSION ≥ v"1.11-" ? [AutoEnzyme()] : [],
            )

            # 3D tensor inputs (C, W, B) with dims=1
            y_3d = dev(reshape(Float32[1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0], 3, 2, 2))
            ŷ_3d = dev(softmax(randn(Float32, 3, 2, 2); dims=1))
            y_3d_smooth = dev(y_3d .* 0.9f0 .+ 0.1f0 / 3)
            @test celoss_smooth(ŷ_3d, y_3d) ≈ celoss(ŷ_3d, y_3d_smooth)
            @jet celoss_smooth(ŷ_3d, y_3d)
            @test_gradients(
                Base.Fix2(celoss_smooth, y_3d),
                ŷ_3d;
                atol=1.0f-3,
                rtol=1.0f-3,
                skip_backends=VERSION ≥ v"1.11-" ? [AutoEnzyme()] : [],
            )

            # 4D tensor inputs (C, H, W, B) with dims=1
            y_4d = dev(softmax(randn(Float32, 3, 2, 2, 2); dims=1))
            ŷ_4d = dev(softmax(randn(Float32, 3, 2, 2, 2); dims=1))
            y_4d_smooth = dev(y_4d .* 0.9f0 .+ 0.1f0 / 3)
            @test celoss_smooth(ŷ_4d, y_4d) ≈ celoss(ŷ_4d, y_4d_smooth)
            @jet celoss_smooth(ŷ_4d, y_4d)
            @test_gradients(
                Base.Fix2(celoss_smooth, y_4d),
                ŷ_4d;
                atol=1.0f-3,
                rtol=1.0f-3,
                skip_backends=VERSION ≥ v"1.11-" ? [AutoEnzyme()] : [],
            )

            # dims=2 for (B, C)
            celoss_smooth_dim2 = CrossEntropyLoss(; dims=2, label_smoothing=0.1)
            y_bc = dev(Float32[1 0 0; 0 1 0; 0 0 1; 1 0 0])
            ŷ_bc = dev(softmax(randn(Float32, 4, 3); dims=2))
            y_bc_smooth = dev(y_bc .* 0.9f0 .+ 0.1f0 / 3)
            @test celoss_smooth_dim2(ŷ_bc, y_bc) ≈
                CrossEntropyLoss(; dims=2)(ŷ_bc, y_bc_smooth)
            @jet celoss_smooth_dim2(ŷ_bc, y_bc)

            # dims=(:)
            celoss_smooth_colon = CrossEntropyLoss(; dims=(:), label_smoothing=0.1)
            y_colon = dev(Float32[1, 0, 0, 0])
            ŷ_colon = dev(softmax(randn(Float32, 4); dims=(:)))
            y_colon_smooth = dev(y_colon .* 0.9f0 .+ 0.1f0 / 4)
            @test celoss_smooth_colon(ŷ_colon, y_colon) ≈
                CrossEntropyLoss(; dims=(:))(ŷ_colon, y_colon_smooth)
            @jet celoss_smooth_colon(ŷ_colon, y_colon)
        end

        @testset "Logit CrossEntropyLoss" begin
            logitceloss = CrossEntropyLoss(; logits=Val(true))

            @test logitceloss(logŷ, y) ≈ lossvalue
            @test logitceloss(logylp, yl) ≈ sum(-yl .* log.(softmax(logylp)))

            logitceloss_smooth = CrossEntropyLoss(; logits=Val(true), label_smoothing=0.1)

            @test logitceloss(logŷ, y_smoothed) ≈ lossvalue_smoothed
            @test logitceloss_smooth(logŷ, y) ≈ lossvalue_smoothed

            logitceloss_smooth2 = CrossEntropyLoss(; logits=Val(true), label_smoothing=2sf)
            @test logitceloss_smooth2(logylp, yl) ≈ sum(-yls .* log.(softmax(logylp)))

            @jet logitceloss(logŷ, y)
            @jet logitceloss_smooth(logŷ, y)

            @test_gradients(
                Base.Fix2(logitceloss, y),
                logŷ;
                atol=1.0f-3,
                rtol=1.0f-3,
                skip_backends=VERSION ≥ v"1.11-" ? [AutoEnzyme()] : [],
            )

            # 1D vector inputs
            y_1d = dev([0.0f0, 1.0f0])
            logŷ_1d = dev([0.2f0, 0.8f0])
            y_1d_smooth = dev(y_1d .* 0.9f0 .+ 0.1f0 / 2)
            @test logitceloss_smooth(logŷ_1d, y_1d) ≈ logitceloss(logŷ_1d, y_1d_smooth)
            @jet logitceloss_smooth(logŷ_1d, y_1d)
            @test_gradients(
                Base.Fix2(logitceloss_smooth, y_1d),
                logŷ_1d;
                atol=1.0f-3,
                rtol=1.0f-3,
                skip_backends=VERSION ≥ v"1.11-" ? [AutoEnzyme()] : [],
            )

            # 3D tensor inputs (C, W, B) with dims=1
            y_3d = dev(reshape(Float32[1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0], 3, 2, 2))
            logŷ_3d = dev(randn(Float32, 3, 2, 2))
            y_3d_smooth = dev(y_3d .* 0.9f0 .+ 0.1f0 / 3)
            @test logitceloss_smooth(logŷ_3d, y_3d) ≈ logitceloss(logŷ_3d, y_3d_smooth)
            @jet logitceloss_smooth(logŷ_3d, y_3d)
            @test_gradients(
                Base.Fix2(logitceloss_smooth, y_3d),
                logŷ_3d;
                atol=1.0f-3,
                rtol=1.0f-3,
                skip_backends=VERSION ≥ v"1.11-" ? [AutoEnzyme()] : [],
            )

            # 4D tensor inputs (C, H, W, B) with dims=1
            y_4d = dev(softmax(randn(Float32, 3, 2, 2, 2); dims=1))
            logŷ_4d = dev(randn(Float32, 3, 2, 2, 2))
            y_4d_smooth = dev(y_4d .* 0.9f0 .+ 0.1f0 / 3)
            @test logitceloss_smooth(logŷ_4d, y_4d) ≈ logitceloss(logŷ_4d, y_4d_smooth)
            @jet logitceloss_smooth(logŷ_4d, y_4d)
            @test_gradients(
                Base.Fix2(logitceloss_smooth, y_4d),
                logŷ_4d;
                atol=1.0f-3,
                rtol=1.0f-3,
                skip_backends=VERSION ≥ v"1.11-" ? [AutoEnzyme()] : [],
            )

            # dims=2 for (B, C)
            logitceloss_smooth_dim2 = CrossEntropyLoss(;
                logits=Val(true), dims=2, label_smoothing=0.1
            )
            y_bc = dev(Float32[1 0 0; 0 1 0; 0 0 1; 1 0 0])
            logŷ_bc = dev(randn(Float32, 4, 3))
            y_bc_smooth = dev(y_bc .* 0.9f0 .+ 0.1f0 / 3)
            @test logitceloss_smooth_dim2(logŷ_bc, y_bc) ≈
                CrossEntropyLoss(; logits=Val(true), dims=2)(logŷ_bc, y_bc_smooth)
            @jet logitceloss_smooth_dim2(logŷ_bc, y_bc)

            # dims=(:)
            logitceloss_smooth_colon = CrossEntropyLoss(;
                logits=Val(true), dims=(:), label_smoothing=0.1
            )
            y_colon = dev(Float32[1, 0, 0, 0])
            logŷ_colon = dev(randn(Float32, 4))
            y_colon_smooth = dev(y_colon .* 0.9f0 .+ 0.1f0 / 4)
            @test logitceloss_smooth_colon(logŷ_colon, y_colon) ≈
                CrossEntropyLoss(; logits=Val(true), dims=(:))(
                logŷ_colon, y_colon_smooth
            )
            @jet logitceloss_smooth_colon(logŷ_colon, y_colon)
        end

        logŷ, y = aType(randn(3)), aType(rand(3))
        yls = y .* (1 - 2sf) .+ sf

        @testset "BinaryCrossEntropyLoss" begin
            bceloss = BinaryCrossEntropyLoss()
            bceloss_smooth = BinaryCrossEntropyLoss(; label_smoothing=2sf, epsilon=0)

            @test bceloss_smooth(σ.(logŷ), y) ≈
                -mean(yls .* log.(σ.(logŷ)) .+ (1 .- yls) .* log.(1 .- σ.(logŷ)))

            @test bceloss(σ.(logŷ), y) ≈
                mean(-y .* log.(σ.(logŷ)) .- (1 .- y) .* log.(1 .- σ.(logŷ)))

            @test bceloss(σ.(logŷ), y) ≈ mean(
                -y .* log.(σ.(logŷ) .+ eps.(σ.(logŷ))) -
                (1 .- y) .* log.(1 .- σ.(logŷ) .+ eps.(σ.(logŷ))),
            )

            @test bceloss(aType([0.1, 0.2, 0.9]), 1) ≈ -mean(log, aType([0.1, 0.2, 0.9]))  # constant label

            @jet bceloss(σ.(logŷ), y)
            @jet bceloss_smooth(σ.(logŷ), y)

            @test_gradients(Base.Fix2(bceloss, y), σ.(logŷ); atol=1.0f-3, rtol=1.0f-3,)
        end

        @testset "Logit BinaryCrossEntropyLoss" begin
            logitbceloss = BinaryCrossEntropyLoss(; logits=Val(true))
            logitbceloss_smooth = BinaryCrossEntropyLoss(;
                logits=Val(true), label_smoothing=2sf, epsilon=0
            )

            @test logitbceloss_smooth(logŷ, y) ≈
                -mean(
                yls .* log.(sigmoid(logŷ)) .+ (1 .- yls) .* log.(1 .- sigmoid(logŷ))
            )

            @test logitbceloss(logŷ, y) ≈
                mean(-y .* log.(sigmoid(logŷ)) .- (1 .- y) .* log.(1 .- sigmoid(logŷ)))

            @jet logitbceloss(logŷ, y)
            @jet logitbceloss_smooth(logŷ, y)

            @test_gradients(Base.Fix2(logitbceloss, y), logŷ; atol=1.0f-3, rtol=1.0f-3,)
        end

        @testset "BinaryFocalLoss" begin
            y = aType([
                0 1 0
                1 0 1
            ])
            ŷ = aType([
                0.268941 0.5 0.268941
                0.731059 0.5 0.731059
            ])

            y1 = aType([
                1 0
                0 1
            ])
            ŷ1 = aType([
                0.6 0.3
                0.4 0.7
            ])

            @test BinaryFocalLoss()(ŷ, y) ≈ 0.0728675615927385
            @test BinaryFocalLoss()(ŷ1, y1) ≈ 0.05691642237852222
            @test BinaryFocalLoss(; gamma=0)(ŷ, y) ≈ Lux.CrossEntropyLoss()(ŷ, y)

            @jet BinaryFocalLoss()(ŷ, y)

            @test_gradients(Base.Fix2(BinaryFocalLoss(), y), ŷ; atol=1.0f-3, rtol=1.0f-3)
        end

        @testset "FocalLoss" begin
            y = aType([
                1 0 0 0 1
                0 1 0 1 0
                0 0 1 0 0
            ])
            ŷ = aType(softmax(reshape(-7:7, 3, 5) .* 1.0f0))
            y1 = aType([
                1 0
                0 0
                0 1
            ])
            ŷ1 = aType([
                0.4 0.2
                0.5 0.5
                0.1 0.3
            ])

            @test FocalLoss()(ŷ, y) ≈ 1.1277571935622628
            @test FocalLoss()(ŷ1, y1) ≈ 0.45990566879720157
            @test FocalLoss(; gamma=0)(ŷ, y) ≈ CrossEntropyLoss()(ŷ, y)

            @jet FocalLoss()(ŷ, y)

            __f = Base.Fix2(FocalLoss(), y)
            # FD will lead to out of domain errors
            skip_backends = VERSION ≥ v"1.11-" ? Any[AutoEnzyme()] : []
            push!(skip_backends, AutoFiniteDiff())
            @test_gradients(__f, ŷ; atol=1.0f-3, rtol=1.0f-3, skip_backends)
        end
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
end
