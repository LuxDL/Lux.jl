using ForwardDiff, Zygote, Enzyme, OneHotArrays
using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent

const CRC = ChainRulesCore

include("../../shared_testsetup.jl")

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
