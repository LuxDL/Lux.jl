@testitem "Compiled Loss Functions: Helpers" tags = [:reactant] setup = [SharedTestSetup] begin
    using Reactant, Lux, OneHotArrays

    rng = StableRNG(123)

    @testset "$(mode)" for (mode, atype, dev, ongpu) in MODES
        if mode == "amdgpu"
            @warn "Skipping AMDGPU tests for Reactant"
            continue
        end

        if ongpu
            Reactant.set_default_backend("gpu")
        else
            Reactant.set_default_backend("cpu")
        end

        @testset "xlogx & xlogy" begin
            x = rand(rng, 10)
            y = rand(rng, 10)
            x_ra = Reactant.to_rarray(x)
            y_ra = Reactant.to_rarray(y)

            @test LuxOps.xlogx.(x) ≈ @jit LuxOps.xlogx.(x_ra)
            @test LuxOps.xlogy.(x, y) ≈ @jit LuxOps.xlogy.(x_ra, y_ra)
        end
    end
end

@testitem "Compiled Loss Functions: Regression Loss" tags = [:reactant] setup = [
    SharedTestSetup
] begin
    using Reactant, Lux, OneHotArrays

    rng = StableRNG(123)

    @testset "$(mode)" for (mode, atype, dev, ongpu) in MODES
        if mode == "amdgpu"
            @warn "Skipping AMDGPU tests for Reactant"
            continue
        end

        if ongpu
            Reactant.set_default_backend("gpu")
        else
            Reactant.set_default_backend("cpu")
        end

        y = [1.0, 1.0, 0.0, 0.0]
        ŷ = [0.9, 0.1, 0.1, 0.9]

        y_ra = Reactant.to_rarray(y)
        ŷ_ra = Reactant.to_rarray(ŷ)

        @testset for loss in ("MSE", "MAE", "Huber")
            loss_mean = eval(Symbol(loss * "Loss"))()
            loss_sum = eval(Symbol(loss * "Loss"))(; agg=sum)
            loss_sum2 = eval(Symbol(loss * "Loss"))(; agg=(args...) -> sum(args...))

            @test loss_mean(ŷ, y) ≈ @jit(loss_mean(ŷ_ra, y_ra))
            @test loss_sum(ŷ, y) ≈ @jit(loss_sum(ŷ_ra, y_ra))
            @test loss_sum2(ŷ, y) ≈ @jit(loss_sum2(ŷ_ra, y_ra))
        end

        @testset "MSLE" begin
            y = [123.0, 456.0, 789.0]
            ŷ = [345.0, 332.0, 789.0]

            y_ra = Reactant.to_rarray(y)
            ŷ_ra = Reactant.to_rarray(ŷ)

            loss_msle = MSLELoss()
            @test loss_msle(ŷ, y) ≈ @jit(loss_msle(ŷ_ra, y_ra))
        end
    end
end

@testitem "Compiled Loss Functions: Classification Loss" tags = [:reactant] setup = [
    SharedTestSetup
] begin
    using Reactant, Lux, OneHotArrays

    rng = StableRNG(123)

    @testset "$(mode)" for (mode, atype, dev, ongpu) in MODES
        if mode == "amdgpu"
            @warn "Skipping AMDGPU tests for Reactant"
            continue
        end

        if ongpu
            Reactant.set_default_backend("gpu")
        else
            Reactant.set_default_backend("cpu")
        end

        y = Array(onehotbatch([1, 1, 0, 0], 0:1))
        ŷ = Array([0.1 0.9; 0.9 0.1; 0.9 0.1; 0.1 0.9]')

        y_ra = Reactant.to_rarray(y)
        ŷ_ra = Reactant.to_rarray(ŷ)

        @testset "CrossEntropyLoss" begin
            celoss = CrossEntropyLoss()
            @test celoss(ŷ, y) ≈ @jit(celoss(ŷ_ra, y_ra))

            celoss_ls = CrossEntropyLoss(; label_smoothing=0.1)
            @test celoss_ls(ŷ, y) ≈ @jit(celoss_ls(ŷ_ra, y_ra))

            celoss_lp = CrossEntropyLoss(; logits=Val(true))
            logit_celoss_lp = (ŷ, y) -> celoss_lp(log.(ŷ), y)
            @test logit_celoss_lp(ŷ, y) ≈ @jit(logit_celoss_lp(ŷ_ra, y_ra))

            celoss_lp_ls = CrossEntropyLoss(; logits=Val(true), label_smoothing=0.1)
            logit_celoss_lp_ls = (ŷ, y) -> celoss_lp_ls(log.(ŷ), y)
            @test logit_celoss_lp_ls(ŷ, y) ≈ @jit(logit_celoss_lp_ls(ŷ_ra, y_ra))
        end

        @testset "Binary CrossEntropyLoss" begin
            bceloss = BinaryCrossEntropyLoss()
            @test bceloss(ŷ, y) ≈ @jit(bceloss(ŷ_ra, y_ra))

            bceloss_ls = BinaryCrossEntropyLoss(; label_smoothing=0.1)
            @test bceloss_ls(ŷ, y) ≈ @jit(bceloss_ls(ŷ_ra, y_ra))

            # XXX: reenable once https://github.com/EnzymeAD/Enzyme-JAX/pull/1401 lands
            # bceloss_lp = BinaryCrossEntropyLoss(; logits=Val(true))
            # logit_bceloss_lp = (ŷ, y) -> bceloss_lp(log.(ŷ), y)
            # @test logit_bceloss_lp(ŷ, y) ≈ @jit(logit_bceloss_lp(ŷ_ra, y_ra))

            # bceloss_lp_ls = BinaryCrossEntropyLoss(; logits=Val(true), label_smoothing=0.1)
            # logit_bceloss_lp_ls = (ŷ, y) -> bceloss_lp_ls(log.(ŷ), y)
            # @test logit_bceloss_lp_ls(ŷ, y) ≈ @jit(logit_bceloss_lp_ls(ŷ_ra, y_ra))
        end

        @testset "BinaryFocalLoss" begin
            y = [
                0 1 0
                1 0 1
            ]
            ŷ = [
                0.268941 0.5 0.268941
                0.731059 0.5 0.731059
            ]

            y_ra = Reactant.to_rarray(y)
            ŷ_ra = Reactant.to_rarray(ŷ)

            bfl = BinaryFocalLoss()
            @test bfl(ŷ, y) ≈ @jit(bfl(ŷ_ra, y_ra))
        end

        @testset "FocalLoss" begin
            y = [
                1 0 0 0 1
                0 1 0 1 0
                0 0 1 0 0
            ]
            ŷ = Array(softmax(reshape(-7:7, 3, 5) .* 1.0f0))

            y_ra = Reactant.to_rarray(y)
            ŷ_ra = Reactant.to_rarray(ŷ)

            fl = FocalLoss()
            @test fl(ŷ, y) ≈ @jit(fl(ŷ_ra, y_ra))
        end
    end
end

@testitem "Compiled Loss Functions: Other Losses" tags = [:reactant] setup = [
    SharedTestSetup
] begin
    using Reactant, Lux, OneHotArrays

    rng = StableRNG(123)

    @testset "$(mode)" for (mode, atype, dev, ongpu) in MODES
        if mode == "amdgpu"
            @warn "Skipping AMDGPU tests for Reactant"
            continue
        end

        if ongpu
            Reactant.set_default_backend("gpu")
        else
            Reactant.set_default_backend("cpu")
        end

        @testset "KLDivergenceLoss" begin
            y = [1.0 2.0 3.0]
            ŷ = [4.0 5.0 6.0]

            y_ra = Reactant.to_rarray(y)
            ŷ_ra = Reactant.to_rarray(ŷ)

            kldl = KLDivergenceLoss()
            @test kldl(ŷ, y) ≈ @jit(kldl(ŷ_ra, y_ra))
        end

        @testset "HingeLoss" begin
            y = [1.0, 2.0, 3.0, 4.0]
            ŷ = [5.0, 6.0, 7.0, 8.0]

            y_ra = Reactant.to_rarray(y)
            ŷ_ra = Reactant.to_rarray(ŷ)

            hl = HingeLoss()
            @test hl(ŷ, y) ≈ @jit(hl(ŷ_ra, y_ra))

            hl = HingeLoss(; agg=mean)
            @test hl(ŷ, y) ≈ @jit(hl(ŷ_ra, y_ra))
        end

        @testset "SquaredHingeLoss" begin
            y = [1.0, 2.0, 3.0, 4.0]
            ŷ = [5.0, 6.0, 7.0, 8.0]

            y_ra = Reactant.to_rarray(y)
            ŷ_ra = Reactant.to_rarray(ŷ)

            hl = SquaredHingeLoss()
            @test hl(ŷ, y) ≈ @jit(hl(ŷ_ra, y_ra))

            hl = SquaredHingeLoss(; agg=mean)
            @test hl(ŷ, y) ≈ @jit(hl(ŷ_ra, y_ra))
        end

        @testset "PoissonLoss" begin
            y = [0.1, 0.2, 0.3]
            ŷ = [0.4, 0.5, 0.6]

            y_ra = Reactant.to_rarray(y)
            ŷ_ra = Reactant.to_rarray(ŷ)

            pl = PoissonLoss()
            @test pl(ŷ, y) ≈ @jit(pl(ŷ_ra, y_ra))

            pl = PoissonLoss(; agg=mean)
            @test pl(ŷ, y) ≈ @jit(pl(ŷ_ra, y_ra))
        end

        @testset "DiceCoeffLoss" begin
            y = [1.0, 0.5, 0.3, 2.4]
            ŷ = [0.0, 1.4, 0.5, 1.2]

            y_ra = Reactant.to_rarray(y)
            ŷ_ra = Reactant.to_rarray(ŷ)

            dl = DiceCoeffLoss()
            @test dl(ŷ, y) ≈ @jit(dl(ŷ_ra, y_ra))

            dl = DiceCoeffLoss(; agg=mean)
            @test dl(ŷ, y) ≈ @jit(dl(ŷ_ra, y_ra))
        end

        @testset "Siamese Contrastive Loss" begin
            y = [
                1.0 0.0
                0.0 0.0
                0.0 1.0
            ]
            ŷ = [
                0.4 0.2
                0.5 0.5
                0.1 0.3
            ]

            y_ra = Reactant.to_rarray(y)
            ŷ_ra = Reactant.to_rarray(ŷ)

            sl = SiameseContrastiveLoss()
            @test sl(ŷ, y) ≈ @jit(sl(ŷ_ra, y_ra))

            sl = SiameseContrastiveLoss(; agg=mean)
            @test sl(ŷ, y) ≈ @jit(sl(ŷ_ra, y_ra))
        end
    end
end
