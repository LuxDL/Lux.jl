@testitem "Compiled Loss Functions" tags=[:reactant] setup=[SharedTestSetup] begin
    using Reactant, Enzyme, Lux, OneHotArrays

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

            fn1(x) = LuxOps.xlogx.(x)
            fn2(x, y) = LuxOps.xlogy.(x, y)

            fn1_compiled = @compile fn1(x_ra)
            @test fn1(x) ≈ fn1_compiled(x_ra)

            fn2_compiled = @compile fn2(x_ra, y_ra)
            @test fn2(x, y) ≈ fn2_compiled(x_ra, y_ra)
        end

        @testset "Regression Loss" begin
            y = [1.0, 1.0, 0.0, 0.0]
            ŷ = [0.9, 0.1, 0.1, 0.9]

            y_ra = Reactant.to_rarray(y)
            ŷ_ra = Reactant.to_rarray(ŷ)

            @testset for loss in ("MSE", "MAE", "Huber")
                loss_mean = eval(Symbol(loss * "Loss"))()
                loss_sum = eval(Symbol(loss * "Loss"))(; agg=sum)
                loss_sum2 = eval(Symbol(loss * "Loss"))(; agg=(args...) -> sum(args...))

                loss_mean_compiled = @compile loss_mean(ŷ_ra, y_ra)
                @test loss_mean(ŷ, y) ≈ loss_mean_compiled(ŷ_ra, y_ra)

                loss_sum_compiled = @compile loss_sum(ŷ_ra, y_ra)
                @test loss_sum(ŷ, y) ≈ loss_sum_compiled(ŷ_ra, y_ra)

                loss_sum2_compiled = @compile loss_sum2(ŷ_ra, y_ra)
                @test loss_sum2(ŷ, y) ≈ loss_sum2_compiled(ŷ_ra, y_ra)
            end

            @testset "MSLE" begin
                y = [123.0, 456.0, 789.0]
                ŷ = [345.0, 332.0, 789.0]

                y_ra = Reactant.to_rarray(y)
                ŷ_ra = Reactant.to_rarray(ŷ)

                loss_msle = MSLELoss()
                loss_msle_compiled = @compile loss_msle(ŷ_ra, y_ra)
                @test loss_msle(ŷ, y) ≈ loss_msle_compiled(ŷ_ra, y_ra)
            end
        end

        @testset "Classification Loss" begin
            y = onehotbatch([1, 1, 0, 0], 0:1) |> Array
            ŷ = [0.1 0.9; 0.9 0.1; 0.9 0.1; 0.1 0.9]' |> Array

            y_ra = Reactant.to_rarray(y)
            ŷ_ra = Reactant.to_rarray(ŷ)

            @testset "CrossEntropyLoss" begin
                celoss = CrossEntropyLoss()
                celoss_compiled = @compile celoss(ŷ_ra, y_ra)
                @test celoss(ŷ, y) ≈ celoss_compiled(ŷ_ra, y_ra)

                celoss_ls = CrossEntropyLoss(; label_smoothing=0.1)
                celoss_ls_compiled = @compile celoss_ls(ŷ_ra, y_ra)
                @test celoss_ls(ŷ, y) ≈ celoss_ls_compiled(ŷ_ra, y_ra)

                celoss_lp = CrossEntropyLoss(; logits=Val(true))
                celoss_lp_compiled = @compile celoss_lp(log.(ŷ_ra), y_ra)
                @test celoss_lp(log.(ŷ), y) ≈ celoss_lp_compiled(log.(ŷ_ra), y_ra)

                celoss_lp_ls = CrossEntropyLoss(; logits=Val(true), label_smoothing=0.1)
                celoss_lp_ls_compiled = @compile celoss_lp_ls(log.(ŷ_ra), y_ra)
                @test celoss_lp_ls(log.(ŷ), y) ≈ celoss_lp_ls_compiled(log.(ŷ_ra), y_ra)
            end

            @testset "Binary CrossEntropyLoss" begin
                bceloss = BinaryCrossEntropyLoss()
                bceloss_compiled = @compile bceloss(ŷ_ra, y_ra)
                @test bceloss(ŷ, y) ≈ bceloss_compiled(ŷ_ra, y_ra)

                bceloss_ls = BinaryCrossEntropyLoss(; label_smoothing=0.1)
                bceloss_ls_compiled = @compile bceloss_ls(ŷ_ra, y_ra)
                @test bceloss_ls(ŷ, y) ≈ bceloss_ls_compiled(ŷ_ra, y_ra)

                bceloss_lp = BinaryCrossEntropyLoss(; logits=Val(true))
                bceloss_lp_compiled = @compile bceloss_lp(log.(ŷ_ra), y_ra)
                @test bceloss_lp(log.(ŷ), y) ≈ bceloss_lp_compiled(log.(ŷ_ra), y_ra)

                bceloss_lp_ls = BinaryCrossEntropyLoss(;
                    logits=Val(true), label_smoothing=0.1)
                bceloss_lp_ls_compiled = @compile bceloss_lp_ls(log.(ŷ_ra), y_ra)
                @test bceloss_lp_ls(log.(ŷ), y) ≈ bceloss_lp_ls_compiled(log.(ŷ_ra), y_ra)
            end

            @testset "BinaryFocalLoss" begin
                y = [0 1 0
                     1 0 1]
                ŷ = [0.268941 0.5 0.268941
                     0.731059 0.5 0.731059]

                y_ra = Reactant.to_rarray(y)
                ŷ_ra = Reactant.to_rarray(ŷ)

                bfl = BinaryFocalLoss()
                bfl_compiled = @compile bfl(ŷ_ra, y_ra)
                @test bfl(ŷ, y) ≈ bfl_compiled(ŷ_ra, y_ra)
            end

            @testset "FocalLoss" begin
                y = [1 0 0 0 1
                     0 1 0 1 0
                     0 0 1 0 0]
                ŷ = softmax(reshape(-7:7, 3, 5) .* 1.0f0) |> Array

                y_ra = Reactant.to_rarray(y)
                ŷ_ra = Reactant.to_rarray(ŷ)

                fl = FocalLoss()
                fl_compiled = @compile fl(ŷ_ra, y_ra)
                @test fl(ŷ, y) ≈ fl_compiled(ŷ_ra, y_ra)
            end
        end

        @testset "Other Losses" begin end
    end
end
