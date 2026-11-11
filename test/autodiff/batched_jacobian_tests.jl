using ComponentArrays, ForwardDiff, Zygote, Tracker, ReverseDiff

include("../shared_testsetup.jl")

@testset "Batched Jacobian" begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        models = (
            Chain(
                Conv((3, 3), 2 => 4, gelu; pad=SamePad()),
                Conv((3, 3), 4 => 2, gelu; pad=SamePad()),
                FlattenLayer(),
                Dense(18 => 2),
            ),
            Chain(Dense(2, 4, gelu), Dense(4, 2)),
        )
        Xs = (aType(randn(rng, Float32, 3, 3, 2, 4)), aType(randn(rng, Float32, 2, 4)))

        for (model, X) in zip(models, Xs)
            ps, st = dev(Lux.setup(rng, model))
            smodel = StatefulLuxLayer(model, ps, st)

            J1 = if ongpu
                allow_unstable() do
                    only(Zygote.jacobian(smodel, X))
                end
            else
                allow_unstable() do
                    ForwardDiff.jacobian(smodel, X)
                end
            end

            @testset "$(backend)" for backend in (AutoZygote(), AutoForwardDiff())
                J2 = allow_unstable() do
                    batched_jacobian(smodel, backend, X)
                end
                J2_mat =
                    mapreduce(
                        Base.Fix1(Lux.AutoDiffInternalImpl.batched_row, J2),
                        hcat,
                        1:(size(J2, 1) * size(J2, 3)),
                    )'

                @test Array(J1) ≈ Array(J2_mat) atol = 1.0e-3 rtol = 1.0e-3

                ps = dev(ComponentArray(cpu_device()(ps)))

                smodel = StatefulLuxLayer(model, ps, st)

                J3 = allow_unstable() do
                    batched_jacobian(smodel, backend, X)
                end

                @test Array(J2) ≈ Array(J3) atol = 1.0e-3 rtol = 1.0e-3
            end
        end

        @testset "Issue #636 Chunksize Specialization" begin
            for N in (2, 4, 8, 11, 12, 50, 51), backend in (AutoZygote(), AutoForwardDiff())
                model = @compact(; potential=Dense(N => N, gelu), backend=backend) do x
                    @return allow_unstable() do
                        batched_jacobian(potential, backend, x)
                    end
                end

                ps, st = dev(Lux.setup(Random.default_rng(), model))

                x = dev(randn(Float32, N, 3))
                @test first(model(x, ps, st)) isa AbstractArray{<:Any,3}
            end
        end

        @testset "Simple Batched Jacobian" begin
            # Without any Lux bs just plain old batched jacobian
            ftest(x) = x .^ 2
            x = dev(reshape(Float32.(1:6), 2, 3))

            Jx_true = zeros(Float32, 2, 2, 3)
            Jx_true[1, 1, 1] = 2
            Jx_true[2, 2, 1] = 4
            Jx_true[1, 1, 2] = 6
            Jx_true[2, 2, 2] = 8
            Jx_true[1, 1, 3] = 10
            Jx_true[2, 2, 3] = 12
            Jx_true = dev(Jx_true)

            Jx_fdiff = allow_unstable() do
                batched_jacobian(ftest, AutoForwardDiff(), x)
            end
            @test Jx_fdiff ≈ Jx_true

            Jx_zygote = allow_unstable() do
                batched_jacobian(ftest, AutoZygote(), x)
            end
            @test Jx_zygote ≈ Jx_true

            fincorrect(x) = x[:, 1]
            x = dev(reshape(Float32.(1:6), 2, 3))

            @test_throws AssertionError batched_jacobian(fincorrect, AutoForwardDiff(), x)
            @test_throws AssertionError batched_jacobian(fincorrect, AutoZygote(), x)
        end

        @testset "batched_row unit tests" begin
            for (M, N) in ((2, 3), (3, 2))
                # 2D array
                x2d = dev(randn(Float32, M, N))
                for idx in 1:M
                    @test Array(Lux.AutoDiffInternalImpl.batched_row(x2d, idx)) ==
                        Array(x2d[idx, :])
                end

                # 3D array for K = 1 and K = 3
                for K in (1, 3)
                    x3d = dev(randn(Float32, M, N, K))
                    block_diag = cat([Array(x3d[:, :, k]) for k in 1:K]...; dims=(1, 2))
                    for idx in 1:(M * K)
                        @test Array(Lux.AutoDiffInternalImpl.batched_row(x3d, idx)) ≈
                            block_diag[idx, :]
                    end
                end
            end
        end
    end
end
