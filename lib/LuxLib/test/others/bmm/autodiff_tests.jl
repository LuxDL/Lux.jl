using LuxLib, Test
using LuxLib: batched_matmul, batched_vec
using NNlib: batched_adjoint, batched_transpose
using LuxTestUtils: @test_gradients, AutoEnzyme

include("bmm_testsetup.jl")

@testset "BMM AutoDiff" begin
    rng = StableRNG(1234)

    fn(A, B) = sum(batched_matmul(A, B))
    fn_vec(A, B) = sum(batched_vec(A, B))

    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        M, P, Q = 13, 7, 11
        B = 3

        @testset "Two 3-arrays" begin
            @test_gradients(
                fn,
                aType(randn(rng, Float32, M, P, B)),
                aType(randn(rng, Float32, P, Q, B));
                atol=1.0e-3,
                rtol=1.0e-3,
                skip_backends=[AutoEnzyme()]
            )
            @test_gradients(
                fn,
                batched_adjoint(aType(randn(rng, Float32, P, M, B))),
                aType(randn(rng, Float32, P, Q, B));
                atol=1.0e-3,
                rtol=1.0e-3,
                skip_backends=[AutoEnzyme()]
            )
            @test_gradients(
                fn,
                aType(randn(rng, Float32, M, P, B)),
                batched_transpose(aType(randn(rng, Float32, Q, P, B)));
                atol=1.0e-3,
                rtol=1.0e-3,
                skip_backends=[AutoEnzyme()]
            )
        end

        @testset "One a matrix..." begin
            @test_gradients(
                fn,
                aType(randn(rng, Float32, M, P)),
                aType(randn(rng, Float32, P, Q, B));
                atol=1.0e-3,
                rtol=1.0e-3,
                skip_backends=[AutoEnzyme()]
            )
            @test_gradients(
                fn,
                adjoint(aType(randn(rng, Float32, P, M))),
                aType(randn(rng, Float32, P, Q, B));
                atol=1.0e-3,
                rtol=1.0e-3,
                skip_backends=[AutoEnzyme()]
            )
            @test_gradients(
                fn,
                aType(randn(rng, Float32, M, P)),
                batched_adjoint(aType(randn(rng, Float32, Q, P, B)));
                atol=1.0e-3,
                rtol=1.0e-3,
                skip_backends=[AutoEnzyme()]
            )

            @test_gradients(
                fn,
                aType(randn(rng, Float32, M, P)),
                aType(randn(rng, Float32, P, Q, B));
                atol=1.0e-3,
                rtol=1.0e-3,
                skip_backends=[AutoEnzyme()]
            )
            @test_gradients(
                fn,
                adjoint(aType(randn(rng, Float32, P, M))),
                aType(randn(rng, Float32, P, Q, B));
                atol=1.0e-3,
                rtol=1.0e-3,
                skip_backends=[AutoEnzyme()]
            )
            @test_gradients(
                fn,
                aType(randn(rng, Float32, M, P)),
                batched_adjoint(aType(randn(rng, Float32, Q, P, B)));
                atol=1.0e-3,
                rtol=1.0e-3,
                skip_backends=[AutoEnzyme()]
            )
        end

        @testset "... or equivalent to a matrix" begin
            @test_gradients(
                fn,
                aType(randn(rng, Float32, M, P, 1)),
                aType(randn(rng, Float32, P, Q, B));
                atol=1.0e-3,
                rtol=1.0e-3,
                skip_backends=[AutoEnzyme()]
            )
            @test_gradients(
                fn,
                batched_transpose(aType(randn(rng, Float32, P, M, 1))),
                aType(randn(rng, Float32, P, Q, B));
                atol=1.0e-3,
                rtol=1.0e-3,
                skip_backends=[AutoEnzyme()]
            )
            @test_gradients(
                fn,
                aType(randn(rng, Float32, M, P, 1)),
                batched_transpose(aType(randn(rng, Float32, Q, P, B)));
                atol=1.0e-3,
                rtol=1.0e-3,
                skip_backends=[AutoEnzyme()]
            )
        end
    end

    @testset "EnzymeRules custom rules" begin
        EC = LuxLib.Impl.EnzymeCore
        ER = LuxLib.Impl.EnzymeRules
        cfg_B_overwritten = ER.RevConfig{
            true,false,1,(false, false, false, true),false,false
        }()

        A = randn(Float32, 2, 2, 2)
        B = randn(Float32, 2, 2, 2)
        C = zeros(Float32, 2, 2, 2)

        dC = ones(Float32, 2, 2, 2)
        dA = zeros(Float32, 2, 2, 2)
        dB = zeros(Float32, 2, 2, 2)

        C_anno = EC.Duplicated(C, dC)
        A_anno = EC.Duplicated(A, dA)
        B_anno = EC.Duplicated(B, dB)

        for func in (NNlib.batched_mul!, LuxLib.Impl.batched_matmul_loopvec_impl!)
            fill!(dC, 1.0f0)
            fill!(dA, 0.0f0)
            fill!(dB, 0.0f0)
            B_orig = copy(B)

            ret = ER.augmented_primal(
                cfg_B_overwritten, EC.Const(func), Nothing, C_anno, A_anno, B_anno
            )
            cache_A, cache_B = ret.tape

            @test cache_A === nothing
            @test cache_B ≈ B_orig

            # Overwrite B before reverse
            B .= 999.0f0

            ER.reverse(
                cfg_B_overwritten, EC.Const(func), Nothing, ret.tape, C_anno, A_anno, B_anno
            )

            expected_dA = zeros(Float32, 2, 2, 2)
            NNlib.batched_mul!(
                expected_dA, ones(Float32, 2, 2, 2), NNlib.batched_adjoint(B_orig)
            )
            @test dA ≈ expected_dA

            # Reset B
            B .= B_orig
        end
    end
end
