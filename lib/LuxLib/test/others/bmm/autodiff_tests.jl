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
end
