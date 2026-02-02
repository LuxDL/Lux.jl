include("bmm_testsetup.jl")

using LuxTestUtils: ZYGOTE_TESTING_ENABLED

if ZYGOTE_TESTING_ENABLED[]
    using Tracker, Zygote, NNlib, LuxLib
    using LuxLib: batched_matmul
    using NNlib: batched_adjoint, batched_transpose

    @testset "BMM Tracker AoS" begin
        rng = StableRNG(1234)

        fn(A, B) = sum(batched_matmul(A, B))

        ops = (identity, NNlib.batched_adjoint, NNlib.batched_transpose)

        @testset "$mode" for (mode, aType, ongpu) in MODES
            x = aType(randn(rng, Float32, 3, 3, 2))

            @testset "$(op1) x $(op2)" for (op1, op2) in Iterators.product(ops, ops)
                x1 = op1(x)
                x2 = op2(x)

                ∂x1_tr, ∂x2_tr = Tracker.gradient(fn, x1, x2)
                ∂x1_zy, ∂x2_zy = Zygote.gradient(fn, x1, x2)

                @test ∂x1_tr ≈ ∂x1_zy atol = 1.0e-3 rtol = 1.0e-3
                @test ∂x2_tr ≈ ∂x2_zy atol = 1.0e-3 rtol = 1.0e-3

                @test ∂x1_tr isa Tracker.TrackedArray
                @test ∂x2_tr isa Tracker.TrackedArray
            end
        end
    end
else
    @info "Skipping BMM Tracker tests because Zygote testing is disabled"
end
