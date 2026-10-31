include("../shared_testsetup.jl")
include("../reactant_testsetup.jl")

using Reactant, Lux, Random

# Two layers of one type with different settings, applied to same-shaped inputs in one compiled
# function, must each get their own `@trace`'d call.
@testset "Reactant: @trace call cache keys on layer settings" begin
    @testset "$(mode)" for (mode, atype, dev, ongpu) in MODES
        dev = reactant_device(; force=true)
        Reactant.set_default_backend(ongpu ? "gpu" : "cpu")

        rng = StableRNG(12345)
        # Channels with different means and scales, so 8 groups and 1 group disagree.
        x =
            randn(rng, Float32, 8, 8, 16, 2) .* reshape(1:16, 1, 1, :, 1) .+
            reshape(1:16, 1, 1, :, 1)
        x_ra = dev(x)

        @testset "$(name)" for (name, l1, l2) in (
            ("GroupNorm groups", GroupNorm(16, 8), GroupNorm(16, 1)),
            (
                "Conv stride",
                Conv((3, 3), 16 => 4; pad=1),
                Conv((3, 3), 16 => 4; pad=1, stride=2),
            ),
        )
            ps, st = Lux.setup(rng, l1)
            ps_ra, st_ra = dev((ps, st))
            both(x, ps, st) = (first(l1(x, ps, st)), first(l2(x, ps, st)))
            y1_ra, y2_ra = @jit both(x_ra, ps_ra, st_ra)
            y1, y2 = first(l1(x, ps, st)), first(l2(x, ps, st))

            @test size(y2_ra) == size(y2)
            @test Array(y1_ra) ≈ y1 atol = 1.0e-4 rtol = 1.0e-2
            # Guarded, so a wrong shape fails here instead of throwing in `isapprox`.
            @test size(y2_ra) == size(y2) &&
                isapprox(Array(y2_ra), y2; atol=1.0e-4, rtol=1.0e-2)
        end
    end
end
