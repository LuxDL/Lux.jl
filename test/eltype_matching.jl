# This file intentionally doesn't have a `_tests.jl` suffix to avoid being used by
# ReTestItems. Since it relies on setting a preference, it needs to be run in a separate
# process.
using Lux, ForwardDiff, ReverseDiff, Tracker, Zygote, StableRNGs, Test

@info "Running eltype matching tests: \"$(Lux.LuxPreferences.ELTYPE_MISMATCH_HANDLING)\"..."

include("setup_modes.jl")

@testset "BACKEND GROUP: $(mode)" for (mode, aType, dev, ongpu) in MODES
    rng = StableRNG(123)

    model = Chain(Dense(2 => 3, tanh), Dense(3 => 2))
    x = aType(rand(2, 3))
    ps, st = dev(Lux.setup(rng, model))

    x_ad_arrs = Any[ForwardDiff.Dual.(x), Tracker.param(x)]
    if !ongpu
        push!(x_ad_arrs, Tracker.param.(x))
        push!(x_ad_arrs, ReverseDiff.track(x))
        push!(x_ad_arrs, ReverseDiff.track.(x))
    end

    # We only log once so can't really check the warning
    if Lux.LuxPreferences.ELTYPE_MISMATCH_HANDLING == "none" ||
        Lux.LuxPreferences.ELTYPE_MISMATCH_HANDLING == "warn"
        y, st_ = model(x, ps, st)
        @test eltype(y) == Float64  # We don't change the eltype

        @testset "Operator Overloading AD: $(typeof(x_arr))" for x_arr in x_ad_arrs
            @test Lux.Utils.eltype(first(model(x_arr, ps, st))) == Float64
        end

        dx, dps, _ = Zygote.gradient(sum ∘ first ∘ model, x, ps, st)
        @test Lux.Utils.eltype(dx) == Float64
        @test Lux.recursive_eltype(dps, Val(true)) == Float32
    end

    if Lux.LuxPreferences.ELTYPE_MISMATCH_HANDLING == "error"
        @test_throws Lux.EltypeMismatchException model(x, ps, st)

        @testset "Operator Overloading AD: $(typeof(x_arr))" for x_arr in x_ad_arrs
            @test_throws Lux.EltypeMismatchException model(x_arr, ps, st)
        end
    end

    if Lux.LuxPreferences.ELTYPE_MISMATCH_HANDLING == "convert"
        y, st_ = model(x, ps, st)
        @test eltype(y) == Float32

        @testset "Operator Overloading AD: $(typeof(x_arr))" for x_arr in x_ad_arrs
            y = first(model(x_arr, ps, st))

            if eltype(x_arr) <: ReverseDiff.TrackedReal
                @test Lux.Utils.eltype(y) == Float64
            else
                @test Lux.Utils.eltype(y) == Float32
            end
        end

        dx, dps, _ = Zygote.gradient(sum ∘ first ∘ model, x, ps, st)
        @test Lux.Utils.eltype(dx) == Float64
        @test Lux.recursive_eltype(dps, Val(true)) == Float32
    end
end
