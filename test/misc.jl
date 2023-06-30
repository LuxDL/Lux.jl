using Lux, Zygote

include("test_utils.jl")

rng = get_stable_rng()

@testset "$mode: Simple Zygote Second Order Derivative" for (mode, aType, dev, ongpu) in MODES
    x = rand(rng, Float32, 1, 128) |> aType
    nn = Lux.Dense(1 => 1)
    ps, st = Lux.setup(rng, nn) |> dev

    function test_f(x, ps)
        mz, back = Zygote.pullback(x -> first(nn(x, ps, st)), x)
        ep = only(back(one.(mz)))
        return sum(mz) + sum(ep)
    end

    @test_nowarn test_f(x, ps)

    @test begin
        y, back = Zygote.pullback(test_f, x, ps)
        ∂x, ∂ps = back(one(y))
        ∂x !== nothing && ∂ps !== nothing
    end

    # Weird Zygote Quirks
    @test_broken begin
        ∂x, ∂ps = Zygote.jacobian(test_f, x, ps)
        ∂x !== nothing && ∂ps !== nothing
    end
end
