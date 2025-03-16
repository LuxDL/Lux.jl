@testitem "complex differentiation: issue #977" tags = [:misc] begin
    using Lux, Zygote, Random

    rng = Random.default_rng()
    Random.seed!(rng, 666)

    rbf(x) = exp.(-(x .^ 2))

    U = Lux.Chain(Lux.Dense(1, 10, rbf), Lux.Dense(10, 3, rbf))

    θ, st = Lux.setup(rng, U)

    function complex_step_differentiation(f::Function, x::Float64, ϵ::Float64)
        return imag(f(x + ϵ * im)) / ϵ
    end

    loss(t) = sum(complex_step_differentiation(τ -> U([τ], θ, st)[begin], t, 1.0e-5))

    if pkgversion(LuxLib) ≥ v"1.3.10"
        @test only(Zygote.gradient(loss, 1.0)) isa Float64
    else
        @test_broken only(Zygote.gradient(loss, 1.0)) isa Float64
    end
end
