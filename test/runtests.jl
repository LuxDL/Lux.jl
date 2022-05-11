using Test, Random, Statistics, Zygote, Metalhead, Lux, Functors
import Flux: relu, pullback, sigmoid, gradient
import Lux:
    apply,
    setup,
    parameterlength,
    statelength,
    initialparameters,
    initialstates,
    update_state,
    trainmode,
    testmode,
    transform,
    AbstractExplicitLayer,
    AbstractExplicitContainerLayer

function gradtest(model, input, ps, st)
    y, pb = Zygote.pullback(p -> model(input, p, st)[1], ps)
    gs = pb(ones(Float32, size(y)))
      
    # if we make it to here with no error, success!
    return true
end

function run_model(m::AbstractExplicitLayer, x, mode=:test)
    if mode == :test
        ps, st = setup(Random.default_rng(), m)
        st = testmode(st)
        return apply(m, x, ps, st)[1]
    end
end

function Base.isapprox(nt1::NamedTuple{fields}, nt2::NamedTuple{fields}) where {fields}
    checkapprox(xy) = xy[1] ≈ xy[2]
    checkapprox(t::Tuple{Nothing,Nothing}) = true
    all(checkapprox, zip(values(nt1), values(nt2)))
end


@testset "Lux" begin
    @testset "Layers" begin
        @testset "Basic" begin
            include("layers/basic.jl")
        end
        @testset "Normalization" begin
            include("layers/normalize.jl")
        end
    end

    # Might not want to run always
    @testset "Metalhead Models" begin
        @testset "ConvNets -- ImageNet" begin
            include("models/convnets.jl")
        end
    end
end
