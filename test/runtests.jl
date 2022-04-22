using Test, Random, Statistics, Zygote, Metalhead, ExplicitFluxLayers
import Flux: relu, pullback, sigmoid, gradient
import ExplicitFluxLayers:
    apply,
    setup,
    parameterlength,
    statelength,
    initialparameters,
    initialstates,
    trainmode,
    transform,
    AbstractExplicitLayer,
    AbstractExplicitContainerLayer,
    Dense,
    BatchNorm,
    SkipConnection,
    Parallel,
    Chain,
    WrappedFunction,
    NoOpLayer,
    Conv,
    MaxPool,
    MeanPool,
    GlobalMaxPool,
    GlobalMeanPool,
    Upsample

function gradtest(model, input, ps, st)
    y, pb = Zygote.pullback(p -> model(input, p, st)[1], ps)
    gs = pb(ones(Float32, size(y)))
      
    # if we make it to here with no error, success!
    return true
end

function run_model(m::AbstractExplicitLayer, x)
    ps, st = setup(Random.default_rng(), m)
    return apply(m, x, ps, st)[1]
end

function Base.isapprox(nt1::NamedTuple{fields}, nt2::NamedTuple{fields}) where {fields}
    all(isapprox, values(nt1), values(nt2))
end


@testset "ExplicitFluxLayers" begin
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
