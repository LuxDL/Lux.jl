using Test, Random, Statistics
import Flux: relu, pullback, sigmoid, gradient
import ExplicitFluxLayers:
    apply,
    setup,
    parameterlength,
    statelength,
    trainmode,
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

@testset "ExplicitFluxLayers" begin
    @testset "Layers" begin
        @testset "Basic" begin
            include("layers/basic.jl")
        end
        @testset "Normalization" begin
            include("layers/normalize.jl")
        end
    end
end
