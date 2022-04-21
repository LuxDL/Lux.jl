using Test, Random, Statistics
import Flux: relu, pullback, sigmoid
import ExplicitFluxLayers:
    apply,
    setup,
    parameterlength,
    statelength,
    trainmode,
    Dense,
    BatchNorm,
    SkipConnection,
    Parallel,
    Chain,
    WrappedFunction

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
