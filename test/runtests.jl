using Test, Random, Statistics
import Flux: relu, pullback, sigmoid
import ExplicitFluxLayers: apply, setup, parameterlength, statelength, Dense, BatchNorm, trainmode

@testset "ExplicitFluxLayers" begin
    @testset "Linear" begin
        include("linear.jl")
    end

    @testset "Normalization" begin
        include("normalization.jl")
    end
end
