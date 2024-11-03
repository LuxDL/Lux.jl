@testitem "Utils.nfan" begin
    using WeightInitializers: Utils

    @test Utils.nfan() == (1, 1) # Fallback
    @test Utils.nfan(4) == (1, 4) # Vector
    @test Utils.nfan(4, 5) == (5, 4) # Matrix
    @test Utils.nfan((4, 5, 6)) == Utils.nfan(4, 5, 6) # Tuple
    @test Utils.nfan(4, 5, 6) == 4 .* (5, 6) # Convolution
end
