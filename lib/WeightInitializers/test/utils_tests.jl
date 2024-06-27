@testitem "_nfan" begin
    using WeightInitializers: _nfan

    @test _nfan() == (1, 1) # Fallback
    @test _nfan(4) == (1, 4) # Vector
    @test _nfan(4, 5) == (5, 4) # Matrix
    @test _nfan((4, 5, 6)) == _nfan(4, 5, 6) # Tuple
    @test _nfan(4, 5, 6) == 4 .* (5, 6) # Convolution
end
