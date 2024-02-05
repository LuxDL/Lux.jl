using SafeTestsets, Test, TestSetExtensions

@testset ExtendedTestSet "LuxLib" begin
    @safetestset "Dropout" include("api/dropout.jl")

    @testset "Normalization" begin
        @safetestset "BatchNorm" include("api/batchnorm.jl")
        @safetestset "GroupNorm" include("api/groupnorm.jl")
        @safetestset "InstanceNorm" include("api/instancenorm.jl")
        @safetestset "LayerNorm" include("api/layernorm.jl")
    end

    @safetestset "ForwardDiff Extension" include("ext/LuxLibForwardDiffExt.jl")

    @safetestset "Efficient Jacobian-Vector-Products" include("jvp.jl")

    @safetestset "Aqua Tests" include("aqua.jl")
end
