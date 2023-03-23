using SafeTestsets, Test

@testset "LuxLib" begin
    @time @safetestset "Dropout" begin include("api/dropout.jl") end

    @time @safetestset "BatchNorm" begin include("api/batchnorm.jl") end
    @time @safetestset "GroupNorm" begin include("api/groupnorm.jl") end
    @time @safetestset "InstanceNorm" begin include("api/instancenorm.jl") end
    @time @safetestset "LayerNorm" begin include("api/layernorm.jl") end

    @time @safetestset "ForwardDiff Extension" begin include("ext/LuxLibForwardDiffExt.jl") end
end
