using LuxDeviceUtils, ComponentArrays, Random

@testset "https://github.com/LuxDL/LuxDeviceUtils.jl/issues/10 patch" begin
    dev = LuxCPUDevice()
    ps = (; weight=randn(10, 1), bias=randn(1))

    ps_ca = ps |> ComponentArray

    ps_ca_dev = ps_ca |> dev

    @test ps_ca_dev isa ComponentArray

    @test ps_ca_dev.weight == ps.weight
    @test ps_ca_dev.bias == ps.bias

    @test ps_ca_dev == (ps |> dev |> ComponentArray)
end
