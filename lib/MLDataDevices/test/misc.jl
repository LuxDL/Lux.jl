using LuxDeviceUtils, ComponentArrays, Random
using ArrayInterface: parameterless_type

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

using ReverseDiff, Tracker, ForwardDiff

@testset "AD Types" begin
    x = randn(Float32, 10)

    x_rdiff = ReverseDiff.track(x)
    @test get_device(x_rdiff) isa LuxCPUDevice
    x_rdiff = ReverseDiff.track.(x)
    @test get_device(x_rdiff) isa LuxCPUDevice

    gdev = gpu_device()

    x_tracker = Tracker.param(x)
    @test get_device(x_tracker) isa LuxCPUDevice
    x_tracker = Tracker.param.(x)
    @test get_device(x_tracker) isa LuxCPUDevice
    x_tracker_dev = Tracker.param(x) |> gdev
    @test get_device(x_tracker_dev) isa parameterless_type(typeof(gdev))
    x_tracker_dev = Tracker.param.(x) |> gdev
    @test get_device(x_tracker_dev) isa parameterless_type(typeof(gdev))

    x_fdiff = ForwardDiff.Dual.(x)
    @test get_device(x_fdiff) isa LuxCPUDevice
    x_fdiff_dev = ForwardDiff.Dual.(x) |> gdev
    @test get_device(x_fdiff_dev) isa parameterless_type(typeof(gdev))
end
