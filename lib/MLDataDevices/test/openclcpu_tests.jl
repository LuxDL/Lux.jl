using MLDataDevices, Random, Test
using ArrayInterface: parameterless_type

@testset "CPU Fallback" begin
    @test !MLDataDevices.functional(OpenCLDevice)
    @test cpu_device() isa CPUDevice
    @test gpu_device() isa CPUDevice
    @test_throws MLDataDevices.Internal.DeviceSelectionException gpu_device(; force=true)
    @test_throws Exception default_device_rng(OpenCLDevice())
end
