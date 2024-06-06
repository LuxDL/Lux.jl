using LuxDeviceUtils, Random

@testset "CPU Fallback" begin
    @test cpu_device() isa LuxCPUDevice
    @test gpu_device() isa LuxCPUDevice
    @test_throws LuxDeviceUtils.LuxDeviceSelectionException gpu_device(;
        force_gpu_usage=true)
end

using AMDGPU

@testset "Loaded Trigger Package" begin
    @test LuxDeviceUtils.GPU_DEVICE[] === nothing

    if LuxDeviceUtils.functional(LuxAMDGPUDevice)
        @info "AMDGPU is functional"
        @test gpu_device() isa LuxAMDGPUDevice
        @test gpu_device(; force_gpu_usage=true) isa LuxAMDGPUDevice
    else
        @info "AMDGPU is NOT functional"
        @test gpu_device() isa LuxCPUDevice
        @test_throws LuxDeviceUtils.LuxDeviceSelectionException gpu_device(;
            force_gpu_usage=true)
    end
    @test LuxDeviceUtils.GPU_DEVICE[] !== nothing
end

using FillArrays, Zygote  # Extensions

@testset "Data Transfer" begin
    ps = (a=(c=zeros(10, 1), d=1), b=ones(10, 1), e=:c, d="string",
        rng_default=Random.default_rng(), rng=MersenneTwister(),
        one_elem=Zygote.OneElement(2.0f0, (2, 3), (1:3, 1:4)), farray=Fill(1.0f0, (2, 3)))

    device = gpu_device()
    aType = LuxDeviceUtils.functional(LuxAMDGPUDevice) ? ROCArray : Array
    rngType = LuxDeviceUtils.functional(LuxAMDGPUDevice) ? AMDGPU.rocRAND.RNG :
              Random.AbstractRNG

    ps_xpu = ps |> device
    @test ps_xpu.a.c isa aType
    @test ps_xpu.b isa aType
    @test ps_xpu.a.d == ps.a.d
    @test ps_xpu.e == ps.e
    @test ps_xpu.d == ps.d
    @test ps_xpu.rng_default isa rngType
    @test ps_xpu.rng == ps.rng

    if LuxDeviceUtils.functional(LuxAMDGPUDevice)
        @test ps_xpu.one_elem isa ROCArray
        @test ps_xpu.farray isa ROCArray
    else
        @test ps_xpu.one_elem isa Zygote.OneElement
        @test ps_xpu.farray isa Fill
    end

    ps_cpu = ps_xpu |> cpu_device()
    @test ps_cpu.a.c isa Array
    @test ps_cpu.b isa Array
    @test ps_cpu.a.c == ps.a.c
    @test ps_cpu.b == ps.b
    @test ps_cpu.a.d == ps.a.d
    @test ps_cpu.e == ps.e
    @test ps_cpu.d == ps.d
    @test ps_cpu.rng_default isa Random.TaskLocalRNG
    @test ps_cpu.rng == ps.rng

    if LuxDeviceUtils.functional(LuxAMDGPUDevice)
        @test ps_cpu.one_elem isa Array
        @test ps_cpu.farray isa Array
    else
        @test ps_cpu.one_elem isa Zygote.OneElement
        @test ps_cpu.farray isa Fill
    end
end

@testset "Multiple Devices CUDA" begin
    if LuxDeviceUtils.functional(LuxAMDGPUDevice)
        ps = (; weight=rand(Float32, 10), bias=rand(Float32, 10))
        ps_cpu = deepcopy(ps)
        cdev = cpu_device()
        for idx in 1:length(AMDGPU.devices())
            amdgpu_device = gpu_device(idx)
            @test typeof(amdgpu_device.device) <: AMDGPU.HIPDevice
            @test AMDGPU.device_id(amdgpu_device.device) == idx

            ps = ps |> amdgpu_device
            @test ps.weight isa ROCArray
            @test ps.bias isa ROCArray
            @test AMDGPU.device_id(AMDGPU.device(ps.weight)) == idx
            @test AMDGPU.device_id(AMDGPU.device(ps.bias)) == idx
            @test isequal(cdev(ps.weight), ps_cpu.weight)
            @test isequal(cdev(ps.bias), ps_cpu.bias)
        end

        ps = ps |> cdev
        @test ps.weight isa Array
        @test ps.bias isa Array
    end
end
