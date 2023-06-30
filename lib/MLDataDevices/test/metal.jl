using LuxDeviceUtils, Random

@testset "CPU Fallback" begin
    @test cpu_device() isa LuxCPUDevice
    @test gpu_device() isa LuxCPUDevice
    @test_throws LuxDeviceUtils.LuxDeviceSelectionException gpu_device(;
        force_gpu_usage=true)
end

using Metal

@testset "Loaded Trigger Package" begin
    @test Lux.GPU_BACKEND[] === nothing

    if Metal.functional()
        @info "Metal is functional"
        @test gpu_device() isa LuxMetalDevice
        @test gpu_device(; force_gpu_usage=true) isa LuxMetalDevice
    else
        @info "Metal is NOT functional"
        @test gpu_device() isa LuxMetalDevice
        @test_throws LuxDeviceUtils.LuxDeviceSelectionException gpu_device(;
            force_gpu_usage=true)
    end
    @test Lux.GPU_BACKEND[] !== nothing
end

using FillArrays, Zygote  # Extensions

@testset "Data Transfer" begin
    ps = (a=(c=zeros(10, 1), d=1),
        b=ones(10, 1),
        e=:c,
        d="string",
        rng=Random.default_rng(),
        one_elem=Zygote.OneElement(2.0f0, (2, 3), (1:3, 1:4)),
        farray=Fill(1.0f0, (2, 3)))

    device = gpu_device()
    aType = Metal.functional() ? MtlArray : Array

    ps_xpu = ps |> device
    @test ps_xpu.a.c isa aType
    @test ps_xpu.b isa aType
    @test ps_xpu.a.d == ps.a.d
    @test ps_xpu.e == ps.e
    @test ps_xpu.d == ps.d
    @test ps_xpu.rng == ps.rng

    if Metal.functional()
        @test ps_xpu.one_elem isa MtlArray
        @test ps_xpu.farray isa MtlArray
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
    @test ps_cpu.rng == ps.rng

    if Metal.functional()
        @test ps_cpu.one_elem isa Array
        @test ps_cpu.farray isa Array
    else
        @test ps_cpu.one_elem isa Zygote.OneElement
        @test ps_cpu.farray isa Fill
    end
end
