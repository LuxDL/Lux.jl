using LuxDeviceUtils, Random

@testset "CPU Fallback" begin
    @test !LuxDeviceUtils.functional(LuxoneAPIDevice)
    @test cpu_device() isa LuxCPUDevice
    @test gpu_device() isa LuxCPUDevice
    @test_throws LuxDeviceUtils.LuxDeviceSelectionException gpu_device(;
        force_gpu_usage=true)
end

using oneAPI

@testset "Loaded Trigger Package" begin
    @test LuxDeviceUtils.GPU_DEVICE[] === nothing

    if LuxDeviceUtils.functional(LuxoneAPIDevice)
        @info "oneAPI is functional"
        @test gpu_device() isa LuxoneAPIDevice
        @test gpu_device(; force_gpu_usage=true) isa LuxoneAPIDevice
    else
        @info "oneAPI is NOT functional"
        @test gpu_device() isa LuxoneAPIDevice
        @test_throws LuxDeviceUtils.LuxDeviceSelectionException gpu_device(;
            force_gpu_usage=true)
    end
    @test LuxDeviceUtils.GPU_DEVICE[] !== nothing
end

using FillArrays, Zygote  # Extensions

@testset "Data Transfer" begin
    ps = (a=(c=zeros(10, 1), d=1), b=ones(10, 1), e=:c, d="string",
        mixed=[2.0f0, 3.0, ones(2, 3)],  # mixed array types
        rng_default=Random.default_rng(), rng=MersenneTwister(),
        one_elem=Zygote.OneElement(2.0f0, (2, 3), (1:3, 1:4)), farray=Fill(1.0f0, (2, 3)))

    device = gpu_device()
    aType = LuxDeviceUtils.functional(LuxoneAPIDevice) ? oneArray : Array
    rngType = LuxDeviceUtils.functional(LuxoneAPIDevice) ? oneAPI.GPUArrays.RNG :
              Random.AbstractRNG

    ps_xpu = ps |> device
    @test get_device(ps_xpu) isa LuxoneAPIDevice
    @test ps_xpu.a.c isa aType
    @test ps_xpu.b isa aType
    @test ps_xpu.a.d == ps.a.d
    @test ps_xpu.mixed isa Vector
    @test ps_xpu.mixed[1] isa Float32
    @test ps_xpu.mixed[2] isa Float64
    @test ps_xpu.mixed[3] isa aType
    @test ps_xpu.e == ps.e
    @test ps_xpu.d == ps.d
    @test ps_xpu.rng_default isa rngType
    @test ps_xpu.rng == ps.rng

    if LuxDeviceUtils.functional(LuxoneAPIDevice)
        @test ps_xpu.one_elem isa oneArray
        @test ps_xpu.farray isa oneArray
    else
        @test ps_xpu.one_elem isa Zygote.OneElement
        @test ps_xpu.farray isa Fill
    end

    ps_cpu = ps_xpu |> cpu_device()
    @test get_device(ps_cpu) isa LuxCPUDevice
    @test ps_cpu.a.c isa Array
    @test ps_cpu.b isa Array
    @test ps_cpu.a.c == ps.a.c
    @test ps_cpu.b == ps.b
    @test ps_cpu.a.d == ps.a.d
    @test ps_cpu.mixed isa Vector
    @test ps_cpu.mixed[1] isa Float32
    @test ps_cpu.mixed[2] isa Float64
    @test ps_cpu.mixed[3] isa Array
    @test ps_cpu.e == ps.e
    @test ps_cpu.d == ps.d
    @test ps_cpu.rng_default isa Random.TaskLocalRNG
    @test ps_cpu.rng == ps.rng

    if LuxDeviceUtils.functional(LuxoneAPIDevice)
        @test ps_cpu.one_elem isa Array
        @test ps_cpu.farray isa Array
    else
        @test ps_cpu.one_elem isa Zygote.OneElement
        @test ps_cpu.farray isa Fill
    end

    ps_mixed = (; a=rand(2), b=device(rand(2)))
    @test_throws ArgumentError get_device(ps_mixed)
end
