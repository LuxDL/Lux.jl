using LuxDeviceUtils, Random

@testset "CPU Fallback" begin
    @test !LuxDeviceUtils.functional(LuxCUDADevice)
    @test cpu_device() isa LuxCPUDevice
    @test gpu_device() isa LuxCPUDevice
    @test_throws LuxDeviceUtils.LuxDeviceSelectionException gpu_device(;
        force_gpu_usage=true)
end

using LuxCUDA

@testset "Loaded Trigger Package" begin
    @test LuxDeviceUtils.GPU_DEVICE[] === nothing

    if LuxDeviceUtils.functional(LuxCUDADevice)
        @info "LuxCUDA is functional"
        @test gpu_device() isa LuxCUDADevice
        @test gpu_device(; force_gpu_usage=true) isa LuxCUDADevice
    else
        @info "LuxCUDA is NOT functional"
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
    aType = LuxDeviceUtils.functional(LuxCUDADevice) ? CuArray : Array
    rngType = LuxDeviceUtils.functional(LuxCUDADevice) ? CUDA.RNG : Random.AbstractRNG

    ps_xpu = ps |> device
    @test get_device(ps_xpu) isa LuxCUDADevice
    @test ps_xpu.a.c isa aType
    @test ps_xpu.b isa aType
    @test ps_xpu.a.d == ps.a.d
    @test ps_xpu.e == ps.e
    @test ps_xpu.d == ps.d
    @test ps_xpu.rng_default isa rngType
    @test ps_xpu.rng == ps.rng

    if LuxDeviceUtils.functional(LuxCUDADevice)
        @test ps_xpu.one_elem isa CuArray
        @test ps_xpu.farray isa CuArray
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
    @test ps_cpu.e == ps.e
    @test ps_cpu.d == ps.d
    @test ps_cpu.rng_default isa Random.TaskLocalRNG
    @test ps_cpu.rng == ps.rng

    if LuxDeviceUtils.functional(LuxCUDADevice)
        @test ps_cpu.one_elem isa Array
        @test ps_cpu.farray isa Array
    else
        @test ps_cpu.one_elem isa Zygote.OneElement
        @test ps_cpu.farray isa Fill
    end

    ps_mixed = (; a=rand(2), b=device(rand(2)))
    @test_throws ArgumentError get_device(ps_mixed)
end

@testset "Multiple Devices CUDA" begin
    if LuxDeviceUtils.functional(LuxCUDADevice)
        ps = (; weight=rand(Float32, 10), bias=rand(Float32, 10))
        ps_cpu = deepcopy(ps)
        cdev = cpu_device()
        for idx in 1:length(CUDA.devices())
            cuda_device = gpu_device(idx)
            @test typeof(cuda_device.device) <: CUDA.CuDevice
            @test cuda_device.device.handle == (idx - 1)

            ps = ps |> cuda_device
            @test ps.weight isa CuArray
            @test ps.bias isa CuArray
            @test CUDA.device(ps.weight).handle == idx - 1
            @test CUDA.device(ps.bias).handle == idx - 1
            @test isequal(cdev(ps.weight), ps_cpu.weight)
            @test isequal(cdev(ps.bias), ps_cpu.bias)
        end

        ps = ps |> cdev
        @test ps.weight isa Array
        @test ps.bias isa Array
    end
end

using SparseArrays

@testset "CUDA Sparse Arrays" begin
    if LuxDeviceUtils.functional(LuxCUDADevice)
        ps = (; weight=sprand(Float32, 10, 10, 0.1), bias=sprand(Float32, 10, 0.1))
        ps_cpu = deepcopy(ps)
        cdev = cpu_device()
        for idx in 1:length(CUDA.devices())
            cuda_device = gpu_device(idx)
            @test typeof(cuda_device.device) <: CUDA.CuDevice
            @test cuda_device.device.handle == (idx - 1)

            ps = ps |> cuda_device
            @test ps.weight isa CUSPARSE.CuSparseMatrixCSC
            @test ps.bias isa CUSPARSE.CuSparseVector
            @test get_device(ps.weight).device.handle == idx - 1
            @test get_device(ps.bias).device.handle == idx - 1
            @test isequal(cdev(ps.weight), ps_cpu.weight)
            @test isequal(cdev(ps.bias), ps_cpu.bias)
        end

        ps = ps |> cdev
        @test ps.weight isa SparseMatrixCSC
        @test ps.bias isa SparseVector
    end
end
