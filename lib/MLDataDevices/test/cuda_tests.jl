using MLDataDevices, Random, Functors, Test
using ArrayInterface: parameterless_type

@testset "CPU Fallback" begin
    @test !MLDataDevices.functional(CUDADevice)
    @test cpu_device() isa CPUDevice
    @test gpu_device() isa CPUDevice
    @test_throws MLDataDevices.Internal.DeviceSelectionException gpu_device(; force=true)
    @test_throws Exception default_device_rng(CUDADevice(nothing))
    @test_logs (:warn, "`CUDA.jl` hasn't been loaded. Ignoring the device setting.") MLDataDevices.set_device!(
        CUDADevice, nothing, 1
    )
end

using LuxCUDA

if !LuxCUDA.functional()
    @warn "LuxCUDA.jl is not functional. Skipping CUDA tests."
    exit()
end

@testset "Loaded Trigger Package" begin
    @test MLDataDevices.GPU_DEVICE[] === nothing

    if MLDataDevices.functional(CUDADevice)
        @info "LuxCUDA is functional"
        @test gpu_device() isa CUDADevice
        @test gpu_device(; force=true) isa CUDADevice
    else
        @info "LuxCUDA is NOT functional"
        @test gpu_device() isa CPUDevice
        @test_throws MLDataDevices.Internal.DeviceSelectionException gpu_device(;
            force=true
        )
    end
    @test MLDataDevices.GPU_DEVICE[] !== nothing
end

using FillArrays, Zygote  # Extensions

@testset "Data Transfer" begin
    ps = (
        a=(c=zeros(10, 1), d=1),
        b=ones(10, 1),
        e=:c,
        d="string",
        mixed=[2.0f0, 3.0, ones(2, 3)],  # mixed array types
        range=1:10,
        rng_default=Random.default_rng(),
        rng=MersenneTwister(),
        one_elem=Zygote.OneElement(2.0f0, (2, 3), (1:3, 1:4)),
        farray=Fill(1.0f0, (2, 3)),
    )

    device = gpu_device()
    aType = MLDataDevices.functional(CUDADevice) ? CuArray : Array
    rngType = MLDataDevices.functional(CUDADevice) ? CUDA.RNG : Random.AbstractRNG

    ps_xpu = device(ps)
    @test get_device(ps_xpu) isa CUDADevice
    @test get_device_type(ps_xpu) <: CUDADevice
    @test ps_xpu.a.c isa aType
    @test ps_xpu.b isa aType
    @test ps_xpu.a.d == ps.a.d
    @test ps_xpu.mixed isa Vector
    @test ps_xpu.mixed[1] isa Float32
    @test ps_xpu.mixed[2] isa Float64
    @test ps_xpu.mixed[3] isa aType
    @test ps_xpu.range isa AbstractRange
    @test ps_xpu.e == ps.e
    @test ps_xpu.d == ps.d
    @test ps_xpu.rng_default isa rngType
    @test get_device(ps_xpu.rng_default) isa CUDADevice
    @test get_device_type(ps_xpu.rng_default) <: CUDADevice
    @test ps_xpu.rng == ps.rng
    @test get_device(ps_xpu.rng) === nothing
    @test get_device_type(ps_xpu.rng) <: Nothing

    if MLDataDevices.functional(CUDADevice)
        @test ps_xpu.one_elem isa CuArray
        @test ps_xpu.farray isa CuArray
    else
        @test ps_xpu.one_elem isa Zygote.OneElement
        @test ps_xpu.farray isa Fill
    end

    ps_cpu = cpu_device()(ps_xpu)
    @test get_device(ps_cpu) isa CPUDevice
    @test get_device_type(ps_cpu) <: CPUDevice
    @test ps_cpu.a.c isa Array
    @test ps_cpu.b isa Array
    @test ps_cpu.a.c == ps.a.c
    @test ps_cpu.b == ps.b
    @test ps_cpu.a.d == ps.a.d
    @test ps_cpu.mixed isa Vector
    @test ps_cpu.mixed[1] isa Float32
    @test ps_cpu.mixed[2] isa Float64
    @test ps_cpu.mixed[3] isa Array
    @test ps_cpu.range isa AbstractRange
    @test ps_cpu.e == ps.e
    @test ps_cpu.d == ps.d
    @test ps_cpu.rng_default isa Random.TaskLocalRNG
    @test get_device(ps_cpu.rng_default) === nothing
    @test get_device_type(ps_cpu.rng_default) <: Nothing
    @test ps_cpu.rng == ps.rng
    @test get_device(ps_cpu.rng) === nothing
    @test get_device_type(ps_cpu.rng) <: Nothing

    if MLDataDevices.functional(CUDADevice)
        @test ps_cpu.one_elem isa Array
        @test ps_cpu.farray isa Array
    else
        @test ps_cpu.one_elem isa Zygote.OneElement
        @test ps_cpu.farray isa Fill
    end

    struct MyStruct
        x::Any
    end

    Functors.@functor MyStruct

    data = MyStruct(rand(10))
    @test get_device(data) isa CPUDevice
    @test get_device_type(data) <: CPUDevice
    data_dev = device(data)
    if MLDataDevices.functional(CUDADevice)
        @test get_device(data_dev) isa CUDADevice
        @test get_device_type(data_dev) <: CUDADevice
    else
        @test get_device(data_dev) isa CPUDevice
        @test get_device_type(data_dev) <: CPUDevice
    end

    ps_mixed = (; a=rand(2), c=(rand(2), 1), st=MyStruct(rand(2)), b=device(rand(2)))
    @test get_device(ps_mixed.st) isa CPUDevice
    @test get_device_type(ps_mixed.st) <: CPUDevice
    @test get_device(ps_mixed.c) isa CPUDevice
    @test get_device_type(ps_mixed.c) <: CPUDevice
    @test_throws ArgumentError get_device(ps_mixed)
    @test_throws ArgumentError get_device_type(ps_mixed)

    dev = gpu_device()
    x = rand(Float32, 10, 2)
    x_dev = dev(x)
    @test get_device(x_dev) isa parameterless_type(typeof(dev))
    @test get_device_type(x_dev) <: parameterless_type(typeof(dev))

    if MLDataDevices.functional(CUDADevice)
        dev2 = gpu_device(length(CUDA.devices()))
        x_dev2 = dev2(x_dev)
        @test get_device(x_dev2) isa typeof(dev2)
        @test get_device_type(x_dev2) <: parameterless_type(typeof(dev2))
    end

    @testset "get_device_type compile constant" begin
        x = device(rand(10, 10))
        ps = (; weight=x, bias=x, d=(x, x))

        return_val(x) = Val(get_device_type(x))  # If it is a compile time constant then type inference will work
        @test @inferred(return_val(ps)) isa Val{parameterless_type(typeof(device))}

        return_val2(x) = Val(get_device(x))
        @test_throws ErrorException @inferred(return_val2(ps))
    end

    @testset "Issue #1129: no new object" begin
        x = device(rand(Float32, 10, 10))
        y = device(x)
        @test x === y
    end

    @testset "Character Arrays" begin
        # Test that character arrays can be transferred to GPU
        char_array = ['a', 'b', 'c']
        char_array_xpu = device(char_array)

        if MLDataDevices.functional(CUDADevice)
            @test char_array_xpu isa CuArray{Char}
            @test Array(char_array_xpu) == char_array

            # Test transfer back to CPU
            char_array_cpu = cpu_device()(char_array_xpu)
            @test char_array_cpu isa Array{Char}
            @test char_array_cpu == char_array
        else
            @test char_array_xpu isa Array{Char}
            @test char_array_xpu == char_array
        end
    end

    @testset "IsBits Types" begin
        # Test that custom isbits types can be transferred to GPU
        struct SimpleBits
            field::Int32
        end

        isbits_array = [SimpleBits(1), SimpleBits(2), SimpleBits(3)]
        isbits_array_xpu = device(isbits_array)

        if MLDataDevices.functional(CUDADevice)
            @test isbits_array_xpu isa CuArray{SimpleBits}
            @test Array(isbits_array_xpu) == isbits_array

            # Test transfer back to CPU
            isbits_array_cpu = cpu_device()(isbits_array_xpu)
            @test isbits_array_cpu isa Array{SimpleBits}
            @test isbits_array_cpu == isbits_array
        else
            @test isbits_array_xpu isa Array{SimpleBits}
            @test isbits_array_xpu == isbits_array
        end
    end
end

@testset "Functions" begin
    if MLDataDevices.functional(CUDADevice)
        @test get_device(tanh) isa MLDataDevices.UnknownDevice
        @test get_device_type(tanh) <: MLDataDevices.UnknownDevice

        f(x, y) = () -> (x, x .^ 2, y)

        ff = f([1, 2, 3], 1)
        @test get_device(ff) isa CPUDevice
        @test get_device_type(ff) <: CPUDevice

        ff_xpu = CUDADevice()(ff)
        @test get_device(ff_xpu) isa CUDADevice
        @test get_device_type(ff_xpu) <: CUDADevice

        ff_cpu = cpu_device()(ff_xpu)
        @test get_device(ff_cpu) isa CPUDevice
        @test get_device_type(ff_cpu) <: CPUDevice
    end
end

@testset "Wrapped Arrays" begin
    if MLDataDevices.functional(CUDADevice)
        x = CUDADevice()(rand(10, 10))
        @test get_device(x) isa CUDADevice
        @test get_device_type(x) <: CUDADevice
        x_view = view(x, 1:5, 1:5)
        @test get_device(x_view) isa CUDADevice
        @test get_device_type(x_view) <: CUDADevice
    end
end

@testset "Multiple Devices CUDA" begin
    if MLDataDevices.functional(CUDADevice)
        ps = (; weight=rand(Float32, 10), bias=rand(Float32, 10))
        ps_cpu = deepcopy(ps)
        cdev = cpu_device()
        for idx in 1:length(CUDA.devices())
            cuda_device = gpu_device(idx)
            @test typeof(cuda_device.device) <: CUDA.CuDevice
            @test cuda_device.device.handle == (idx - 1)

            ps = cuda_device(ps)
            @test ps.weight isa CuArray
            @test ps.bias isa CuArray
            @test CUDA.device(ps.weight).handle == idx - 1
            @test CUDA.device(ps.bias).handle == idx - 1
            @test isequal(cdev(ps.weight), ps_cpu.weight)
            @test isequal(cdev(ps.bias), ps_cpu.bias)
        end

        ps = cdev(ps)
        @test ps.weight isa Array
        @test ps.bias isa Array
    end
end

using SparseArrays

@testset "CUDA Sparse Arrays" begin
    if MLDataDevices.functional(CUDADevice)
        ps = (; weight=sprand(Float32, 10, 10, 0.1), bias=sprand(Float32, 10, 0.1))
        ps_cpu = deepcopy(ps)
        cdev = cpu_device()
        for idx in 1:length(CUDA.devices())
            cuda_device = gpu_device(idx)
            @test typeof(cuda_device.device) <: CUDA.CuDevice
            @test cuda_device.device.handle == (idx - 1)

            ps = cuda_device(ps_cpu)
            @test ps.weight isa CUSPARSE.CuSparseMatrixCSC
            @test ps.bias isa CUSPARSE.CuSparseVector
            @test get_device(ps.weight).device.handle == idx - 1
            @test get_device(ps.bias).device.handle == idx - 1
            @test isequal(cdev(ps.weight), ps_cpu.weight)
            @test isequal(cdev(ps.bias), ps_cpu.bias)
        end

        ps = cdev(ps)
        @test ps.weight isa SparseMatrixCSC
        @test ps.bias isa SparseVector
    end
end

@testset "setdevice!" begin
    if MLDataDevices.functional(CUDADevice)
        for i in 1:10
            @test_nowarn MLDataDevices.set_device!(CUDADevice, nothing, i)
        end
    end
end
