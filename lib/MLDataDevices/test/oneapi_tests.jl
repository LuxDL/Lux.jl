using MLDataDevices, Random, Test
using ArrayInterface: parameterless_type

@testset "CPU Fallback" begin
    @test !MLDataDevices.functional(oneAPIDevice)
    @test cpu_device() isa CPUDevice
    @test gpu_device() isa CPUDevice
    @test_throws MLDataDevices.Internal.DeviceSelectionException gpu_device(; force=true)
    @test_throws Exception default_device_rng(oneAPIDevice())
end

using oneAPI

@testset "Loaded Trigger Package" begin
    @test MLDataDevices.GPU_DEVICE[] === nothing

    if MLDataDevices.functional(oneAPIDevice)
        @info "oneAPI is functional"
        @test gpu_device() isa oneAPIDevice
        @test gpu_device(; force=true) isa oneAPIDevice
    else
        @info "oneAPI is NOT functional"
        @test gpu_device() isa oneAPIDevice
        @test_throws MLDataDevices.Internal.DeviceSelectionException gpu_device(;
            force=true)
    end
    @test MLDataDevices.GPU_DEVICE[] !== nothing
end

using FillArrays, Zygote  # Extensions

@testset "Data Transfer" begin
    ps = (a=(c=zeros(10, 1), d=1), b=ones(10, 1), e=:c,
        d="string", mixed=[2.0f0, 3.0, ones(2, 3)],  # mixed array types
        range=1:10,
        rng_default=Random.default_rng(), rng=MersenneTwister(),
        one_elem=Zygote.OneElement(2.0f0, (2, 3), (1:3, 1:4)), farray=Fill(1.0f0, (2, 3)))

    device = gpu_device()
    aType = MLDataDevices.functional(oneAPIDevice) ? oneArray : Array
    rngType = MLDataDevices.functional(oneAPIDevice) ? oneAPI.GPUArrays.RNG :
              Random.AbstractRNG

    ps_xpu = ps |> device
    @test get_device(ps_xpu) isa oneAPIDevice
    @test get_device_type(ps_xpu) <: oneAPIDevice
    @test ps_xpu.a.c isa aType
    @test ps_xpu.b isa aType
    @test ps_xpu.a.d == ps.a.d
    @test ps_xpu.mixed isa Vector
    @test ps_xpu.mixed[1] isa Float32
    @test ps_xpu.mixed[2] isa Float64
    @test ps_xpu.mixed[3] isa aType
    @test ps_xpu.range isa aType
    @test ps_xpu.e == ps.e
    @test ps_xpu.d == ps.d
    @test ps_xpu.rng_default isa rngType
    @test get_device(ps_xpu.rng_default) isa oneAPIDevice
    @test get_device_type(ps_xpu.rng_default) <: oneAPIDevice
    @test ps_xpu.rng == ps.rng
    @test get_device(ps_xpu.rng) === nothing
    @test get_device_type(ps_xpu.rng) <: Nothing

    if MLDataDevices.functional(oneAPIDevice)
        @test ps_xpu.one_elem isa oneArray
        @test ps_xpu.farray isa oneArray
    else
        @test ps_xpu.one_elem isa Zygote.OneElement
        @test ps_xpu.farray isa Fill
    end

    ps_cpu = ps_xpu |> cpu_device()
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
    @test ps_cpu.range isa Array
    @test ps_cpu.e == ps.e
    @test ps_cpu.d == ps.d
    @test ps_cpu.rng_default isa Random.TaskLocalRNG
    @test get_device(ps_cpu.rng_default) === nothing
    @test get_device_type(ps_cpu.rng_default) <: Nothing
    @test ps_cpu.rng == ps.rng
    @test get_device(ps_cpu.rng) === nothing
    @test get_device_type(ps_cpu.rng) <: Nothing

    if MLDataDevices.functional(oneAPIDevice)
        @test ps_cpu.one_elem isa Array
        @test ps_cpu.farray isa Array
    else
        @test ps_cpu.one_elem isa Zygote.OneElement
        @test ps_cpu.farray isa Fill
    end

    ps_mixed = (; a=rand(2), b=device(rand(2)))
    @test_throws ArgumentError get_device(ps_mixed)
    @test_throws ArgumentError get_device_type(ps_mixed)

    @testset "get_device_type compile constant" begin
        x = rand(10, 10) |> device
        ps = (; weight=x, bias=x, d=(x, x))

        return_val(x) = Val(get_device_type(x))  # If it is a compile time constant then type inference will work
        @test @inferred(return_val(ps)) isa Val{parameterless_type(typeof(device))}

        return_val2(x) = Val(get_device(x))
        @test @inferred(return_val2(ps)) isa Val{get_device(x)}
    end
end

@testset "Functions" begin
    if MLDataDevices.functional(oneAPIDevice)
        @test get_device(tanh) isa MLDataDevices.UnknownDevice
        @test get_device_type(tanh) <: MLDataDevices.UnknownDevice

        f(x, y) = () -> (x, x .^ 2, y)

        ff = f([1, 2, 3], 1)
        @test get_device(ff) isa CPUDevice
        @test get_device_type(ff) <: CPUDevice

        ff_xpu = ff |> oneAPIDevice()
        @test get_device(ff_xpu) isa oneAPIDevice
        @test get_device_type(ff_xpu) <: oneAPIDevice

        ff_cpu = ff_xpu |> cpu_device()
        @test get_device(ff_cpu) isa CPUDevice
        @test get_device_type(ff_cpu) <: CPUDevice
    end
end

@testset "Wrapper Arrays" begin
    if MLDataDevices.functional(oneAPIDevice)
        x = rand(10, 10) |> oneAPIDevice()
        @test get_device(x) isa oneAPIDevice
        @test get_device_type(x) <: oneAPIDevice
        x_view = view(x, 1:5, 1:5)
        @test get_device(x_view) isa oneAPIDevice
        @test get_device_type(x_view) <: oneAPIDevice
    end
end

@testset "setdevice!" begin
    if MLDataDevices.functional(oneAPIDevice)
        @test_logs (:warn,
            "Support for Multi Device oneAPI hasn't been implemented yet. Ignoring the device setting.") MLDataDevices.set_device!(
            oneAPIDevice, nothing, 1)
    end
end
