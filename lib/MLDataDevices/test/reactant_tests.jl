using MLDataDevices, Random, Test
using ArrayInterface: parameterless_type

@testset "CPU Fallback" begin
    @test !MLDataDevices.functional(ReactantDevice)
    @test cpu_device() isa CPUDevice
    @test reactant_device() isa CPUDevice
    @test_throws MLDataDevices.Internal.DeviceSelectionException reactant_device(;
        force=true
    )
    @test_throws Exception default_device_rng(ReactantDevice())
end

using Reactant

@testset "Loaded Trigger Package" begin
    if MLDataDevices.functional(ReactantDevice)
        @info "Reactant is functional"
        @test reactant_device() isa ReactantDevice
        @test reactant_device(; force=true) isa ReactantDevice
    else
        @info "Reactant is NOT functional"
        @test reactant_device() isa CPUDevice
        @test_throws MLDataDevices.Internal.DeviceSelectionException reactant_device(;
            force=true
        )
    end
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
        one_elem2=FillArrays.OneElement(2.0f0, (2, 3), (1:3, 1:4)),
        zeros_fa=Zeros{Float32}((2, 3)),
        ones_fa=Ones{Float32}((2, 3)),
    )

    device = reactant_device()
    aType = MLDataDevices.functional(ReactantDevice) ? Reactant.ConcreteRArray : Array
    rngType = (
        MLDataDevices.functional(ReactantDevice) ? Reactant.ReactantRNG : Random.AbstractRNG
    )

    ps_xpu = device(ps)
    @test get_device(ps_xpu) isa ReactantDevice
    @test get_device_type(ps_xpu) <: ReactantDevice
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
    @test ps_xpu.rng isa rngType
    if MLDataDevices.functional(ReactantDevice)
        @test get_device(ps_xpu.rng_default) isa ReactantDevice
        @test get_device_type(ps_xpu.rng_default) <: ReactantDevice
        @test get_device(ps_xpu.rng) isa ReactantDevice
        @test get_device_type(ps_xpu.rng) <: ReactantDevice
    end

    if MLDataDevices.functional(ReactantDevice)
        @test ps_xpu.one_elem isa Reactant.RArray
        @test ps_xpu.farray isa Fill{<:Reactant.ConcreteRNumber}
        @test ps_xpu.one_elem2 isa FillArrays.OneElement{<:Reactant.ConcreteRNumber}
        @test ps_xpu.zeros_fa isa Zeros{<:Reactant.ConcreteRNumber}
        @test ps_xpu.ones_fa isa Ones{<:Reactant.ConcreteRNumber}
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
    @test ps_cpu.rng isa Random.TaskLocalRNG
    @test get_device(ps_cpu.rng) === nothing
    @test get_device_type(ps_cpu.rng) <: Nothing

    if MLDataDevices.functional(ReactantDevice)
        @test ps_cpu.one_elem isa Array
        @test ps_cpu.farray isa Fill{Float32}
        @test ps_cpu.one_elem2 isa FillArrays.OneElement{Float32}
        @test ps_cpu.zeros_fa isa Zeros{Float32}
        @test ps_cpu.ones_fa isa Ones{Float32}
    end

    ps_mixed = (; a=rand(2), b=device(rand(2)))
    @test get_device(ps_mixed) isa ReactantDevice
    @test get_device_type(ps_mixed) <: ReactantDevice

    @testset "get_device_type compile constant" begin
        x = device(rand(10, 10))
        ps = (; weight=x, bias=x, d=(x, x))

        return_val(x) = Val(get_device_type(x))  # If it is a compile time constant then type inference will work
        @test @inferred(return_val(ps)) isa Val{parameterless_type(typeof(device))}

        return_val2(x) = Val(get_device(x))
        @test_throws TypeError @inferred(return_val2(ps))
    end

    @testset "Issue #1129: no new object" begin
        x = device(rand(Float32, 10, 10))
        y = device(x)
        # This is hard to guarantee since we rely on Reactant to do the transfer for
        # sharding
        @test_broken x === y
    end
end

@testset "Functions" begin
    if MLDataDevices.functional(ReactantDevice)
        @test get_device(tanh) isa MLDataDevices.UnknownDevice
        @test get_device_type(tanh) <: MLDataDevices.UnknownDevice

        f(x, y) = () -> (x, x .^ 2, y)

        ff = f([1, 2, 3], 1)
        @test get_device(ff) isa CPUDevice
        @test get_device_type(ff) <: CPUDevice

        ff_xpu = ReactantDevice()(ff)
        @test get_device(ff_xpu) isa ReactantDevice
        @test get_device_type(ff_xpu) <: ReactantDevice

        ff_cpu = cpu_device()(ff_xpu)
        @test get_device(ff_cpu) isa CPUDevice
        @test get_device_type(ff_cpu) <: CPUDevice
    end
end

@testset "Wrapped Arrays" begin
    if MLDataDevices.functional(ReactantDevice)
        x = ReactantDevice()(rand(10, 10))
        @test get_device(x) isa ReactantDevice
        @test get_device_type(x) <: ReactantDevice
        x_view = view(x, 1:5, 1:5)
        @test get_device(x_view) isa ReactantDevice
        @test get_device_type(x_view) <: ReactantDevice
    end
end

@testset "setdevice!" begin
    if MLDataDevices.functional(ReactantDevice)
        @test_logs (
            :warn,
            "Setting device for `ReactantDevice` hasn't been implemented yet. Ignoring the device setting.",
        ) MLDataDevices.set_device!(ReactantDevice, nothing, 1)
    end
end

@testset "Random" begin
    if MLDataDevices.functional(ReactantDevice)
        dev = reactant_device()
        rng = MersenneTwister(123)
        @test dev(rng) isa Reactant.ReactantRNG
    end
end

@testset "Track Numbers" begin
    if MLDataDevices.functional(ReactantDevice)
        dev = reactant_device(; track_numbers=Float32)
        x = dev(2.0f0)
        @test x isa ConcreteRNumber{Float32}
        x = dev(2.0)
        @test x isa Float64
    end
end
