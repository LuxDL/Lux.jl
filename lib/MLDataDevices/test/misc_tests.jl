using Adapt, MLDataDevices, ComponentArrays, Random, Test
using ArrayInterface: parameterless_type
using ChainRulesTestUtils: test_rrule
using ReverseDiff, Tracker, ForwardDiff
using SparseArrays, FillArrays, Zygote, RecursiveArrayTools
using Functors: Functors

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "none"))

@testset "Issues Patches" begin
    @testset "#10 patch" begin
        dev = CPUDevice()
        ps = (; weight=randn(10, 1), bias=randn(1))

        ps_ca = ComponentArray(ps)

        ps_ca_dev = dev(ps_ca)

        @test ps_ca_dev isa ComponentArray

        @test ps_ca_dev.weight == ps.weight
        @test ps_ca_dev.bias == ps.bias

        @test ps_ca_dev == (ComponentArray(dev(ps)))
    end
end

@testset "AD Types" begin
    x = randn(Float32, 10)

    x_rdiff = ReverseDiff.track(x)
    @test get_device(x_rdiff) isa CPUDevice
    x_rdiff = ReverseDiff.track.(x)
    @test get_device(x_rdiff) isa CPUDevice

    gdev = gpu_device()

    x_tracker = Tracker.param(x)
    @test get_device(x_tracker) isa CPUDevice
    x_tracker = Tracker.param.(x)
    @test get_device(x_tracker) isa CPUDevice
    x_tracker_dev = gdev(Tracker.param(x))
    @test get_device(x_tracker_dev) isa parameterless_type(typeof(gdev))
    x_tracker_dev = gdev(Tracker.param.(x))
    @test get_device(x_tracker_dev) isa parameterless_type(typeof(gdev))

    x_fdiff = ForwardDiff.Dual.(x)
    @test get_device(x_fdiff) isa CPUDevice
    x_fdiff_dev = gdev(ForwardDiff.Dual.(x))
    @test get_device(x_fdiff_dev) isa parameterless_type(typeof(gdev))
end

@testset "CRC Tests" begin
    dev = cpu_device() # Other devices don't work with FiniteDifferences.jl
    test_rrule(Adapt.adapt_storage, dev, randn(Float64, 10); check_inferred=false)

    gdev = gpu_device()
    if !(gdev isa MetalDevice)  # On intel devices causes problems
        x = randn(10)
        ∂dev, ∂x = Zygote.gradient(sum ∘ Adapt.adapt, gdev, x)
        @test ∂dev === nothing
        @test ∂x ≈ ones(10)

        x = gdev(randn(10))
        ∂dev, ∂x = Zygote.gradient(sum ∘ Adapt.adapt, cpu_device(), x)
        @test ∂dev === nothing
        @test ∂x ≈ gdev(ones(10))
        @test get_device(∂x) isa parameterless_type(typeof(gdev))
    end
end

# The following just test for noops
@testset "NoOps CPU" begin
    cdev = cpu_device()

    @test cdev(sprand(10, 10, 0.9)) isa SparseMatrixCSC
    @test cdev(1:10) isa AbstractRange
    @test cdev(Zygote.OneElement(2.0f0, (2, 3), (1:3, 1:4))) isa Zygote.OneElement
end

@testset "RecursiveArrayTools" begin
    gdev = gpu_device()

    diffeqarray = DiffEqArray([rand(10) for _ in 1:10], rand(10))
    @test get_device(diffeqarray) isa CPUDevice

    diffeqarray_dev = gdev(diffeqarray)
    @test get_device(diffeqarray_dev) isa parameterless_type(typeof(gdev))

    vecarray = VectorOfArray([rand(10) for _ in 1:10])
    @test get_device(vecarray) isa CPUDevice

    vecarray_dev = gdev(vecarray)
    @test get_device(vecarray_dev) isa parameterless_type(typeof(gdev))
end

@testset "CPU default rng" begin
    @test default_device_rng(CPUDevice()) isa Random.TaskLocalRNG
end

@testset "CPU setdevice!" begin
    @test_logs (
        :warn,
        "Setting device for `CPUDevice` doesn't make sense. Ignoring the device setting.",
    ) MLDataDevices.set_device!(CPUDevice, nothing, 1)
end

@testset "get_device on Arrays" begin
    x = rand(10, 10)
    x_view = view(x, 1:5, 1:5)

    @test get_device(x) isa CPUDevice
    @test get_device(x_view) isa CPUDevice

    struct MyArrayType <: AbstractArray{Float32,2}
        data::Array{Float32,2}
    end

    x_custom = MyArrayType(rand(10, 10))

    @test get_device(x_custom) isa CPUDevice
end

@testset "loaded and functional" begin
    @test MLDataDevices.loaded(CPUDevice)
    @test MLDataDevices.functional(CPUDevice)
end

@testset "writing to preferences" begin
    @test_logs (
        :info,
        "Deleted the local preference for `gpu_backend`. Restart Julia to use the new backend.",
    ) gpu_backend!()

    for backend in (
        :CUDA,
        :AMDGPU,
        :oneAPI,
        :Metal,
        :OpenCL,
        AMDGPUDevice(),
        CUDADevice(),
        MetalDevice(),
        oneAPIDevice(),
        OpenCLDevice(),
    )
        backend_name = if backend isa Symbol
            string(backend)
        else
            MLDataDevices.Internal.get_device_name(backend)
        end
        @test_logs (
            :info,
            "GPU backend has been set to $(backend_name). Restart Julia to use the new backend.",
        ) gpu_backend!(backend)
    end

    gpu_backend!(:CUDA)
    @test_logs (:info, "GPU backend is already set to CUDA. No action is required.") gpu_backend!(
        :CUDA
    )

    @test_throws ArgumentError gpu_backend!("my_backend")
end

@testset "get_device_type compile constant" begin
    x = rand(10, 10)
    ps = (; weight=x, bias=x, d=(x, x))

    return_val(x) = Val(get_device_type(x))  # If it is a compile time constant then type inference will work
    @test @inferred(return_val(ps)) isa Val{CPUDevice}

    return_val2(x) = Val(get_device(x))
    @test @inferred(return_val2(ps)) isa Val{CPUDevice{Missing}()}
end

@testset "undefined references array" begin
    x = Matrix{Any}(undef, 10, 10)

    @test get_device(x) isa MLDataDevices.UnknownDevice
    @test get_device_type(x) <: MLDataDevices.UnknownDevice
end

@testset "isleaf" begin
    @testset "basics" begin
        # Functors.isleaf fallback
        @test MLDataDevices.isleaf(rand(2))
        @test !MLDataDevices.isleaf((rand(2),))

        struct Tleaf
            x::Any
        end
        Functors.@functor Tleaf
        MLDataDevices.isleaf(::Tleaf) = true
        Adapt.adapt_structure(dev::CPUDevice, t::Tleaf) = Tleaf(2 .* dev(t.x))

        cpu = cpu_device()
        t = Tleaf(ones(2))
        y = cpu(t)
        @test y.x == 2 .* ones(2)
        y = cpu([(t,)])
        @test y[1][1].x == 2 .* ones(2)
    end

    @testset "shared parameters" begin
        x = rand(1)
        m = (; a=x, b=x')
        count = Ref(0)
        mcopy = Functors.fmap(m; exclude=MLDataDevices.isleaf) do x
            count[] += 1
            return copy(x)
        end
        @test count[] == 1
        @test mcopy.a === mcopy.b'
    end

    @testset "bitstypes and wrapped types" begin
        struct BitsType
            x::Int32
            y::Float64
        end

        @testset for x in [1.0, 'a', BitsType(1, 2.0)]
            @test MLDataDevices.isleaf([x])
            @test !MLDataDevices.isleaf([x]')
            @test !MLDataDevices.isleaf(transpose([x]))
            @test !MLDataDevices.isleaf(PermutedDimsArray([x;;], (1, 2)))
        end
    end
end

@testset "Zygote.gradient(wrapped arrays)" begin
    using Zygote

    x = rand(4, 4)
    cdev = cpu_device()

    @test get_device(only(Zygote.gradient(x -> sum(abs2, cdev(x)), x'))) isa CPUDevice

    gdev = gpu_device()

    @test get_device(only(Zygote.gradient(x -> sum(abs2, gdev(x)), x'))) isa CPUDevice
end

@testset "Zygote and ChainRules OneElement #1016" begin
    using Zygote

    cdev = cpu_device()
    gdev = gpu_device()

    g = only(Zygote.gradient(x -> cdev(2 .* gdev(x))[1], Float32[1, 2, 3]))
    @test g isa Vector{Float32}

    g = only(
        Zygote.gradient(x -> cdev(gdev(x) * gdev(x))[1, 2], Float32[1 2 3; 4 5 6; 7 8 9])
    )
    @test g isa Matrix{Float32}
end

@testset "OneHotArrays" begin
    using OneHotArrays

    x = onehotbatch("abracadabra", 'a':'e', 'e')
    @test get_device(x) isa CPUDevice

    gdev = gpu_device()
    x_g = gdev(x)
    @test get_device(x_g) isa parameterless_type(typeof(gdev))

    if BACKEND_GROUP == "none" || BACKEND_GROUP == "reactant"
        using Reactant

        rdev = reactant_device()
        x_rd = rdev(x)
        @test get_device(x_rd) isa ReactantDevice
        @test x_rd isa OneHotArray
    end
end

@testset "Device Movement Behavior: FluxML/Flux.jl#2513" begin
    dev = gpu_device()

    x = randn(5)
    x2 = (x, x)
    cx2 = dev(x2)

    @test cx2[1] === cx2[2]
end

@testset "get_device/get_device_type empty" begin
    struct WrapperStruct{T}
        value::T
    end

    x = WrapperStruct((;))
    @test get_device(x) === nothing
    @test get_device_type(x) === Nothing

    x = (rand(Float32, 10), WrapperStruct((;)))
    @test get_device(x) == CPUDevice()
    @test get_device_type(x) == CPUDevice
end

@testset "Character Arrays" begin
    # Test CPU device with character arrays
    cdev = cpu_device()
    char_array = ['a', 'b', 'c', 'd']
    char_array_cpu = cdev(char_array)
    @test char_array_cpu isa Array{Char}
    @test char_array_cpu == char_array
    @test get_device(char_array_cpu) isa CPUDevice

    # Test GPU device with character arrays
    gdev = gpu_device()
    char_array_gpu = gdev(char_array)
    @test get_device(char_array_gpu) isa parameterless_type(typeof(gdev))
end
