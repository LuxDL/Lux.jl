using Adapt, DeviceUtils, ComponentArrays, Random
using ArrayInterface: parameterless_type
using ChainRulesTestUtils: test_rrule
using ReverseDiff, Tracker, ForwardDiff
using SparseArrays, FillArrays, Zygote, RecursiveArrayTools

@testset "https://github.com/LuxDL/DeviceUtils.jl/issues/10 patch" begin
    dev = CPUDevice()
    ps = (; weight=randn(10, 1), bias=randn(1))

    ps_ca = ps |> ComponentArray

    ps_ca_dev = ps_ca |> dev

    @test ps_ca_dev isa ComponentArray

    @test ps_ca_dev.weight == ps.weight
    @test ps_ca_dev.bias == ps.bias

    @test ps_ca_dev == (ps |> dev |> ComponentArray)
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
    x_tracker_dev = Tracker.param(x) |> gdev
    @test get_device(x_tracker_dev) isa parameterless_type(typeof(gdev))
    x_tracker_dev = Tracker.param.(x) |> gdev
    @test get_device(x_tracker_dev) isa parameterless_type(typeof(gdev))

    x_fdiff = ForwardDiff.Dual.(x)
    @test get_device(x_fdiff) isa CPUDevice
    x_fdiff_dev = ForwardDiff.Dual.(x) |> gdev
    @test get_device(x_fdiff_dev) isa parameterless_type(typeof(gdev))
end

@testset "CRC Tests" begin
    dev = cpu_device() # Other devices don't work with FiniteDifferences.jl
    test_rrule(Adapt.adapt_storage, dev, randn(Float64, 10); check_inferred=true)

    gdev = gpu_device()
    if !(gdev isa MetalDevice)  # On intel devices causes problems
        x = randn(10)
        ∂dev, ∂x = Zygote.gradient(sum ∘ Adapt.adapt_storage, gdev, x)
        @test ∂dev === nothing
        @test ∂x ≈ ones(10)

        x = randn(10) |> gdev
        ∂dev, ∂x = Zygote.gradient(sum ∘ Adapt.adapt_storage, cpu_device(), x)
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

    diffeqarray_dev = diffeqarray |> gdev
    @test get_device(diffeqarray_dev) isa parameterless_type(typeof(gdev))

    vecarray = VectorOfArray([rand(10) for _ in 1:10])
    @test get_device(vecarray) isa CPUDevice

    vecarray_dev = vecarray |> gdev
    @test get_device(vecarray_dev) isa parameterless_type(typeof(gdev))
end

@testset "CPU default rng" begin
    @test default_device_rng(CPUDevice()) isa Random.TaskLocalRNG
end

@testset "CPU setdevice!" begin
    @test_logs (:warn,
        "Setting device for `CPUDevice` doesn't make sense. Ignoring the device setting.") DeviceUtils.set_device!(
        CPUDevice, nothing, 1)
end

@testset "get_device on Arrays" begin
    x = rand(10, 10)
    x_view = view(x, 1:5, 1:5)

    @test get_device(x) isa CPUDevice
    @test get_device(x_view) isa CPUDevice

    struct MyArrayType <: AbstractArray{Float32, 2}
        data::Array{Float32, 2}
    end

    x_custom = MyArrayType(rand(10, 10))

    @test get_device(x_custom) isa CPUDevice
end

@testset "loaded and functional" begin
    @test DeviceUtils.loaded(CPUDevice)
    @test DeviceUtils.functional(CPUDevice)
end

@testset "writing to preferences" begin
    @test_logs (:info,
        "Deleted the local preference for `gpu_backend`. Restart Julia to use the new backend.") gpu_backend!()

    for backend in (:CUDA, :AMDGPU, :oneAPI, :Metal, AMDGPUDevice(),
        CUDADevice(), MetalDevice(), oneAPIDevice())
        backend_name = backend isa Symbol ? string(backend) :
                       DeviceUtils._get_device_name(backend)
        @test_logs (:info,
            "GPU backend has been set to $(backend_name). Restart Julia to use the new backend.") gpu_backend!(backend)
    end

    gpu_backend!(:CUDA)
    @test_logs (:info, "GPU backend is already set to CUDA. No action is required.") gpu_backend!(:CUDA)

    @test_throws ArgumentError gpu_backend!("my_backend")
end

@testset "get_device_type compile constant" begin
    x = rand(10, 10)
    ps = (; weight=x, bias=x, d=(x, x))

    return_val(x) = Val(get_device_type(x))  # If it is a compile time constant then type inference will work
    @test @inferred(return_val(ps)) isa Val{typeof(cpu_device())}

    return_val2(x) = Val(get_device(x))
    @test @inferred(return_val2(ps)) isa Val{cpu_device()}
end
