using Adapt, LuxDeviceUtils, ComponentArrays, Random
using ArrayInterface: parameterless_type
using ChainRulesTestUtils: test_rrule
using ReverseDiff, Tracker, ForwardDiff
using SparseArrays, FillArrays, Zygote, RecursiveArrayTools

@testset "https://github.com/LuxDL/LuxDeviceUtils.jl/issues/10 patch" begin
    dev = LuxCPUDevice()
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
    @test get_device(x_rdiff) isa LuxCPUDevice
    x_rdiff = ReverseDiff.track.(x)
    @test get_device(x_rdiff) isa LuxCPUDevice

    gdev = gpu_device()

    x_tracker = Tracker.param(x)
    @test get_device(x_tracker) isa LuxCPUDevice
    x_tracker = Tracker.param.(x)
    @test get_device(x_tracker) isa LuxCPUDevice
    x_tracker_dev = Tracker.param(x) |> gdev
    @test get_device(x_tracker_dev) isa parameterless_type(typeof(gdev))
    x_tracker_dev = Tracker.param.(x) |> gdev
    @test get_device(x_tracker_dev) isa parameterless_type(typeof(gdev))

    x_fdiff = ForwardDiff.Dual.(x)
    @test get_device(x_fdiff) isa LuxCPUDevice
    x_fdiff_dev = ForwardDiff.Dual.(x) |> gdev
    @test get_device(x_fdiff_dev) isa parameterless_type(typeof(gdev))
end

@testset "CRC Tests" begin
    dev = cpu_device() # Other devices don't work with FiniteDifferences.jl
    test_rrule(Adapt.adapt_storage, dev, randn(Float64, 10); check_inferred=true)

    gdev = gpu_device()
    if !(gdev isa LuxMetalDevice)  # On intel devices causes problems
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
    @test get_device(diffeqarray) isa LuxCPUDevice

    diffeqarray_dev = diffeqarray |> gdev
    @test get_device(diffeqarray_dev) isa parameterless_type(typeof(gdev))

    vecarray = VectorOfArray([rand(10) for _ in 1:10])
    @test get_device(vecarray) isa LuxCPUDevice

    vecarray_dev = vecarray |> gdev
    @test get_device(vecarray_dev) isa parameterless_type(typeof(gdev))
end
