using MLDataDevices, Random, Test

@testset "Device Eltype Functionality" begin
    @testset "CPU Device with Eltype" begin
        # Test default behavior (missing eltype)
        cpu_default = cpu_device()
        @test cpu_default isa CPUDevice{Missing}

        # Test eltype=nothing (preserve type)
        cpu_preserve = cpu_device(nothing)
        @test cpu_preserve isa CPUDevice{Nothing}

        # Test specific eltype
        cpu_f32 = cpu_device(Float32)
        @test cpu_f32 isa CPUDevice{Float32}

        cpu_f64 = cpu_device(Float64)
        @test cpu_f64 isa CPUDevice{Float64}

        # Test invalid eltype
        @test_throws MethodError cpu_device(Int)
        @test_throws MethodError cpu_device(String)
    end

    @testset "CPU Device Array Conversion" begin
        x_f64 = [1.0, 2.0, 3.0]  # Float64 input
        x_f32 = Float32[1.0, 2.0, 3.0]  # Float32 input
        x_int = [1, 2, 3]  # Integer input

        # Test missing eltype (preserve)
        cpu_default = cpu_device()
        y_f64 = cpu_default(x_f64)
        @test eltype(y_f64) === Float64
        @test y_f64 == x_f64

        # Test nothing eltype (preserve)
        cpu_preserve = cpu_device(nothing)
        y_f64_preserve = cpu_preserve(x_f64)
        @test eltype(y_f64_preserve) === Float64
        @test y_f64_preserve == x_f64

        # Test specific eltype conversion
        cpu_f32 = cpu_device(Float32)
        y_f32 = cpu_f32(x_f64)
        @test eltype(y_f32) === Float32
        @test y_f32 ≈ Float32.(x_f64)

        # Test that integer arrays are not converted
        y_int = cpu_f32(x_int)
        @test eltype(y_int) === Int
        @test y_int == x_int

        # Test complex floating point conversion
        x_complex = ComplexF64[1.0 + 2.0im, 3.0 + 4.0im]
        y_complex = cpu_f32(x_complex)
        @test eltype(y_complex) === ComplexF32
        @test y_complex ≈ ComplexF32.(x_complex)
    end

    @testset "Device Constructor Backward Compatibility" begin
        # Test that old constructors still work
        cpu_old = CPUDevice()
        @test cpu_old isa CPUDevice{Missing}

        cuda_old = CUDADevice(nothing)
        @test cuda_old isa CUDADevice{Nothing,Missing}

        amdgpu_old = AMDGPUDevice(nothing)
        @test amdgpu_old isa AMDGPUDevice{Nothing,Missing}

        metal_old = MetalDevice()
        @test metal_old isa MetalDevice{Missing}

        oneapi_old = oneAPIDevice()
        @test oneapi_old isa oneAPIDevice{Missing}

        opencl_old = OpenCLDevice()
        @test opencl_old isa OpenCLDevice{Missing}

        reactant_old = ReactantDevice()
        @test reactant_old isa ReactantDevice{Missing,Missing,Missing,Missing}
    end
end
