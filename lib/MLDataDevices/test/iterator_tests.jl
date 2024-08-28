using MLDataDevices, MLUtils

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "none"))

if BACKEND_GROUP == "cuda" || BACKEND_GROUP == "all"
    using LuxCUDA
end

if BACKEND_GROUP == "amdgpu" || BACKEND_GROUP == "all"
    using AMDGPU
end

if BACKEND_GROUP == "metal" || BACKEND_GROUP == "all"
    using Metal
end

if BACKEND_GROUP == "oneapi" || BACKEND_GROUP == "all"
    using oneAPI
end

DEVICES = [CPUDevice, CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice]

freed_if_can_be_freed(x) = freed_if_can_be_freed(get_device_type(x), x)
freed_if_can_be_freed(::Type{CPUDevice}, x) = true
function freed_if_can_be_freed(::Type, x)
    try
        Array(x)
        return false
    catch err
        err isa ArgumentError && return true
        rethrow()
    end
end

@testset "Device Iterator: $(dev_type)" for dev_type in DEVICES
    dev = dev_type()

    !MLDataDevices.functional(dev) && continue

    @info "Testing Device Iterator for $(dev)..."

    @testset "Basic Device Iterator" begin
        datalist = [rand(10) for _ in 1:10]

        prev_batch = nothing
        for data in DeviceIterator(dev, datalist)
            prev_batch === nothing || @test freed_if_can_be_freed(prev_batch)
            prev_batch = data
            @test size(data) == (10,)
            @test get_device_type(data) == dev_type
        end
    end
end
