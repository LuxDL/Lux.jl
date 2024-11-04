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

if BACKEND_GROUP == "xla" || BACKEND_GROUP == "all"
    using Reactant
    if "gpu" in keys(Reactant.XLA.backends)
        Reactant.set_default_backend("gpu")
    end
end

DEVICES = [CPUDevice, CUDADevice, AMDGPUDevice, MetalDevice, oneAPIDevice, ReactantDevice]

freed_if_can_be_freed(x) = freed_if_can_be_freed(get_device_type(x), x)
freed_if_can_be_freed(::Type{CPUDevice}, x) = true
freed_if_can_be_freed(::Type{ReactantDevice}, x) = true
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

    @testset "DataLoader: parallel=$parallel" for parallel in (true, false)
        @info "Testing DataLoader with parallel=$parallel"
        X = rand(Float64, 3, 33)
        post = DataLoader(X; batchsize=13, shuffle=false, parallel) |> dev
        if dev_type === ReactantDevice
            pre = post # XXX: deadlocks and other shenanigans
        else
            pre = DataLoader(dev(X); batchsize=13, shuffle=false, parallel)
        end

        for epoch in 1:2
            prev_pre, prev_post = nothing, nothing
            for (p, q) in zip(pre, post)
                @test get_device_type(p) == dev_type
                @test get_device_type(q) == dev_type
                # Ordering is not guaranteed in parallel
                !parallel && @test p ≈ q

                if dev_type === CPUDevice || dev_type === ReactantDevice
                    continue
                end

                prev_pre === nothing || @test !freed_if_can_be_freed(prev_pre)
                prev_pre = p

                prev_post === nothing || @test freed_if_can_be_freed(prev_post)
                prev_post = q
            end
        end

        Y = rand(Float64, 1, 33)
        post = DataLoader((; x=X, y=Y); batchsize=13, shuffle=false, parallel) |> dev
        if dev_type === ReactantDevice
            pre = post # XXX: deadlocks and other shenanigans
        else
            pre = DataLoader((; x=dev(X), y=dev(Y)); batchsize=13, shuffle=false, parallel)
        end

        for epoch in 1:2
            prev_pre, prev_post = nothing, nothing
            for (p, q) in zip(pre, post)
                @test get_device_type(p.x) == dev_type
                @test get_device_type(p.y) == dev_type
                @test get_device_type(q.x) == dev_type
                @test get_device_type(q.y) == dev_type
                # Ordering is not guaranteed in parallel
                !parallel && @test p.x ≈ q.x
                !parallel && @test p.y ≈ q.y

                if dev_type === CPUDevice || dev_type === ReactantDevice
                    continue
                end

                if prev_pre !== nothing
                    @test !freed_if_can_be_freed(prev_pre.x)
                    @test !freed_if_can_be_freed(prev_pre.y)
                end
                prev_pre = p

                if prev_post !== nothing
                    @test freed_if_can_be_freed(prev_post.x)
                    @test freed_if_can_be_freed(prev_post.y)
                end
                prev_post = q
            end
        end
    end
end
