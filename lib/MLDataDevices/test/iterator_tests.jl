using MLDataDevices, MLUtils, Test, LuxTestUtils

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "none"))

if LuxTestUtils.test_cuda(BACKEND_GROUP)
    using CUDA, LuxCUDA
end

if BACKEND_GROUP != "cuda"
    # For CUDA only tests avoid loading Reactant since they are often incompatible
    using Reactant
end

if LuxTestUtils.test_amdgpu(BACKEND_GROUP)
    using AMDGPU
end

if LuxTestUtils.test_metal(BACKEND_GROUP)
    using Metal
end

if LuxTestUtils.test_oneapi(BACKEND_GROUP)
    using oneAPI
end

if LuxTestUtils.test_opencl(BACKEND_GROUP)
    using OpenCL, pocl_jll
end

DEVICES = [
    CPUDevice,
    CUDADevice,
    AMDGPUDevice,
    MetalDevice,
    oneAPIDevice,
    OpenCLDevice,
    ReactantDevice,
]

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

struct FailingIterator
    fail_at::Int
    total::Int
end

Base.length(iter::FailingIterator) = iter.total

function Base.iterate(iter::FailingIterator, state::Int=1)
    state > iter.total && return nothing
    state == iter.fail_at && error("boom at step $(state)")
    return fill(Float64(state), 4, 3), state + 1
end

@testset "Device Iterator: $(dev_type)" for dev_type in DEVICES
    dev = dev_type()

    if !MLDataDevices.functional(dev)
        @warn "Device $(dev_type) is not functional. Skipping tests."
        continue
    end

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

    @testset "Prefetch Options" begin
        datalist = [rand(4, 5) for _ in 1:4]
        @test DeviceIterator(dev, datalist).prefetch == 1
        @test_throws ArgumentError DeviceIterator(dev, datalist; prefetch=-1)

        @testset "prefetch=$prefetch" for prefetch in (0, 1, 2)
            @test iterate(DeviceIterator(dev, Matrix{Float64}[]; prefetch)) === nothing
            @test isempty(collect(DeviceIterator(dev, Matrix{Float64}[]; prefetch)))

            single = [rand(4, 3)]
            single_seen = Any[]
            for d in DeviceIterator(dev, single; prefetch)
                @test get_device_type(d) == dev_type
                @test Array(d) ≈ Array(dev(single[1]))
                push!(single_seen, d)
            end
            @test length(single_seen) == 1
            if dev_type !== CPUDevice && dev_type !== ReactantDevice
                @test !freed_if_can_be_freed(single_seen[1])
            end

            multi = [rand(4, 5), rand(4, 5), rand(4, 2)]
            prev_b = nothing
            seen_count = 0
            for (i, d) in enumerate(DeviceIterator(dev, multi; prefetch))
                prev_b === nothing || @test freed_if_can_be_freed(prev_b)
                prev_b = d
                seen_count += 1
                @test get_device_type(d) == dev_type
                @test Array(d) ≈ Array(dev(multi[i]))
            end
            @test seen_count == length(multi)
            if dev_type !== CPUDevice && dev_type !== ReactantDevice
                @test !freed_if_can_be_freed(prev_b)
            end

            first_b = first(DeviceIterator(dev, multi; prefetch))
            @test Array(first_b) ≈ Array(dev(multi[1]))
            steps = 0
            for d in DeviceIterator(dev, multi; prefetch)
                steps += 1
                @test Array(d) ≈ Array(dev(multi[steps]))
                steps == 2 && break
            end
            @test steps == 2

            for fail_at in (1, 2, 3)
                @test_throws ErrorException collect(
                    DeviceIterator(dev, FailingIterator(fail_at, 4); prefetch)
                )
            end
        end
    end

    @testset "Element Types, Containers & Aliasing: prefetch=$prefetch" for prefetch in
                                                                            (0, 1)
        a = rand(3, 4)
        samples = Any[
            rand(Float64, 3, 4),
            rand(Float16, 3, 4),
            rand(1:10, 3, 4),
            rand(UInt8, 3, 4),
            BitArray(rand(Bool, 3, 4)),
            (; x=rand(2, 3), y=rand(1:5, 3)),
            (a, a),
        ]
        for s in samples
            batches = [s, deepcopy(s)]
            ref = dev(s)
            for d in DeviceIterator(dev, batches; prefetch)
                if s isa NamedTuple
                    @test get_device_type(d.x) == dev_type
                    @test get_device_type(d.y) == dev_type
                    @test eltype(d.x) == eltype(ref.x)
                    @test eltype(d.y) == eltype(ref.y)
                    @test Array(d.x) == Array(ref.x)
                    @test Array(d.y) == Array(ref.y)
                elseif s isa Tuple
                    @test get_device_type(d[1]) == dev_type
                    @test Array(d[1]) == Array(ref[1])
                    if dev_type !== ReactantDevice
                        @test d[1] === d[2]
                    end
                else
                    @test get_device_type(d) == dev_type
                    @test eltype(d) == eltype(ref)
                    @test Array(d) == Array(ref)
                end
            end
        end

        if dev_type === CUDADevice
            dev_preserve = CUDADevice{Nothing,Nothing}(nothing)
            f64_batches = [rand(Float64, 3, 4), rand(Float64, 3, 2)]
            for (i, d) in enumerate(DeviceIterator(dev_preserve, f64_batches; prefetch))
                @test eltype(d) == Float64
                @test Array(d) == f64_batches[i]
            end
        end
    end

    @testset "DataLoader: parallel=$parallel, buffer=$buffer" for parallel in
                                                                  (false, true, 2),
        buffer in (false, true)

        if dev_type === ReactantDevice && (parallel isa Integer || buffer)
            continue
        end

        @info "Testing DataLoader with parallel=$parallel, buffer=$buffer"
        partial = !buffer
        X = rand(Float64, 3, 33)
        post = dev(DataLoader(X; batchsize=13, partial, shuffle=false, parallel, buffer))
        if dev_type === ReactantDevice
            pre = post # XXX: deadlocks and other shenanigans
        else
            pre = DataLoader(dev(X); batchsize=13, partial, shuffle=false, parallel, buffer)
        end

        for epoch in 1:2
            prev_pre, prev_post = nothing, nothing
            for (i, (p, q)) in enumerate(zip(pre, post))
                @test get_device_type(p) == dev_type
                @test get_device_type(q) == dev_type
                # Ordering is not guaranteed in parallel
                if parallel == false
                    @test p ≈ q
                    if dev_type === CUDADevice && buffer
                        lo = (i - 1) * 13 + 1
                        hi = min(size(X, 2), i * 13)
                        @test Array(q) ≈ Float32.(X[:, lo:hi])
                    end
                end

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
        post = dev(
            DataLoader((; x=X, y=Y); batchsize=13, partial, shuffle=false, parallel, buffer)
        )
        if dev_type === ReactantDevice
            pre = post # XXX: deadlocks and other shenanigans
        else
            pre = DataLoader(
                (; x=dev(X), y=dev(Y));
                batchsize=13,
                partial,
                shuffle=false,
                parallel,
                buffer,
            )
        end

        for epoch in 1:2
            prev_pre, prev_post = nothing, nothing
            for (p, q) in zip(pre, post)
                @test get_device_type(p.x) == dev_type
                @test get_device_type(p.y) == dev_type
                @test get_device_type(q.x) == dev_type
                @test get_device_type(q.y) == dev_type
                # Ordering is not guaranteed in parallel
                if parallel == false
                    @test p.x ≈ q.x
                    @test p.y ≈ q.y
                end

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

if LuxTestUtils.test_cuda(BACKEND_GROUP) && MLDataDevices.functional(CUDADevice)
    @testset "CUDA Prefetch CUBLAS & Bounded Memory" begin
        dev = CUDADevice()
        @testset "CUBLAS correctness: prefetch=$prefetch" for prefetch in (0, 1, 2)
            W = dev(rand(Float32, 16, 8))
            batches = [rand(Float64, 8, 4) for _ in 1:6]
            gpu_acc = zero(dev(zeros(Float32, 16, 4)))
            for b in DeviceIterator(dev, batches; prefetch)
                gpu_acc .+= W * b
            end
            cpu_acc = sum(Array(W) * Float32.(b) for b in batches)
            @test Array(gpu_acc) ≈ cpu_acc
            CUDA.unsafe_free!(W)
            CUDA.unsafe_free!(gpu_acc)
        end

        @testset "30-step async VRAM bound" begin
            batch_template = rand(Float32, 1024, 2048) # 8 MiB per batch
            batch_bytes = sizeof(batch_template)
            batches = [copy(batch_template) for _ in 1:30]

            # Warmup JIT, stream, and staging buffers before measuring pool growth
            for b in DeviceIterator(dev, batches[1:2]; prefetch=1)
                _ = b .* 2.0f0 .+ 1.0f0
            end
            CUDA.synchronize()
            GC.gc(false)
            CUDA.reclaim()

            mem_before = CUDA.used_memory()
            last_y = nothing
            for b in DeviceIterator(dev, batches; prefetch=1)
                last_y !== nothing && CUDA.unsafe_free!(last_y)
                last_y = b .* 2.0f0 .+ 1.0f0
            end
            CUDA.synchronize()
            mem_after = CUDA.used_memory()
            @test mem_after - mem_before <= 8 * batch_bytes
            last_y !== nothing && CUDA.unsafe_free!(last_y)
            CUDA.synchronize()
            CUDA.reclaim()
        end
    end
end
