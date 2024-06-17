using ReTestItems, Pkg, Tests

const LUX_TEST_GROUP = lowercase(get(ENV, "LUX_TEST_GROUP", "all"))
@info "Running tests for group: $LUX_TEST_GROUP"

if LUX_TEST_GROUP == "all"
    ReTestItems.runtests(@__DIR__)
else
    tag = Symbol(LUX_TEST_GROUP)
    ReTestItems.runtests(@__DIR__; tags=[tag])
end

# Distributed Tests
if LUX_TEST_GROUP == "all" || LUX_TEST_GROUP == "distributed"
    Pkg.add(["MPI", "NCCL"])
    using MPI

    nprocs_str = get(ENV, "JULIA_MPI_TEST_NPROCS", "")
    nprocs = nprocs_str == "" ? clamp(Sys.CPU_THREADS, 2, 4) : parse(Int, nprocs_str)
    testdir = @__DIR__
    isdistributedtest(f) = endswith(f, "_distributedtest.jl")
    distributedtestfiles = String[]
    for (root, dirs, files) in walkdir(testdir)
        for file in files
            if isdistributedtest(file)
                push!(distributedtestfiles, joinpath(root, file))
            end
        end
    end

    @info "Running Distributed Tests with $nprocs processes"

    cur_proj = dirname(Pkg.project().path)

    include("setup_modes.jl")

    @testset "MODE: $(mode)" for (mode, aType, dev, ongpu) in MODES
        if mode == "AMDGPU"
            # AMDGPU needs to cause a deadlock, needs to be investigated
            @test_broken 1 == 2
            continue
        end
        backends = mode == "CUDA" ? ("mpi", "nccl") : ("mpi",)
        for backend_type in backends
            np = backend_type == "nccl" ? min(nprocs, length(CUDA.devices())) : nprocs
            @testset "Backend: $(backend_type)" begin
                @testset "$(basename(file))" for file in distributedtestfiles
                    @info "Running $file with $backend_type backend on $mode device"
                    run(`$(MPI.mpiexec()) -n $(np) $(Base.julia_cmd()) --color=yes \
                        --code-coverage=user --project=$(cur_proj) --startup-file=no $(file) \
                        $(mode) $(backend_type)`)
                    Test.@test true
                end
            end
        end
    end
end
