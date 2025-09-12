using MPI, Pkg, Test

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

cur_proj = dirname(Pkg.project().path)

@info "Running Distributed Tests with $nprocs processes"

include("../setup_modes.jl")

@testset "distributed tests: $(mode)" for (mode, aType, dev, ongpu) in MODES
    backends = mode == "cuda" ? ("mpi", "nccl") : ("mpi",)
    @testset "Backend: $(backend_type)" for backend_type in backends
        np = backend_type == "nccl" ? min(nprocs, length(CUDA.devices())) : nprocs
        @testset "$(basename(file))" for file in distributedtestfiles
            @info "Running $file with $backend_type backend on $mode device"
            try
                run(`$(MPI.mpiexec()) -n $(np) $(Base.julia_cmd()) --color=yes \
                     --code-coverage=user --project=$(cur_proj) --startup-file=no \
                     $(file) $(mode) $(backend_type)`)
                @test true
            catch
                @test false
            end
        end
    end
end
