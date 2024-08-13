using LuxLib
using Pkg
using BenchmarkTools
using InteractiveUtils
using LinearAlgebra

const SUITE = BenchmarkGroup()
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5

# To run benchmarks on a specific GPU backend, add AMDGPU / CUDA / Metal / oneAPI
# to benchmarks/Project.toml and change BENCHMARK_GROUP to the backend name
const BENCHMARK_GROUP = get(ENV, "BENCHMARK_GROUP", "CPU")
const BENCHMARK_CPU_THREADS = Threads.nthreads()

# Number of CPU threads to benchmarks on
if BENCHMARK_CPU_THREADS > Threads.nthreads()
    @error "More CPU threads were requested than are available. Change the \
            JULIA_NUM_THREADS environment variable or pass \
            --threads=$(BENCHMARK_CPU_THREADS) as a julia argument"
end

LinearAlgebra.BLAS.set_num_threads(BENCHMARK_CPU_THREADS)

if BENCHMARK_GROUP == "AMDGPU"
    using AMDGPU # ] add AMDGPU to benchmarks/Project.toml
    @info "Running AMDGPU benchmarks" maxlog=1
    AMDGPU.versioninfo()
elseif BENCHMARK_GROUP == "CUDA"
    using LuxCUDA # ] add LuxCUDA to benchmarks/Project.toml
    @info "Running CUDA benchmarks" maxlog=1
    CUDA.versioninfo()
elseif BENCHMARK_GROUP == "Metal"
    using Metal # ] add Metal to benchmarks/Project.toml
    @info "Running Metal benchmarks" maxlog=1
    Metal.versioninfo()
elseif BENCHMARK_GROUP == "oneAPI"
    using oneAPI # ] add oneAPI to benchmarks/Project.toml
    @info "Running oneAPI benchmarks" maxlog=1
    oneAPI.versioninfo()
else
    @info "Running CPU benchmarks with $(BENCHMARK_CPU_THREADS) thread(s)" maxlog=1
    @info sprint(InteractiveUtils.versioninfo)
end

include("setup.jl")
setup_benchmarks!(SUITE, BENCHMARK_GROUP, BENCHMARK_CPU_THREADS)

results = BenchmarkTools.run(SUITE; verbose=true)

filepath = joinpath(dirname(@__FILE__), "results")
mkpath(filepath)
filename = BENCHMARK_GROUP == "CPU" ?
           string("CPUbenchmarks", BENCHMARK_CPU_THREADS, "threads.json") :
           string(BENCHMARK_GROUP, "benchmarks.json")
BenchmarkTools.save(joinpath(filepath, filename), median(results))

@info "Saved results to $(joinpath(filepath, filename))"
