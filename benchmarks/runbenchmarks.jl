using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @btime, @benchmarkable
using InteractiveUtils: versioninfo
using LinearAlgebra: BLAS
using Statistics: median
using ThreadPinning: pinthreads

pinthreads(:cores)

BLAS.set_num_threads(max(BLAS.get_num_threads(), Threads.nthreads() ÷ 2))

@info sprint(versioninfo)
@info "BLAS threads: $(BLAS.get_num_threads())"

const SUITE = BenchmarkGroup()

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

if BENCHMARK_GROUP == "AMDGPU"
    using AMDGPU # ] add AMDGPU to benchmarks/Project.toml
    @info "Running AMDGPU benchmarks" maxlog=1
elseif BENCHMARK_GROUP == "CUDA"
    using CUDA # ] add CUDA to benchmarks/Project.toml
    @info "Running CUDA benchmarks" maxlog=1
else
    @info "Running CPU benchmarks with $(BENCHMARK_CPU_THREADS) thread(s)" maxlog=1
end

# Main benchmark files
include("setups/layers.jl")

include("setup.jl")
setup_benchmarks(SUITE, BENCHMARK_GROUP, BENCHMARK_CPU_THREADS)

BenchmarkTools.tune!(SUITE; verbose=true)
results = BenchmarkTools.run(SUITE; verbose=true)
filepath = joinpath(dirname(@__FILE__), "results")
mkpath(filepath)
filename = BENCHMARK_GROUP == "CPU" ?
           string("CPUbenchmarks", BENCHMARK_CPU_THREADS, "threads.json") :
           string(BENCHMARK_GROUP, "benchmarks.json")
BenchmarkTools.save(joinpath(filepath, filename), median(results))

display(median(results))

@info "Saved results to $(joinpath(filepath, filename))"
