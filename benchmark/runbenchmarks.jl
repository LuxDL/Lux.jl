using PkgJogger, InteractiveUtils, LinearAlgebra, Lux, ThreadPinning, Printf
using LuxCUDA

BLAS.set_num_threads(Threads.nthreads() ÷ 2)
ThreadPinning.pinthreads(:cores)
ThreadPinning.threadinfo()

@info sprint(versioninfo)
if CUDA.functional()
    @info sprint(CUDA.versioninfo)
end
@info "BLAS threads: $(BLAS.get_num_threads())"

@jog Lux

result = JogLux.benchmark(; verbose=true)
filename = JogLux.save_benchmarks(result)

@info "Benchmark results saved to $filename"
