# using BenchmarkTools

# const GPU_BACKENDS = ["AMDGPU", "CUDA", "Metal", "oneAPI"]
# const NUM_CPU_THREADS = [1, 2, 4, 8]

# #Start with CPU benchmarks for 1 thread and add other results
# const CPU_results_1thread_filepath = joinpath(
#     dirname(@__FILE__), "results", "CPUbenchmarks1threads.json")
# @assert(ispath(CPU_results_1thread_filepath))
# const RESULTS = BenchmarkTools.load(CPU_results_1thread_filepath)[1]
# @assert RESULTS isa BenchmarkTools.BenchmarkGroup

# for n in NUM_CPU_THREADS
#     filename = string("CPUbenchmarks", n, "threads.json")
#     filepath = joinpath(dirname(@__FILE__), "results", filename)
#     if !ispath(filepath)
#         @warn "No file found at path: $(filepath)"
#     else
#         nthreads_results = BenchmarkTools.load(filepath)[1]
#         if nthreads_results isa BenchmarkTools.BenchmarkGroup
#             for benchmark in keys(RESULTS), sim_method in keys(RESULTS[benchmark])
#                 RESULTS[benchmark][sim_method]["CPU"][string(n, " ", "thread(s)")] = nthreads_results[benchmark][sim_method]["CPU"][string(
#                     n, " ", "thread(s)")]
#             end
#         else
#             @warn "Unexpected file format for file at path: $(filepath)"
#         end
#     end
# end

# for backend in GPU_BACKENDS
#     filename = string(backend, "benchmarks.json")
#     filepath = joinpath(dirname(@__FILE__), "results", filename)
#     if !ispath(filepath)
#         @warn "No file found at path: $(filepath)"
#     else
#         backend_results = BenchmarkTools.load(filepath)[1]
#         if backend_results isa BenchmarkTools.BenchmarkGroup
#             for benchmark in keys(RESULTS)
#                 for sim_method in keys(RESULTS[benchmark])
#                     RESULTS[benchmark][sim_method]["GPU"][backend] = backend_results[benchmark][sim_method]["GPU"][backend]
#                 end
#             end
#         else
#             @warn "Unexpected file format for file at path: $(filepath)"
#         end
#     end
# end

# BenchmarkTools.save(
#     joinpath(dirname(@__FILE__), "results", "combinedbenchmarks.json"), RESULTS)
