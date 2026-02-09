using Pkg, LuxLib, Test, ParallelTestRunner, LuxTestUtils

parsed_args = parse_args(
    @isdefined(TEST_ARGS) ? TEST_ARGS : ARGS;
    custom=["BACKEND_GROUP", "BLAS_BACKEND", "LOOP_VECTORIZATION"],
)

const BACKEND_GROUP = lowercase(
    something(get(parsed_args.custom, "BACKEND_GROUP", nothing), "all")
)

const BLAS_BACKEND = lowercase(
    something(get(parsed_args.custom, "BLAS_BACKEND", nothing), "default")
)

@assert BLAS_BACKEND in ("default", "appleaccelerate", "blis", "mkl")
@info "Running tests with BLAS backend: $(BLAS_BACKEND)"

const LOOP_VECTORIZATION = parse(
    Bool,
    lowercase(something(get(parsed_args.custom, "LOOP_VECTORIZATION", nothing), "true")),
)

testsuite = find_tests(@__DIR__)
delete!(testsuite, "shared_testsetup")

total_jobs = min(
    something(parsed_args.jobs, ParallelTestRunner.default_njobs()), length(keys(testsuite))
)

withenv(
    "XLA_REACTANT_GPU_MEM_FRACTION" => 1 / (total_jobs + 0.1),
    "XLA_REACTANT_GPU_PREALLOCATE" => false,
    "JULIA_CUDA_HARD_MEMORY_LIMIT" => "$(100 / (total_jobs + 0.1))%",
    "BACKEND_GROUP" => BACKEND_GROUP,
    "LOAD_LOOPVECTORIZATION" => LOOP_VECTORIZATION,
    "BLAS_BACKEND" => BLAS_BACKEND,
) do
    runtests(LuxLib, parsed_args; testsuite)
end
