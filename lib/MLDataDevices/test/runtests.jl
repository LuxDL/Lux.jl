using Pkg, MLDataDevices, Test, ParallelTestRunner, LuxTestUtils

parsed_args = parse_args(@isdefined(TEST_ARGS) ? TEST_ARGS : ARGS; custom=["BACKEND_GROUP"])

const BACKEND_GROUP = lowercase(
    something(get(parsed_args.custom, "BACKEND_GROUP", nothing), "all")
)

testsuite = find_tests(@__DIR__)

# Filter testsuite based on BACKEND_GROUP
backend_test_files = Set([
    "reactant_tests",
    "cuda_tests",
    "amdgpu_tests",
    "metal_tests",
    "oneapi_tests",
    "opencl_tests",
    "openclcpu_tests",
])

for file in keys(testsuite)
    if file ∈ backend_test_files
        # Remove backend-specific tests unless that backend is being tested
        if BACKEND_GROUP == "all"
            # Keep all
        elseif BACKEND_GROUP == "cpu"
            delete!(testsuite, file)
        elseif file == "$(BACKEND_GROUP)_tests"
            # Keep this backend's tests
        else
            delete!(testsuite, file)
        end
    end
end

delete!(testsuite, "common")
delete!(testsuite, "iterator_tests")
delete!(testsuite, "misc_tests")

total_jobs = min(
    something(parsed_args.jobs, ParallelTestRunner.default_njobs()), length(keys(testsuite))
)

additional_testsuite = Dict{String,Expr}()
for file in ("iterator_tests.jl", "misc_tests.jl")
    testfile = joinpath(@__DIR__, file)
    additional_testsuite[file] = :(include($testfile))
end

withenv(
    "XLA_REACTANT_GPU_MEM_FRACTION" => 1 / (total_jobs + 0.1),
    "XLA_REACTANT_GPU_PREALLOCATE" => false,
    "BACKEND_GROUP" => BACKEND_GROUP,
) do
    @testset "MLDataDevices" begin
        runtests(MLDataDevices, parsed_args; testsuite)

        runtests(MLDataDevices, parsed_args; testsuite=additional_testsuite)
    end
end
