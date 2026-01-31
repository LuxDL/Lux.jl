using Pkg, WeightInitializers, Test, ParallelTestRunner, LuxTestUtils

parsed_args = parse_args(@isdefined(TEST_ARGS) ? TEST_ARGS : ARGS; custom=["BACKEND_GROUP"])

const BACKEND_GROUP = lowercase(
    something(get(parsed_args.custom, "BACKEND_GROUP", nothing), "all")
)

const EXTRA_PKGS = LuxTestUtils.packages_to_install(BACKEND_GROUP)

if !isempty(EXTRA_PKGS)
    @info "Installing Extra Packages for testing" EXTRA_PKGS = EXTRA_PKGS
    Pkg.add(EXTRA_PKGS)
    Base.retry_load_extensions()
    Pkg.instantiate()
end

testsuite = find_tests(@__DIR__)
delete!(testsuite, "common")

total_jobs = min(
    something(parsed_args.jobs, ParallelTestRunner.default_njobs()), length(keys(testsuite))
)

withenv(
    "XLA_REACTANT_GPU_MEM_FRACTION" => 1 / (total_jobs + 0.1),
    "XLA_REACTANT_GPU_PREALLOCATE" => false,
    "BACKEND_GROUP" => BACKEND_GROUP,
) do
    runtests(WeightInitializers, parsed_args; testsuite)
end
