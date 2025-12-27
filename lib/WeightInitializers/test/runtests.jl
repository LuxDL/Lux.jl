using Pkg, ReTestItems, WeightInitializers
using InteractiveUtils, CPUSummary, LuxTestUtils

@info sprint(versioninfo)

function parse_test_args()
    test_args_from_env = @isdefined(TEST_ARGS) ? TEST_ARGS : ARGS
    test_args = Dict{String,String}()
    for arg in test_args_from_env
        if contains(arg, "=")
            key, value = split(arg, "="; limit=2)
            test_args[key] = value
        end
    end
    @info "Parsed test args" test_args
    return test_args
end

const PARSED_TEST_ARGS = parse_test_args()

const BACKEND_GROUP = lowercase(get(PARSED_TEST_ARGS, "BACKEND_GROUP", "All"))

const EXTRA_PKGS = LuxTestUtils.packages_to_install(BACKEND_GROUP)

if !isempty(EXTRA_PKGS)
    @info "Installing Extra Packages for testing" EXTRA_PKGS = EXTRA_PKGS
    Pkg.add(EXTRA_PKGS)
    Base.retry_load_extensions()
    Pkg.instantiate()
end

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Int(CPUSummary.num_cores()), 4)))
)
const RETESTITEMS_NWORKER_THREADS = parse(
    Int,
    get(
        ENV,
        "RETESTITEMS_NWORKER_THREADS",
        string(max(Int(CPUSummary.sys_threads()) ÷ RETESTITEMS_NWORKERS, 1)),
    ),
)

withenv("BACKEND_GROUP" => BACKEND_GROUP) do
    ReTestItems.runtests(
        WeightInitializers;
        nworkers=RETESTITEMS_NWORKERS,
        nworker_threads=RETESTITEMS_NWORKER_THREADS,
    )
end
