using ReTestItems, Pkg, LuxTestUtils
using InteractiveUtils, CPUSummary

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

const BACKEND_GROUP = lowercase(get(PARSED_TEST_ARGS, "BACKEND_GROUP", "all"))
const EXTRA_PKGS = PackageSpec[]
const EXTRA_DEV_PKGS = PackageSpec[]

const LUXLIB_BLAS_BACKEND = lowercase(
    get(PARSED_TEST_ARGS, "LUXLIB_BLAS_BACKEND", "default")
)
@assert LUXLIB_BLAS_BACKEND in ("default", "appleaccelerate", "blis", "mkl")
@info "Running tests with BLAS backend: $(LUXLIB_BLAS_BACKEND)"

if (BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda")
    if isdir(joinpath(@__DIR__, "../../LuxCUDA"))
        @info "Using local LuxCUDA"
        push!(EXTRA_DEV_PKGS, PackageSpec(; path=joinpath(@__DIR__, "../../LuxCUDA")))
    else
        push!(EXTRA_PKGS, PackageSpec(; name="LuxCUDA"))
    end
end
(BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") &&
    push!(EXTRA_PKGS, PackageSpec(; name="AMDGPU"))
(BACKEND_GROUP == "all" || BACKEND_GROUP == "oneapi") &&
    push!(EXTRA_PKGS, PackageSpec(; name="oneAPI"))
(BACKEND_GROUP == "all" || BACKEND_GROUP == "metal") &&
    push!(EXTRA_PKGS, PackageSpec(; name="Metal"))

if !isempty(EXTRA_PKGS) || !isempty(EXTRA_DEV_PKGS)
    @info "Installing Extra Packages for testing" EXTRA_PKGS EXTRA_DEV_PKGS
    isempty(EXTRA_PKGS) || Pkg.add(EXTRA_PKGS)
    isempty(EXTRA_DEV_PKGS) || Pkg.develop(EXTRA_DEV_PKGS)
    Base.retry_load_extensions()
    Pkg.instantiate()
end

const LUXLIB_TEST_GROUP = get(PARSED_TEST_ARGS, "LUXLIB_TEST_GROUP", "all")
const LUXLIB_LOAD_LOOPVEC = get(PARSED_TEST_ARGS, "LUXLIB_LOAD_LOOPVEC", "true")

const RETESTITEMS_NWORKERS = parse(
    Int,
    get(
        ENV,
        "RETESTITEMS_NWORKERS",
        string(min(Int(CPUSummary.num_cores()), Sys.isapple() ? 2 : 4)),
    ),
)

const RETESTITEMS_NWORKER_THREADS = parse(
    Int,
    get(
        ENV,
        "RETESTITEMS_NWORKER_THREADS",
        string(max(Int(CPUSummary.sys_threads()) ÷ RETESTITEMS_NWORKERS, 1)),
    ),
)

@info "Running tests for group: $LUXLIB_TEST_GROUP with $RETESTITEMS_NWORKERS workers"

using LuxLib

withenv(
    "LUXLIB_LOAD_LOOPVEC" => LUXLIB_LOAD_LOOPVEC,
    "BACKEND_GROUP" => BACKEND_GROUP,
    "LUXLIB_BLAS_BACKEND" => LUXLIB_BLAS_BACKEND,
) do
    ReTestItems.runtests(
        LuxLib;
        tags=(LUXLIB_TEST_GROUP == "all" ? nothing : [Symbol(LUXLIB_TEST_GROUP)]),
        nworkers=RETESTITEMS_NWORKERS,
        nworker_threads=RETESTITEMS_NWORKER_THREADS,
        testitem_timeout=3600,
    )
end
