using ReTestItems, Pkg, LuxTestUtils, Preferences

Preferences.set_preferences!("LuxLib", "instability_check" => "error")

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))
const EXTRA_PKGS = String[]

(BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") && push!(EXTRA_PKGS, "LuxCUDA")
(BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") && push!(EXTRA_PKGS, "AMDGPU")

if !isempty(EXTRA_PKGS)
    @info "Installing Extra Packages for testing" EXTRA_PKGS=EXTRA_PKGS
    Pkg.add(EXTRA_PKGS)
    Pkg.update()
    Base.retry_load_extensions()
    Pkg.instantiate()
end

const LUXLIB_TEST_GROUP = get(ENV, "LUXLIB_TEST_GROUP", "all")
@info "Running tests for group: $LUXLIB_TEST_GROUP"
const RETESTITEMS_NWORKERS = parse(Int, get(ENV, "RETESTITEMS_NWORKERS", "0"))

ReTestItems.runtests(
    @__DIR__; tags=(LUXLIB_TEST_GROUP == "all" ? nothing : [Symbol(LUXLIB_TEST_GROUP)]),
    nworkers=ifelse(BACKEND_GROUP ∈ ("cpu", "all"), 1, RETESTITEMS_NWORKERS))
