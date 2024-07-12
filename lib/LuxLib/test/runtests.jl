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

if LUXLIB_TEST_GROUP == "all"
    ReTestItems.runtests("common_ops")
    ReTestItems.runtests("others")
    ReTestItems.runtests("normalization"; nworkers=0)
else
    ReTestItems.runtests("common_ops"; tags=[Symbol(LUXLIB_TEST_GROUP)])
    ReTestItems.runtests("others"; tags=[Symbol(LUXLIB_TEST_GROUP)])
    if LUXLIB_TEST_GROUP == "normalization"
        ReTestItems.runtests("normalization"; tags=[Symbol(LUXLIB_TEST_GROUP)], nworkers=0)
    end
end
