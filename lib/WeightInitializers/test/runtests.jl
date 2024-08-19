using Pkg, ReTestItems, WeightInitializers
using InteractiveUtils, Hwloc

@info sprint(versioninfo)

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "All"))

const EXTRA_PKGS = String[]

(BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") && push!(EXTRA_PKGS, "CUDA")
(BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") && push!(EXTRA_PKGS, "AMDGPU")
(BACKEND_GROUP == "all" || BACKEND_GROUP == "metal") && push!(EXTRA_PKGS, "Metal")
(BACKEND_GROUP == "all" || BACKEND_GROUP == "oneapi") && push!(EXTRA_PKGS, "oneAPI")

if !isempty(EXTRA_PKGS)
    @info "Installing Extra Packages for testing" EXTRA_PKGS=EXTRA_PKGS
    Pkg.add(EXTRA_PKGS)
    Pkg.update()
    Base.retry_load_extensions()
    Pkg.instantiate()
end

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Hwloc.num_physical_cores(), 4))))
const RETESTITEMS_NWORKER_THREADS = parse(Int,
    get(ENV, "RETESTITEMS_NWORKER_THREADS",
        string(max(Hwloc.num_virtual_cores() ÷ RETESTITEMS_NWORKERS, 1))))

ReTestItems.runtests(WeightInitializers; nworkers=RETESTITEMS_NWORKERS,
    nworker_threads=RETESTITEMS_NWORKER_THREADS)
