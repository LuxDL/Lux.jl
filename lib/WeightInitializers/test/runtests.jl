using Pkg, ReTestItems

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

using WeightInitializers

ReTestItems.runtests(WeightInitializers)
