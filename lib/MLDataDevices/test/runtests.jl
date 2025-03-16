using Pkg: Pkg, PackageSpec
using SafeTestsets, Test

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "none"))

const EXTRA_PKGS = PackageSpec[]
const EXTRA_DEV_PKGS = PackageSpec[]

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
(BACKEND_GROUP == "all" || BACKEND_GROUP == "xla") &&
    push!(EXTRA_PKGS, PackageSpec(; name="Reactant"))

if !isempty(EXTRA_PKGS) || !isempty(EXTRA_DEV_PKGS)
    @info "Installing Extra Packages for testing" EXTRA_PKGS EXTRA_DEV_PKGS
    isempty(EXTRA_PKGS) || Pkg.add(EXTRA_PKGS)
    isempty(EXTRA_DEV_PKGS) || Pkg.develop(EXTRA_DEV_PKGS)
    Base.retry_load_extensions()
    Pkg.instantiate()
end

@testset "MLDataDevices Tests" begin
    all_files = [
        "cuda_tests.jl",
        "amdgpu_tests.jl",
        "metal_tests.jl",
        "oneapi_tests.jl",
        "xla_tests.jl",
    ]
    file_names = if BACKEND_GROUP == "all"
        all_files
    elseif BACKEND_GROUP ∈ ("cpu", "none")
        []
    else
        [BACKEND_GROUP * "_tests.jl"]
    end
    @testset "$(file_name)" for file_name in file_names
        run(`$(Base.julia_cmd()) --color=yes --project=$(dirname(Pkg.project().path))
             --startup-file=no --code-coverage=user $(@__DIR__)/$file_name`)
        Test.@test true
    end

    @safetestset "Iterator Tests" include("iterator_tests.jl")
    @safetestset "Misc Tests" include("misc_tests.jl")
    @safetestset "QA Tests" include("qa_tests.jl")
end
