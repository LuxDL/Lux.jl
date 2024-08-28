import Pkg
using SafeTestsets, Test

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "none"))

const EXTRA_PKGS = String[]

(BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") && push!(EXTRA_PKGS, "LuxCUDA")
(BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") && push!(EXTRA_PKGS, "AMDGPU")
(BACKEND_GROUP == "all" || BACKEND_GROUP == "oneapi") && push!(EXTRA_PKGS, "oneAPI")
(BACKEND_GROUP == "all" || BACKEND_GROUP == "metal") && push!(EXTRA_PKGS, "Metal")

if !isempty(EXTRA_PKGS)
    @info "Installing Extra Packages for testing" EXTRA_PKGS=EXTRA_PKGS
    Pkg.add(EXTRA_PKGS)
    Pkg.update()
    Base.retry_load_extensions()
    Pkg.instantiate()
end

@testset "MLDataDevices Tests" begin
    file_names = BACKEND_GROUP == "all" ?
                 ["cuda_tests.jl", "amdgpu_tests.jl", "metal_tests.jl", "oneapi_tests.jl"] :
                 (BACKEND_GROUP == "cpu" ? [] : [BACKEND_GROUP * "_tests.jl"])
    @testset "$(file_name)" for file_name in file_names
        run(`$(Base.julia_cmd()) --color=yes --project=$(dirname(Pkg.project().path))
             --startup-file=no --code-coverage=user $(@__DIR__)/$file_name`)
        Test.@test true
    end

    @safetestset "Iterator Tests" include("iterator_tests.jl")
    @safetestset "Misc Tests" include("misc_tests.jl")
    @safetestset "QA Tests" include("qa_tests.jl")
end
