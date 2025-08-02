using Pkg: Pkg, PackageSpec
using Test

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

const BACKEND_GROUP = lowercase(get(PARSED_TEST_ARGS, "BACKEND_GROUP", "none"))

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
(BACKEND_GROUP == "all" || BACKEND_GROUP == "reactant") &&
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
        "reactant_tests.jl",
    ]
    file_names = if BACKEND_GROUP == "all"
        all_files
    elseif BACKEND_GROUP ∈ ("cpu", "none")
        []
    else
        [BACKEND_GROUP * "_tests.jl"]
    end

    append!(file_names, ["iterator_tests.jl", "misc_tests.jl", "qa_tests.jl"])

    @testset "$(file_name)" for file_name in file_names
        withenv("BACKEND_GROUP" => BACKEND_GROUP) do
            run(`$(Base.julia_cmd()) --color=yes --project=$(dirname(Pkg.project().path))
                --startup-file=no --code-coverage=user $(@__DIR__)/$file_name`)
            Test.@test true
        end
    end
end
