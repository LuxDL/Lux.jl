using Pkg: Pkg, PackageSpec
using Test
using LuxTestUtils

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

const EXTRA_PKGS = LuxTestUtils.packages_to_install(BACKEND_GROUP)

if !isempty(EXTRA_PKGS)
    @info "Installing Extra Packages for testing" EXTRA_PKGS
    isempty(EXTRA_PKGS) || Pkg.add(EXTRA_PKGS)
    Base.retry_load_extensions()
    Pkg.instantiate()
end

@testset "MLDataDevices Tests" begin
    all_files = map(
        Base.Fix2(*, "_tests.jl"),
        ["reactant", "cuda", "amdgpu", "metal", "oneapi", "opencl"],
    )
    file_names = if BACKEND_GROUP == "all"
        all_files
    elseif BACKEND_GROUP ∈ ("cpu", "none")
        []
    elseif BACKEND_GROUP == "opencl"
        ["opencl_tests.jl", "openclcpu_tests.jl"]
    else
        [BACKEND_GROUP * "_tests.jl"]
    end

    append!(file_names, ["iterator_tests.jl", "misc_tests.jl", "qa_tests.jl"])

    @testset "$(file_name)" for file_name in file_names
        @info "Running $(file_name)"
        withenv("BACKEND_GROUP" => BACKEND_GROUP) do
            run(`$(Base.julia_cmd()) --color=yes --project=$(dirname(Pkg.project().path))
                --startup-file=no --code-coverage=user $(@__DIR__)/$file_name`)
            Test.@test true
        end
    end
end
