using ReTestItems, Pkg, Preferences, Test
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

const BACKEND_GROUP = lowercase(get(PARSED_TEST_ARGS, "BACKEND_GROUP", "all"))
const ALL_LUX_TEST_GROUPS = [
    "core_layers", "normalize_layers", "autodiff", "recurrent_layers", "misc", "reactant"
]

INPUT_TEST_GROUP = lowercase(get(PARSED_TEST_ARGS, "LUX_TEST_GROUP", "all"))
const LUX_TEST_GROUP = if startswith("!", INPUT_TEST_GROUP[1])
    exclude_group = lowercase.(split(INPUT_TEST_GROUP[2:end], ","))
    filter(x -> x ∉ exclude_group, ALL_LUX_TEST_GROUPS)
else
    [INPUT_TEST_GROUP]
end
@info "Running tests for group: $LUX_TEST_GROUP"

const EXTRA_PKGS = LuxTestUtils.packages_to_install(BACKEND_GROUP)

try
    if !isempty(EXTRA_PKGS)
        @info "Installing Extra Packages for testing" EXTRA_PKGS
        isempty(EXTRA_PKGS) || Pkg.add(EXTRA_PKGS)
        Base.retry_load_extensions()
        Pkg.instantiate()
        Pkg.precompile()
    end
catch err
    @error "Error occurred while installing extra packages" err
end

using Lux

# Eltype Matching Tests
if ("all" in LUX_TEST_GROUP || "misc" in LUX_TEST_GROUP)
    @testset "eltype_mismath_handling: $option" for option in
                                                    ("none", "warn", "convert", "error")
        set_preferences!(Lux, "eltype_mismatch_handling" => option; force=true)
        try
            withenv("BACKEND_GROUP" => BACKEND_GROUP) do
                run(
                    `$(Base.julia_cmd()) --color=yes --project=$(dirname(Pkg.project().path))
                    --startup-file=no --code-coverage=user $(@__DIR__)/eltype_matching.jl`,
                )
                @test true
            end
        catch
            @test false
        end
    end
    set_preferences!(Lux, "eltype_mismatch_handling" => "none"; force=true)
end

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

@testset "Lux.jl Tests" begin
    @testset "[$(tag)] [$(i)/$(length(LUX_TEST_GROUP))]" for (i, tag) in
                                                             enumerate(LUX_TEST_GROUP)
        nworkers = (tag == "reactant") ? 1 : RETESTITEMS_NWORKERS
        tag == "extras" && continue

        withenv(
            "BACKEND_GROUP" => BACKEND_GROUP, "LUX_CURRENT_TEST_GROUP" => string(tag)
        ) do
            ReTestItems.runtests(
                Lux;
                tags=(tag == "all" ? nothing : [Symbol(tag)]),
                testitem_timeout=2400,
                nworkers,
                nworker_threads=RETESTITEMS_NWORKER_THREADS,
            )
        end
    end
end

# Various Downstream Integration Tests
# We only run these on 1.11+ due to nicer handling of [sources]
if ("all" in LUX_TEST_GROUP || "extras" in LUX_TEST_GROUP) && VERSION ≥ v"1.11-"
    testdir = @__DIR__
    isintegrationtest(f) = endswith(f, "_integrationtest.jl")
    integrationtestfiles = String[]
    for (root, dirs, files) in walkdir(testdir)
        for file in files
            if isintegrationtest(file)
                push!(integrationtestfiles, joinpath(root, file))
            end
        end
    end

    test_groups = Dict{String,Vector{String}}()
    for file in integrationtestfiles
        dir = dirname(file)
        if !haskey(test_groups, dir)
            test_groups[dir] = String[file]
        else
            push!(test_groups[dir], file)
        end
    end

    @testset "Downstream Integration Tests" begin
        withenv("BACKEND_GROUP" => BACKEND_GROUP) do
            @testset "$(basename(dir))" for (dir, files) in test_groups
                run(`$(Base.julia_cmd()) --color=yes --project=$(dir) \
                     --startup-file=no -e \
                     'using Pkg; Pkg.update(); Pkg.precompile()'`)

                @testset "$(basename(file))" for file in files
                    try
                        run(`$(Base.julia_cmd()) --code-coverage=user --color=yes \
                             --project=$(dir) --startup-file=no $(file)`)
                        @test true
                    catch err
                        @error "Error while running $(file)" exception = err
                        @test false
                    end
                end
            end
        end
    end
end

# Distributed Tests
if ("all" in LUX_TEST_GROUP || "extras" in LUX_TEST_GROUP) && VERSION ≥ v"1.11-"
    @testset "Distributed Tests" begin
        distributed_proj = joinpath(@__DIR__, "distributed")
        try
            run(`$(Base.julia_cmd()) --color=yes --project=$(distributed_proj) \
                 --startup-file=no -e 'using Pkg; Pkg.update(); Pkg.precompile()'`)
        catch err
            @error "Error while running Pkg.update(). Continuing without it." exception =
                err
        end

        distributed_test_file = joinpath(distributed_proj, "distributed_test_runner.jl")
        withenv("BACKEND_GROUP" => BACKEND_GROUP) do
            try
                run(`$(Base.julia_cmd()) --color=yes --code-coverage=user \
                    --project=$(distributed_proj) --startup-file=no \
                    $(distributed_test_file)`)
                @test true
            catch err
                @error "Error while running $(distributed_test_file)" exception = err
                @test false
            end
        end
    end
end

# Set preferences tests
if ("all" in LUX_TEST_GROUP || "others" in LUX_TEST_GROUP)
    @testset "DispatchDoctor Preferences" begin
        @testset "set_dispatch_doctor_preferences!" begin
            @test_throws ArgumentError Lux.set_dispatch_doctor_preferences!("invalid")
            @test_throws ArgumentError Lux.set_dispatch_doctor_preferences!(;
                luxcore="invalid"
            )

            Lux.set_dispatch_doctor_preferences!("disable")
            @test Preferences.load_preference(LuxCore, "instability_check") == "disable"
            @test Preferences.load_preference(LuxLib, "instability_check") == "disable"

            Lux.set_dispatch_doctor_preferences!(; luxcore="warn", luxlib="error")
            @test Preferences.load_preference(LuxCore, "instability_check") == "warn"
            @test Preferences.load_preference(LuxLib, "instability_check") == "error"

            Lux.set_dispatch_doctor_preferences!(; luxcore="error")
            @test Preferences.load_preference(LuxCore, "instability_check") == "error"
            @test Preferences.load_preference(LuxLib, "instability_check") == "disable"
        end
    end
end
