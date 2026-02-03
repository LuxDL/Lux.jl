using Pkg, Lux, Test, ParallelTestRunner, LuxTestUtils, Preferences

parsed_args = parse_args(@isdefined(TEST_ARGS) ? TEST_ARGS : ARGS; custom=["BACKEND_GROUP"])

const BACKEND_GROUP = lowercase(
    something(get(parsed_args.custom, "BACKEND_GROUP", nothing), "all")
)

# Find all tests
testsuite = find_tests(@__DIR__)

filter_tests!(testsuite, parsed_args)

# Remove shared setup files that shouldn't be run directly
delete!(testsuite, "shared_testsetup")
delete!(testsuite, "setup_modes")
delete!(testsuite, "eltype_matching")
delete!(testsuite, "reactant_testsetup")
for k in keys(testsuite)
    if startswith(k, "distributed") || startswith(k, "downstream")
        delete!(testsuite, k)
    end
end

total_jobs = min(
    something(parsed_args.jobs, ParallelTestRunner.default_njobs()), length(keys(testsuite))
)

withenv(
    "XLA_REACTANT_GPU_MEM_FRACTION" => 1 / (total_jobs + 0.1),
    "XLA_REACTANT_GPU_PREALLOCATE" => false,
    "JULIA_CUDA_HARD_MEMORY_LIMIT" => "$(100 / (total_jobs + 0.1))%",
    "BACKEND_GROUP" => BACKEND_GROUP,
) do
    runtests(Lux, parsed_args; testsuite)
end

# Eltype Matching Tests
if isempty(parsed_args.positionals) || "others" ∈ parsed_args.positionals
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

# Various Downstream Integration Tests
# We only run these on 1.11+ due to nicer handling of [sources]
if (
    (isempty(parsed_args.positionals) || "others" ∈ parsed_args.positionals) &&
    VERSION ≥ v"1.11-"
)
    testdir = @__DIR__
    isintegrationtest(f) = endswith(f, "_integrationtest.jl")
    integrationtestfiles = String[]
    for (root, dirs, files) in walkdir(testdir)
        for file in files
            if isintegrationtest(file)
                fullpath = joinpath(root, file)
                if VERSION >= v"1.12-" && occursin("SimpleChains", fullpath)
                    continue
                end
                push!(integrationtestfiles, fullpath)
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
if (isempty(parsed_args.positionals) || "others" ∈ parsed_args.positionals) &&
    VERSION ≥ v"1.11-"
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
if (isempty(parsed_args.positionals) || "others" ∈ parsed_args.positionals)
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
