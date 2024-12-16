using ReTestItems, Pkg, Preferences, Test
using InteractiveUtils, Hwloc

@info sprint(versioninfo)

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))
const ALL_LUX_TEST_GROUPS = [
    "core_layers", "normalize_layers", "autodiff", "recurrent_layers", "misc"
]

Sys.iswindows() || push!(ALL_LUX_TEST_GROUPS, "reactant")

INPUT_TEST_GROUP = lowercase(get(ENV, "LUX_TEST_GROUP", "all"))
const LUX_TEST_GROUP = if startswith("!", INPUT_TEST_GROUP[1])
    exclude_group = lowercase.(split(INPUT_TEST_GROUP[2:end], ","))
    filter(x -> x ∉ exclude_group, ALL_LUX_TEST_GROUPS)
else
    [INPUT_TEST_GROUP]
end
@info "Running tests for group: $LUX_TEST_GROUP"

const EXTRA_PKGS = Pkg.PackageSpec[]
const EXTRA_DEV_PKGS = Pkg.PackageSpec[]

if ("all" in LUX_TEST_GROUP || "misc" in LUX_TEST_GROUP)
    push!(EXTRA_PKGS, Pkg.PackageSpec("MPI"))
    (BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") &&
        push!(EXTRA_PKGS, Pkg.PackageSpec("NCCL"))
    push!(EXTRA_PKGS, Pkg.PackageSpec("Flux"))
end

if !Sys.iswindows()
    ("all" in LUX_TEST_GROUP || "reactant" in LUX_TEST_GROUP) &&
        push!(EXTRA_PKGS, Pkg.PackageSpec("Reactant"))
end

if (BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda")
    if isdir(joinpath(@__DIR__, "../lib/LuxCUDA"))
        @info "Using local LuxCUDA"
        push!(EXTRA_DEV_PKGS, Pkg.PackageSpec(; path=joinpath(@__DIR__, "../lib/LuxCUDA")))
    else
        push!(EXTRA_PKGS, Pkg.PackageSpec("LuxCUDA"))
    end
end
(BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") &&
    push!(EXTRA_PKGS, Pkg.PackageSpec(; name="AMDGPU"))

if !isempty(EXTRA_PKGS) || !isempty(EXTRA_DEV_PKGS)
    @info "Installing Extra Packages for testing" EXTRA_PKGS EXTRA_DEV_PKGS
    isempty(EXTRA_PKGS) || Pkg.add(EXTRA_PKGS)
    isempty(EXTRA_DEV_PKGS) || Pkg.develop(EXTRA_DEV_PKGS)
    Base.retry_load_extensions()
    Pkg.instantiate()
    Pkg.precompile()
end

if BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda"
    using LuxCUDA
    @info sprint(CUDA.versioninfo)
end

using Lux

@testset "Load Tests" begin
    @testset "Load Packages Tests" begin
        @test_throws ErrorException FromFluxAdaptor()(1)
        showerror(stdout, Lux.FluxModelConversionException("cannot convert"))
        println()

        @test_throws ErrorException ToSimpleChainsAdaptor(nothing)(Dense(2 => 2))
        showerror(stdout, Lux.SimpleChainsModelConversionException(Dense(2 => 2)))
        println()

        @test_throws ErrorException vector_jacobian_product(
            x -> x, AutoZygote(), rand(2), rand(2))

        @test_throws ArgumentError batched_jacobian(x -> x, AutoTracker(), rand(2, 2))
        @test_throws ErrorException batched_jacobian(x -> x, AutoZygote(), rand(2, 2))
    end

    @testset "Ext Loading Check" begin
        @test !Lux.is_extension_loaded(Val(:Zygote))
        using Zygote
        @test Lux.is_extension_loaded(Val(:Zygote))
    end

    # These need to be run before MPI or NCCL is ever loaded
    @testset "Ensure MPI and NCCL are loaded" begin
        @test_throws ErrorException MPIBackend()
        @test_throws ErrorException NCCLBackend()
    end

    @testset "rule_config" begin
        @test_throws ErrorException Lux.AutoDiffInternalImpl.rule_config(Val(:Zygote2))
    end
end

# Type Stability tests fail if run with DispatchDoctor enabled
if "all" in LUX_TEST_GROUP || "core_layers" in LUX_TEST_GROUP
    @testset "Zygote Type Stability" begin
        include("zygote_type_stability.jl")
    end
end

# Eltype Matching Tests
if ("all" in LUX_TEST_GROUP || "misc" in LUX_TEST_GROUP)
    @testset "eltype_mismath_handling: $option" for option in (
        "none", "warn", "convert", "error")
        set_preferences!(Lux, "eltype_mismatch_handling" => option; force=true)
        try
            run(`$(Base.julia_cmd()) --color=yes --project=$(dirname(Pkg.project().path))
                --startup-file=no --code-coverage=user $(@__DIR__)/eltype_matching.jl`)
            @test true
        catch
            @test false
        end
    end
    set_preferences!(Lux, "eltype_mismatch_handling" => "none"; force=true)
end

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS",
        string(min(Hwloc.num_physical_cores(), Sys.isapple() ? 2 : 4))))

const RETESTITEMS_NWORKER_THREADS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKER_THREADS",
        string(max(Hwloc.num_virtual_cores() ÷ RETESTITEMS_NWORKERS, 1))))

@testset "Lux.jl Tests" begin
    for (i, tag) in enumerate(LUX_TEST_GROUP)
        @info "Running tests for group: [$(i)/$(length(LUX_TEST_GROUP))] $tag"

        nworkers = (tag == "reactant") || (BACKEND_GROUP == "amdgpu") ? 0 :
                   RETESTITEMS_NWORKERS

        ReTestItems.runtests(Lux;
            tags=(tag == "all" ? nothing : [Symbol(tag)]), testitem_timeout=2400,
            nworkers, nworker_threads=RETESTITEMS_NWORKER_THREADS
        )
    end
end

# Distributed Tests
if ("all" in LUX_TEST_GROUP || "misc" in LUX_TEST_GROUP)
    using MPI

    nprocs_str = get(ENV, "JULIA_MPI_TEST_NPROCS", "")
    nprocs = nprocs_str == "" ? clamp(Sys.CPU_THREADS, 2, 4) : parse(Int, nprocs_str)
    testdir = @__DIR__
    isdistributedtest(f) = endswith(f, "_distributedtest.jl")
    distributedtestfiles = String[]
    for (root, dirs, files) in walkdir(testdir)
        for file in files
            if isdistributedtest(file)
                push!(distributedtestfiles, joinpath(root, file))
            end
        end
    end

    @info "Running Distributed Tests with $nprocs processes"

    cur_proj = dirname(Pkg.project().path)

    include("setup_modes.jl")

    @testset "distributed tests: $(mode)" for (mode, aType, dev, ongpu) in MODES
        backends = mode == "cuda" ? ("mpi", "nccl") : ("mpi",)
        for backend_type in backends
            np = backend_type == "nccl" ? min(nprocs, length(CUDA.devices())) : nprocs
            @testset "Backend: $(backend_type)" begin
                @testset "$(basename(file))" for file in distributedtestfiles
                    @info "Running $file with $backend_type backend on $mode device"
                    try
                        run(`$(MPI.mpiexec()) -n $(np) $(Base.julia_cmd()) --color=yes \
                            --code-coverage=user --project=$(cur_proj) --startup-file=no \
                            $(file) $(mode) $(backend_type)`)
                        @test true
                    catch
                        @test false
                    end
                end
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
                luxcore="invalid")

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
