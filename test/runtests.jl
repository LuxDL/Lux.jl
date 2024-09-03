using ReTestItems, Pkg, Preferences, Test
using InteractiveUtils, Hwloc

@info sprint(versioninfo)

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))
const ALL_LUX_TEST_GROUPS = [
    "core_layers", "contrib", "helpers", "distributed", "normalize_layers",
    "others", "autodiff", "recurrent_layers", "fluxcompat"]

__INPUT_TEST_GROUP = lowercase(get(ENV, "LUX_TEST_GROUP", "all"))
const LUX_TEST_GROUP = if startswith("!", __INPUT_TEST_GROUP[1])
    exclude_group = lowercase.(split(__INPUT_TEST_GROUP[2:end], ","))
    filter(x -> x ∉ exclude_group, ALL_LUX_TEST_GROUPS)
else
    [__INPUT_TEST_GROUP]
end
@info "Running tests for group: $LUX_TEST_GROUP"

const EXTRA_PKGS = String[]

if ("all" in LUX_TEST_GROUP || "distributed" in LUX_TEST_GROUP)
    push!(EXTRA_PKGS, "MPI")
    (BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") && push!(EXTRA_PKGS, "NCCL")
end
("all" in LUX_TEST_GROUP || "fluxcompat" in LUX_TEST_GROUP) && push!(EXTRA_PKGS, "Flux")
(BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") && push!(EXTRA_PKGS, "LuxCUDA")
(BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") && push!(EXTRA_PKGS, "AMDGPU")

if !isempty(EXTRA_PKGS)
    @info "Installing Extra Packages for testing" EXTRA_PKGS=EXTRA_PKGS
    Pkg.add(EXTRA_PKGS)
    Pkg.update()
    Base.retry_load_extensions()
    Pkg.instantiate()
end

using Lux

Lux.set_dispatch_doctor_preferences!(; luxcore="error", luxlib="error")

@testset "Load Tests" begin
    @testset "Load Packages Tests" begin
        @test_throws ErrorException FromFluxAdaptor()(1)
        showerror(stdout, Lux.FluxModelConversionException("cannot convert"))

        @test_throws ErrorException ToSimpleChainsAdaptor(nothing)(Dense(2 => 2))
        showerror(stdout, Lux.SimpleChainsModelConversionException(Dense(2 => 2)))

        @test_throws ErrorException vector_jacobian_product(
            x -> x, AutoZygote(), rand(2), rand(2))

        @test_throws ArgumentError batched_jacobian(x -> x, AutoEnzyme(), rand(2, 2))
        @test_throws ErrorException batched_jacobian(x -> x, AutoZygote(), rand(2, 2))
    end

    @testset "Ext Loading Check" begin
        @test !Lux.is_extension_loaded(Val(:Zygote))
        using Zygote
        @test Lux.is_extension_loaded(Val(:Zygote))

        @test !Lux.is_extension_loaded(Val(:DynamicExpressions))
        using DynamicExpressions
        @test Lux.is_extension_loaded(Val(:DynamicExpressions))
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

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Hwloc.num_physical_cores(), 4))))

@testset "Lux.jl Tests" begin
    for (i, tag) in enumerate(LUX_TEST_GROUP)
        (tag == "distributed" || tag == "eltype_match") && continue
        @info "Running tests for group: [$(i)/$(length(LUX_TEST_GROUP))] $tag"

        ReTestItems.runtests(Lux; tags=(tag == "all" ? nothing : [Symbol(tag)]),
            nworkers=RETESTITEMS_NWORKERS, testitem_timeout=3600)
    end
end

# Distributed Tests
if ("all" in LUX_TEST_GROUP || "distributed" in LUX_TEST_GROUP)
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
                    run(`$(MPI.mpiexec()) -n $(np) $(Base.julia_cmd()) --color=yes \
                        --code-coverage=user --project=$(cur_proj) --startup-file=no $(file) \
                        $(mode) $(backend_type)`)
                    Test.@test true
                end
            end
        end
    end
end

# Eltype Matching Tests
if ("all" in LUX_TEST_GROUP || "eltype_match" in LUX_TEST_GROUP)
    @testset "eltype_mismath_handling: $option" for option in (
        "none", "warn", "convert", "error")
        set_preferences!(Lux, "eltype_mismatch_handling" => option; force=true)
        run(`$(Base.julia_cmd()) --color=yes --project=$(dirname(Pkg.project().path))
            --startup-file=no --code-coverage=user $(@__DIR__)/eltype_matching.jl`)
        Test.@test true
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
