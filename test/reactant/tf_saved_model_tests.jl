include("../shared_testsetup.jl")

using CondaPkg, Lux, Random, Reactant, Test

model_dir, x_saved, res_saved = let
    dev = reactant_device()

    model = Chain(
        Conv((5, 5), 1 => 6, relu),
        BatchNorm(6),
        MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu),
        BatchNorm(16),
        MaxPool((2, 2)),
        FlattenLayer(3),
        Chain(Dense(256 => 128, relu), Dense(128 => 84, relu), Dense(84 => 10)),
    )

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model) |> dev
    x = rand(Float32, 28, 28, 1, 4) |> dev

    dir = tempname()
    Lux.Serialization.export_as_tf_saved_model(dir, model, x, ps, st)
    res = Array(@jit(model(x, ps, Lux.testmode(st)))[1])
    dir, Array(x), res
end

@testset "TensorFlow SavedModel Pure-Julia Export" begin
    @test !haskey(
        Base.loaded_modules,
        Base.PkgId(Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d"), "PythonCall"),
    )
    @test isfile(joinpath(model_dir, "saved_model.pb"))
    @test filesize(joinpath(model_dir, "saved_model.pb")) > 0
    @test isfile(joinpath(model_dir, "variables", "variables.index"))
    @test isfile(joinpath(model_dir, "variables", "variables.data-00000-of-00001"))
    @test filesize(joinpath(model_dir, "variables", "variables.data-00000-of-00001")) > 0
end

if CondaPkg.backend() !== :Null
    env_dir = joinpath(dirname(CondaPkg.cur_deps_file()), ".CondaPkg")
    if isdir(env_dir)
        py_path = try
            CondaPkg.which("python")
        catch
            nothing
        end
        installed_py = if py_path === nothing
            nothing
        else
            try
                v_match = match(r"(\d+(?:\.\d+)+)", readchomp(`$py_path --version`))
                v_match === nothing ? nothing : VersionNumber(v_match[1])
            catch
                nothing
            end
        end
        if installed_py === nothing || installed_py >= v"3.14"
            rm(env_dir; recursive=true, force=true)
        end
    end
    CondaPkg.add("python"; version="<3.14,>=3.9,<4", resolve=false)
    CondaPkg.add_pip("tensorflow"; version=">=2.17", resolve=false)
    CondaPkg.add_pip("numpy"; version=">=1", resolve=false)
    CondaPkg.resolve(; force=true)
end

using PythonCall

@testset "TensorFlow SavedModel Validation" begin
    tf = if CondaPkg.backend() === :Null && get(ENV, "CI", "false") != "true"
        try
            pyimport("tensorflow")
        catch err
            @warn "Failed to import `tensorflow` with `CondaPkg.backend() === :Null`: $err"
            nothing
        end
    else
        pyimport("tensorflow")
    end

    if tf === nothing
        @test_skip false
    else
        np = pyimport("numpy")

        restored_model = tf.saved_model.load(model_dir)

        x_tf = tf.constant(np.asarray(permutedims(x_saved, (4, 3, 2, 1))); dtype=tf.float32)
        res_tf = pywith(tf.device("/CPU:0")) do _
            return permutedims(PyArray(restored_model.f(x_tf)[0]), (2, 1))
        end

        @test res_saved ≈ res_tf atol = 1e-2 rtol = 1e-2
    end
end
