@testitem "Tensorflow Saved Model Export" tags = [:reactant] setup = [SharedTestSetup] begin
    using Lux, Reactant, PythonCall, Random

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

    model_dir = tempname()

    Lux.Serialization.export_as_tf_saved_model(model_dir, model, x, ps, st)

    res = Array(@jit(model(x, ps, Lux.testmode(st)))[1])

    tf = pyimport("tensorflow")
    np = pyimport("numpy")

    restored_model = tf.saved_model.load(model_dir)

    x_tf = tf.constant(np.asarray(permutedims(Array(x), (4, 3, 2, 1))); dtype=tf.float32)
    res_tf = permutedims(PyArray(restored_model.f(x_tf)[0]), (2, 1))

    @test res ≈ res_tf atol = 1e-3 rtol = 1e-3
end
