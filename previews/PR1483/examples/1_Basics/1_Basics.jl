using Lux, Random

rng = Random.default_rng()
Random.seed!(rng, 0)

x = [1, 2, 3]

x = [1 2; 3 4]

x = rand(rng, 5, 3)

x = rand(BigFloat, 5, 3)

x = rand(Float32, 5, 3)

length(x)

size(x)

x

x[2, 3]

x[:, 3]

x + x

x - x

x .+ 1

zeros(5, 5) .+ (1:5)

zeros(5, 5) .+ (1:5)'

(1:5) .* (1:5)'

W = randn(5, 10)
x = rand(10)
W * x

x = reshape(1:8, 2, 4)

x_copy = copy(x)
view(x_copy, :, 1) .= 0

println("Original Array ", x)
println("Mutated Array ", x_copy)

rng = Xoshiro(0)     # Creates a Xoshiro PRNG with seed 0

random_vectors = Vector{Vector{Float64}}(undef, 3)
for i in 1:3
    random_vectors[i] = rand(Lux.replicate(rng), 10)
    println("Iteration $i ", random_vectors[i])
end
@assert random_vectors[1] ≈ random_vectors[2] ≈ random_vectors[3]

for i in 1:3
    println("Iteration $i ", rand(rng, 10))
end

using ComponentArrays, ForwardDiff, Zygote

f(x) = x' * x / 2
∇f(x) = x  # `∇` can be typed as `\nabla<TAB>`
v = randn(rng, Float32, 4)

println("Actual Gradient: ", ∇f(v))
println("Computed Gradient via Reverse Mode AD (Zygote): ", only(Zygote.gradient(f, v)))
println("Computed Gradient via Forward Mode AD (ForwardDiff): ", ForwardDiff.gradient(f, v))

f(x) = x .* x ./ 2
x = randn(rng, Float32, 5)
v = ones(Float32, 5)

jvp = jacobian_vector_product(f, AutoForwardDiff(), x, v)
println("JVP: ", jvp)

vjp = vector_jacobian_product(f, AutoZygote(), x, v)
println("VJP: ", vjp)

model = Dense(10 => 5)

rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps)

n_samples = 20
x_dim = 10
y_dim = 5
nothing #hide

W = randn(rng, Float32, y_dim, x_dim)
b = randn(rng, Float32, y_dim)
nothing #hide

x_samples = randn(rng, Float32, x_dim, n_samples)
y_samples = W * x_samples .+ b .+ 0.01f0 .* randn(rng, Float32, y_dim, n_samples)
println("x shape: ", size(x_samples), "; y shape: ", size(y_samples))

using Optimisers, Printf

lossfn = MSELoss()

println("Loss Value with ground true parameters: ", lossfn(W * x_samples .+ b, y_samples))

function train_model!(model, ps, st, opt, nepochs::Int)
    tstate = Training.TrainState(model, ps, st, opt)
    for i in 1:nepochs
        grads, loss, _, tstate = Training.single_train_step!(
            AutoZygote(), lossfn, (x_samples, y_samples), tstate
        )
        if i == 1 || i % 1000 == 0 || i == nepochs
            @printf "Loss Value after %6d iterations: %.8f\n" i loss
        end
    end
    return tstate.model, tstate.parameters, tstate.states
end

model, ps, st = train_model!(model, ps, st, Descent(0.01f0), 10000)

println("Loss Value after training: ", lossfn(first(model(x_samples, ps, st)), y_samples))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
