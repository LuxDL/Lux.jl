# Using EFL for Neural ODEs
using ExplicitFluxLayers, ComponentArrays, DiffEqSensitivity, OrdinaryDiffEq, Random, Zygote, Flux

struct NeuralODE{M<:EFL.AbstractExplicitLayer,So,Se,T,K} <: EFL.AbstractExplicitContainerLayer{(:model,)}
    model::M
    solver::So
    sensealg::Se
    tspan::T
    kwargs::K
end

function NeuralODE(
    model::EFL.AbstractExplicitLayer,
    solver=Tsit5(),
    sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()),
    tspan=(0.0f0, 1.0f0);
    kwargs...,
)
    return NeuralODE(model, solver, sensealg, tspan, kwargs)
end

function (n::NeuralODE)(x, ps, st)
    dudt(u, p, t) = n.model(u, p, st)[1]
    prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps)
    return solve(prob, n.solver; sensealg=n.sensealg, n.kwargs...) , st
end

node = NeuralODE(EFL.Dense(2, 2))

# Won't be needed if/once we move to ComponentArray as default
ps, st = EFL.setup(MersenneTwister(0), node) .|> ComponentArray

x = randn(Float32, 2, 1)

gradient(p -> sum(Array(node(x, p, st)[1])), ps)

# GPU
ps, st = (ps, st) .|> gpu

x = x |> gpu

gradient(p -> sum(gpu(node(x, p, st)[1])), ps)
