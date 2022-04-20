# Using EFL for Neural ODEs
using ExplicitFluxLayers, DiffEqSensitivity, OrdinaryDiffEq, Random, Flux

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
    function dudt(u, p, t)
        u_, st = n.model(u, p, st)
        return u_
    end
    prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps)
    return solve(prob, n.solver; sensealg=n.sensealg, n.kwargs...), st
end

diffeqsol_to_array(x::ODESolution{T,N,<:AbstractVector{<:CuArray}}) where {T,N} = gpu(x)
diffeqsol_to_array(x::ODESolution) = Array(x)

node =  EFL.Chain(
    NeuralODE(
        EFL.Chain(
            x -> x .^ 3,
            EFL.Dense(2, 50, tanh),
            EFL.BatchNorm(50),
            EFL.Dense(50, 2),
        )
    ),
    diffeqsol_to_array
)

for device in [cpu, gpu]
    ps, st = EFL.setup(MersenneTwister(0), node) .|> device

    x = randn(MersenneTwister(1), Float32, 2, 128) |> device

    gradient(p -> sum(node(x, p, st)[1]), ps)
end
