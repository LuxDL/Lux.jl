# # Training a Neural ODE to Model Gravitational Waveforms

# This code is adapted from [Astroinformatics/ScientificMachineLearning](https://github.com/Astroinformatics/ScientificMachineLearning/blob/c93aac3a460d70b4cce98836b677fd9b732e94b7/neuralode_gw.ipynb)
#
# The code has been minimally adapted from
# [Keith et. al. 2021](https://arxiv.org/abs/2102.12695) which originally used Flux.jl

# ## Package Imports

using Lux, ComponentArrays, LineSearches, OrdinaryDiffEqLowOrderRK, Optimization,
      OptimizationOptimJL, Printf, Random, SciMLSensitivity
using CairoMakie

# ## Define some Utility Functions

# !!! tip
#
#     This section can be skipped. It defines functions to simulate the model, however,
#     from a scientific machine learning perspective, isn't super relevant.

# We need a very crude 2-body path. Assume the 1-body motion is a newtonian 2-body position
# vector $r = r_1 - r_2$ and use Newtonian formulas to get $r_1$, $r_2$ (e.g. Theoretical
# Mechanics of Particles and Continua 4.3)

function one2two(path, m₁, m₂)
    M = m₁ + m₂
    r₁ = m₂ / M .* path
    r₂ = -m₁ / M .* path
    return r₁, r₂
end

# Next we define a function to perform the change of variables:
# $$(\chi(t),\phi(t)) \mapsto (x(t),y(t))$$

@views function soln2orbit(soln, model_params=nothing)
    @assert size(soln, 1) ∈ [2, 4] "size(soln,1) must be either 2 or 4"

    if size(soln, 1) == 2
        χ = soln[1, :]
        ϕ = soln[2, :]

        @assert length(model_params)==3 "model_params must have length 3 when size(soln,2) = 2"
        p, M, e = model_params
    else
        χ = soln[1, :]
        ϕ = soln[2, :]
        p = soln[3, :]
        e = soln[4, :]
    end

    r = p ./ (1 .+ e .* cos.(χ))
    x = r .* cos.(ϕ)
    y = r .* sin.(ϕ)

    orbit = vcat(x', y')
    return orbit
end

# This function uses second-order one-sided difference stencils at the endpoints;
# see https://doi.org/10.1090/S0025-5718-1988-0935077-0

function d_dt(v::AbstractVector, dt)
    a = -3 / 2 * v[1] + 2 * v[2] - 1 / 2 * v[3]
    b = (v[3:end] .- v[1:(end - 2)]) / 2
    c = 3 / 2 * v[end] - 2 * v[end - 1] + 1 / 2 * v[end - 2]
    return [a; b; c] / dt
end

# This function uses second-order one-sided difference stencils at the endpoints;
# see https://doi.org/10.1090/S0025-5718-1988-0935077-0

function d2_dt2(v::AbstractVector, dt)
    a = 2 * v[1] - 5 * v[2] + 4 * v[3] - v[4]
    b = v[1:(end - 2)] .- 2 * v[2:(end - 1)] .+ v[3:end]
    c = 2 * v[end] - 5 * v[end - 1] + 4 * v[end - 2] - v[end - 3]
    return [a; b; c] / (dt^2)
end

# Now we define a function to compute the trace-free moment tensor from the orbit

function orbit2tensor(orbit, component, mass=1)
    x = orbit[1, :]
    y = orbit[2, :]

    Ixx = x .^ 2
    Iyy = y .^ 2
    Ixy = x .* y
    trace = Ixx .+ Iyy

    if component[1] == 1 && component[2] == 1
        tmp = Ixx .- trace ./ 3
    elseif component[1] == 2 && component[2] == 2
        tmp = Iyy .- trace ./ 3
    else
        tmp = Ixy
    end

    return mass .* tmp
end

function h_22_quadrupole_components(dt, orbit, component, mass=1)
    mtensor = orbit2tensor(orbit, component, mass)
    mtensor_ddot = d2_dt2(mtensor, dt)
    return 2 * mtensor_ddot
end

function h_22_quadrupole(dt, orbit, mass=1)
    h11 = h_22_quadrupole_components(dt, orbit, (1, 1), mass)
    h22 = h_22_quadrupole_components(dt, orbit, (2, 2), mass)
    h12 = h_22_quadrupole_components(dt, orbit, (1, 2), mass)
    return h11, h12, h22
end

function h_22_strain_one_body(dt::T, orbit) where {T}
    h11, h12, h22 = h_22_quadrupole(dt, orbit)

    h₊ = h11 - h22
    hₓ = T(2) * h12

    scaling_const = √(T(π) / 5)
    return scaling_const * h₊, -scaling_const * hₓ
end

function h_22_quadrupole_two_body(dt, orbit1, mass1, orbit2, mass2)
    h11_1, h12_1, h22_1 = h_22_quadrupole(dt, orbit1, mass1)
    h11_2, h12_2, h22_2 = h_22_quadrupole(dt, orbit2, mass2)
    h11 = h11_1 + h11_2
    h12 = h12_1 + h12_2
    h22 = h22_1 + h22_2
    return h11, h12, h22
end

function h_22_strain_two_body(dt::T, orbit1, mass1, orbit2, mass2) where {T}
    ## compute (2,2) mode strain from orbits of BH 1 of mass1 and BH2 of mass 2

    @assert abs(mass1 + mass2 - 1.0)<1e-12 "Masses do not sum to unity"

    h11, h12, h22 = h_22_quadrupole_two_body(dt, orbit1, mass1, orbit2, mass2)

    h₊ = h11 - h22
    hₓ = T(2) * h12

    scaling_const = √(T(π) / 5)
    return scaling_const * h₊, -scaling_const * hₓ
end

function compute_waveform(dt::T, soln, mass_ratio, model_params=nothing) where {T}
    @assert mass_ratio≤1 "mass_ratio must be <= 1"
    @assert mass_ratio≥0 "mass_ratio must be non-negative"

    orbit = soln2orbit(soln, model_params)
    if mass_ratio > 0
        m₂ = inv(T(1) + mass_ratio)
        m₁ = mass_ratio * m₂

        orbit₁, orbit₂ = one2two(orbit, m₁, m₂)
        waveform = h_22_strain_two_body(dt, orbit₁, m₁, orbit₂, m₂)
    else
        waveform = h_22_strain_one_body(dt, orbit)
    end
    return waveform
end

# ## Simulating the True Model

# `RelativisticOrbitModel` defines system of odes which describes motion of point like
# particle in schwarzschild background, uses

# $$u[1] = \chi$$
# $$u[2] = \phi$$

# where, $p$, $M$, and $e$ are constants

function RelativisticOrbitModel(u, (p, M, e), t)
    χ, ϕ = u

    numer = (p - 2 - 2 * e * cos(χ)) * (1 + e * cos(χ))^2
    denom = sqrt((p - 2)^2 - 4 * e^2)

    χ̇ = numer * sqrt(p - 6 - 2 * e * cos(χ)) / (M * (p^2) * denom)
    ϕ̇ = numer / (M * (p^(3 / 2)) * denom)

    return [χ̇, ϕ̇]
end

mass_ratio = 0.0         # test particle
u0 = Float64[π, 0.0]     # initial conditions
datasize = 250
tspan = (0.0f0, 6.0f4)   # timespace for GW waveform
tsteps = range(tspan[1], tspan[2]; length=datasize)  # time at each timestep
dt_data = tsteps[2] - tsteps[1]
dt = 100.0
const ode_model_params = [100.0, 1.0, 0.5]; # p, M, e

# Let's simulate the true model and plot the results using `OrdinaryDiffEq.jl`

prob = ODEProblem(RelativisticOrbitModel, u0, tspan, ode_model_params)
soln = Array(solve(prob, RK4(); saveat=tsteps, dt, adaptive=false))
waveform = first(compute_waveform(dt_data, soln, mass_ratio, ode_model_params))

begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="Time", ylabel="Waveform")

    l = lines!(ax, tsteps, waveform; linewidth=2, alpha=0.75)
    s = scatter!(ax, tsteps, waveform; marker=:circle, markersize=12, alpha=0.5)

    axislegend(ax, [[l, s]], ["Waveform Data"])

    fig
end

# ## Defiing a Neural Network Model

# Next, we define the neural network model that takes 1 input (time) and has two outputs.
# We'll make a function `ODE_model` that takes the initial conditions, neural network
# parameters and a time as inputs and returns the derivatives.

# It is typically never recommended to use globals but incase you do use them, make sure
# to mark them as `const`.

# We will deviate from the standard Neural Network initialization and use
# `WeightInitializers.jl`,
const nn = Chain(Base.Fix1(fast_activation, cos),
    Dense(1 => 32, cos; init_weight=truncated_normal(; std=1e-4), init_bias=zeros32),
    Dense(32 => 32, cos; init_weight=truncated_normal(; std=1e-4), init_bias=zeros32),
    Dense(32 => 2; init_weight=truncated_normal(; std=1e-4), init_bias=zeros32))
ps, st = Lux.setup(Random.default_rng(), nn)

# Similar to most DL frameworks, Lux defaults to using `Float32`, however, in this case we
# need Float64

const params = ComponentArray(ps |> f64)

const nn_model = StatefulLuxLayer{true}(nn, nothing, st)

# Now we define a system of odes which describes motion of point like particle with
# Newtonian physics, uses
#
# $$u[1] = \chi$$
# $$u[2] = \phi$$
#
# where, $p$, $M$, and $e$ are constants

function ODE_model(u, nn_params, t)
    χ, ϕ = u
    p, M, e = ode_model_params

    ## In this example we know that `st` is am empty NamedTuple hence we can safely ignore
    ## it, however, in general, we should use `st` to store the state of the neural network.
    y = 1 .+ nn_model([first(u)], nn_params)

    numer = (1 + e * cos(χ))^2
    denom = M * (p^(3 / 2))

    χ̇ = (numer / denom) * y[1]
    ϕ̇ = (numer / denom) * y[2]

    return [χ̇, ϕ̇]
end

# Let us now simulate the neural network model and plot the results. We'll use the untrained
# neural network parameters to simulate the model.

prob_nn = ODEProblem(ODE_model, u0, tspan, params)
soln_nn = Array(solve(prob_nn, RK4(); u0, p=params, saveat=tsteps, dt, adaptive=false))
waveform_nn = first(compute_waveform(dt_data, soln_nn, mass_ratio, ode_model_params))

begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="Time", ylabel="Waveform")

    l1 = lines!(ax, tsteps, waveform; linewidth=2, alpha=0.75)
    s1 = scatter!(
        ax, tsteps, waveform; marker=:circle, markersize=12, alpha=0.5, strokewidth=2)

    l2 = lines!(ax, tsteps, waveform_nn; linewidth=2, alpha=0.75)
    s2 = scatter!(
        ax, tsteps, waveform_nn; marker=:circle, markersize=12, alpha=0.5, strokewidth=2)

    axislegend(ax, [[l1, s1], [l2, s2]],
        ["Waveform Data", "Waveform Neural Net (Untrained)"]; position=:lb)

    fig
end

# ## Setting Up for Training the Neural Network

# Next, we define the objective (loss) function to be minimized when training the neural
# differential equations.
const mseloss = MSELoss()

function loss(θ)
    pred = Array(solve(prob_nn, RK4(); u0, p=θ, saveat=tsteps, dt, adaptive=false))
    pred_waveform = first(compute_waveform(dt_data, pred, mass_ratio, ode_model_params))
    return mseloss(pred_waveform, waveform)
end

# Warmup the loss function
loss(params)

# Now let us define a callback function to store the loss over time
const losses = Float64[]

function callback(θ, l)
    push!(losses, l)
    @printf "Training \t Iteration: %5d \t Loss: %.10f\n" θ.iter l
    return false
end

# ## Training the Neural Network

# Training uses the BFGS optimizers. This seems to give good results because the Newtonian
# model seems to give a very good initial guess

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
res = Optimization.solve(
    optprob, BFGS(; initial_stepnorm=0.01, linesearch=LineSearches.BackTracking());
    callback, maxiters=1000)

# ## Visualizing the Results

# Let us now plot the loss over time

begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="Iteration", ylabel="Loss")

    lines!(ax, losses; linewidth=4, alpha=0.75)
    scatter!(ax, 1:length(losses), losses; marker=:circle, markersize=12, strokewidth=2)

    fig
end

# Finally let us visualize the results

prob_nn = ODEProblem(ODE_model, u0, tspan, res.u)
soln_nn = Array(solve(prob_nn, RK4(); u0, p=res.u, saveat=tsteps, dt, adaptive=false))
waveform_nn_trained = first(compute_waveform(
    dt_data, soln_nn, mass_ratio, ode_model_params))

begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="Time", ylabel="Waveform")

    l1 = lines!(ax, tsteps, waveform; linewidth=2, alpha=0.75)
    s1 = scatter!(
        ax, tsteps, waveform; marker=:circle, alpha=0.5, strokewidth=2, markersize=12)

    l2 = lines!(ax, tsteps, waveform_nn; linewidth=2, alpha=0.75)
    s2 = scatter!(
        ax, tsteps, waveform_nn; marker=:circle, alpha=0.5, strokewidth=2, markersize=12)

    l3 = lines!(ax, tsteps, waveform_nn_trained; linewidth=2, alpha=0.75)
    s3 = scatter!(ax, tsteps, waveform_nn_trained; marker=:circle,
        alpha=0.5, strokewidth=2, markersize=12)

    axislegend(ax, [[l1, s1], [l2, s2], [l3, s3]],
        ["Waveform Data", "Waveform Neural Net (Untrained)", "Waveform Neural Net"];
        position=:lb)

    fig
end
