# [Nested AutoDiff with Reactant](@id nested_autodiff_reactant)

We will be using the example from [issue 614](https://github.com/LuxDL/Lux.jl/issues/614).

```@example nested_ad_reactant
using Reactant, Enzyme, Lux, Random, LinearAlgebra

const xdev = reactant_device()
const cdev = cpu_device()

# XXX: We need to be able to compile this with a for-loop else tracing time will scale
#      proportionally to the number of elements in the input.
function ∇potential(potential, x)
    dxs = onehot(x)
    ∇p = similar(x)
    for i in eachindex(dxs)
        dxᵢ = dxs[i]
        res = only(Enzyme.autodiff(
            Enzyme.set_abi(Forward, Reactant.ReactantABI), potential, Duplicated(x, dxᵢ)
        ))
        @allowscalar ∇p[i] = res[i]
    end
    return ∇p
end

function ∇²potential(potential, x)
    dxs = onehot(x)
    ∇²p = similar(x)
    for i in eachindex(dxs)
        dxᵢ = dxs[i]
        res = only(Enzyme.autodiff(
            Enzyme.set_abi(Forward, Reactant.ReactantABI),
            ∇potential, Const(potential), Duplicated(x, dxᵢ)
        ))
        @allowscalar ∇²p[i] = res[i]
    end
    return ∇²p
end

struct PotentialNet{P} <: Lux.AbstractLuxWrapperLayer{:potential}
    potential::P
end

function (potential::PotentialNet)(x, ps, st)
    pnet = StatefulLuxLayer{true}(potential.potential, ps, st)
    return ∇²potential(pnet, x), pnet.st
end

model = PotentialNet(Dense(5 => 5, gelu))
ps, st = Lux.setup(Random.default_rng(), model) |> xdev

x_ra = randn(Float32, 5, 3) |> xdev

model_compiled = @compile model(x_ra, ps, st)
model_compiled(x_ra, ps, st)

sumabs2first(model, x, ps, st) = sum(abs2, first(model(x, ps, st)))

function enzyme_gradient(model, x, ps, st)
    return Enzyme.gradient(
        Enzyme.Reverse, Const(sumabs2first), Const(model), Const(x), ps, Const(st)
    )
end

@jit enzyme_gradient(model, x_ra, ps, st)
```
