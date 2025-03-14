# [Nested AutoDiff with Reactant](@id nested_autodiff_reactant)

We will be using the example from [issue 614](https://github.com/LuxDL/Lux.jl/issues/614).

```@example nested_ad_reactant
using Reactant, Enzyme, Lux, Random, LinearAlgebra

const xdev = reactant_device(; force=true)
const cdev = cpu_device()

function ∇potential(potential, x)
    dxs = stack(onehot(x))
    ∇p = similar(x)
    colons = [Colon() for _ in 1:ndims(x)]
    @trace for i in 1:length(x)
        dxᵢ = dxs[colons..., i]
        res = only(Enzyme.autodiff(
            Enzyme.set_abi(Forward, Reactant.ReactantABI), potential, Duplicated(x, dxᵢ)
        ))
        @allowscalar ∇p[i] = res[i]
    end
    return ∇p
end

function ∇²potential(potential, x)
    dxs = stack(onehot(x))
    ∇²p = similar(x)
    colons = [Colon() for _ in 1:ndims(x)]
    @trace for i in 1:length(x)
        dxᵢ = dxs[colons..., i]
        res = only(Enzyme.autodiff(
            Enzyme.set_abi(Forward, Reactant.ReactantABI),
            ∇potential, Const(potential), Duplicated(x, dxᵢ)
        ))
        @allowscalar ∇²p[i] = res[i]
    end
    return ∇²p
end

struct PotentialNet{P} <: AbstractLuxWrapperLayer{:potential}
    potential::P
end

function (potential::PotentialNet)(x, ps, st)
    pnet = StatefulLuxLayer{true}(potential.potential, ps, st)
    return ∇²potential(pnet, x), pnet.st
end

model = PotentialNet(Dense(5 => 5, gelu))
ps, st = Lux.setup(Random.default_rng(), model) |> xdev

x_ra = randn(Float32, 5, 3) |> xdev

@code_hlo model(x_ra, ps, st)

@jit model(x_ra, ps, st)
```

```@example nested_ad_reactant
sumabs2first(model, x, ps, st) = sum(abs2, first(model(x, ps, st)))

function enzyme_gradient(model, x, ps, st)
    return Enzyme.gradient(
        Enzyme.Reverse, Const(sumabs2first), Const(model), Const(x), ps, Const(st)
    )
end

@jit enzyme_gradient(model, x_ra, ps, st)
```
