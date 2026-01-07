# [Nested AutoDiff with Reactant](@id nested_autodiff_reactant)

We will be using the example from [issue 614](https://github.com/LuxDL/Lux.jl/issues/614).

```@example nested_ad_reactant
using Reactant, Enzyme, Lux, Random, LinearAlgebra

const xdev = reactant_device(; force=true)
const cdev = cpu_device()

function ∇potential(potential, x::AbstractMatrix)
    N, B = size(x)
    dxs = Reactant.materialize_traced_array(reshape(stack(onehot(x)), N, B, N, B))
    ∇p = similar(x)
    @trace for i in 1:B
        @trace for j in 1:N
            dxᵢ = dxs[:, :, j, i]
            res = only(Enzyme.autodiff(Forward, potential, Duplicated(x, dxᵢ)))
            @allowscalar ∇p[j, i] = res[j, i]
            @show res
            @show dxᵢ
        end
    end
    return ∇p
end

model = Dense(5 => 5, gelu)
ps, st = Lux.setup(Random.default_rng(), model) |> xdev
pnet = StatefulLuxLayer(model, ps, st)

x_ra = randn(Float32, 5, 3) |> xdev

@code_hlo pnet(x_ra)
@code_hlo ∇potential(pnet, x_ra)

function ∇²potential(potential, x)
    dxs = stack(onehot(x))
    ∇²p = similar(x)
    colons = [Colon() for _ in 1:ndims(x)]
    @trace for i in 1:length(x)
        dxᵢ = dxs[colons..., i]
        res = only(Enzyme.autodiff(
            Forward, ∇potential, Const(potential), Duplicated(x, dxᵢ)
        ))
        @allowscalar ∇²p[i] = res[i]
    end
    return ∇²p
end

@code_hlo ∇²potential(pnet, x_ra)

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
