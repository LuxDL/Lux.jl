# Nested Automatic Differentiation

!!! note

    This is a relatively new feature in Lux, so there might be some rough edges. If you
    encounter any issues, please let us know by opening an issue on the
    [GitHub repository](https://github.com/LuxDL/Lux.jl).

In this manual, we will explore how to use automatic differentiation (AD) inside your layers
or loss functions and have Lux automatically switch the AD backend with a faster one when
needed.

!!! tip

    Don't wan't Lux to do this switching for you? You can disable it by setting the
    `DisableAutomaticNestedADSwitching` Preference to `true`.

    Remember that if you are using ForwardDiff inside a Zygote call, it will drop gradients
    (with a warning message), so it is not recommended to use this combination.

Let's explore this using some questions that were posted on the
[Julia Discourse forum](https://discourse.julialang.org/).

```@example nested_ad
using Lux, LinearAlgebra, Zygote, ForwardDiff, Random
using ComponentArrays, FiniteDiff
```

First let's set the stage using some minor changes that need to be made for this feature to
work:

  - Switching only works if a [`StatefulLuxLayer`](@ref) is being used, with the following
    function calls:
      - `(<some-function> ∘ <StatefulLuxLayer>)(x::AbstractArray)`
      - `(<StatefulLuxLayer> ∘ <some-function>)(x::AbstractArray)`
      - `(<StatefulLuxLayer>)(x::AbstractArray)`
  - Currently we have custom routines implemented for:
      - `Zygote.<gradient|jacobian>`
      - `ForwardDiff.<gradient|jacobian>`
  - Switching only happens for `ChainRules` compatible AD libraries.

We plan to capture `DifferentiationInterface`, `Zygote.pullback`, and `Enzyme.autodiff`
calls in the future (PRs are welcome).

## Nested AD for Neural Differential Equations (DEs)

This problem comes from `@facusapienza` on [Discourse](https://discourse.julialang.org/t/nested-and-different-ad-methods-altogether-how-to-add-ad-calculations-inside-my-loss-function-when-using-neural-differential-equations/108985).
In this case, we want to add a regularization term to the neural DE based on first-order
derivatives. The neural DE part is not important here and we can demonstrate this easily
with a standard neural network.

```@example nested_ad
function loss_function1(model, x, ps, st, y)
    # Make it a stateful layer
    smodel = StatefulLuxLayer(model, ps, st)
    ŷ = smodel(x)
    loss_emp = sum(abs2, ŷ .- y)
    # You can use `Zygote.jacobian` as well but ForwardDiff tends to be more efficient here
    J = ForwardDiff.jacobian(smodel, x)
    loss_reg = abs2(norm(J))
    return loss_emp + loss_reg
end

# Using Batchnorm to show that it is possible
model = Chain(Dense(2 => 4, tanh), BatchNorm(4), Dense(4 => 2))
ps, st = Lux.setup(Xoshiro(0), model)
x = rand(Xoshiro(0), Float32, 2, 10)
y = rand(Xoshiro(11), Float32, 2, 10)

loss_function1(model, x, ps, st, y)
```

So our loss function works, let's take the gradient (forward diff doesn't nest nicely here):

```@example nested_ad
_, ∂x, ∂ps, _, _ = Zygote.gradient(loss_function1, model, x, ps, st, y)
```

Now let's verify the gradients using finite differences:

```@example nested_ad
∂x_fd = FiniteDiff.finite_difference_gradient(x -> loss_function1(model, x, ps, st, y), x)
∂ps_fd = FiniteDiff.finite_difference_gradient(ps -> loss_function1(model, x, ps, st, y),
    ComponentArray(ps))

println("∞-norm(∂x - ∂x_fd): ", norm(∂x .- ∂x_fd, Inf))
println("∞-norm(∂ps - ∂ps_fd): ", norm(ComponentArray(∂ps) .- ∂ps_fd, Inf))
nothing; # hide
```

That's pretty good, of course you will have some error from the finite differences
calculation.

## Loss Function contains Gradient Calculation

Ok here I am going to cheat a bit. This comes from a discussion on nested AD for PINNs
on [Discourse](https://discourse.julialang.org/t/is-it-possible-to-do-nested-ad-elegantly-in-julia-pinns/98888/21).
As the consensus there, we shouldn't use nested AD for 3rd or higher order differentiation.
Note that in the example there, the user uses `ForwardDiff.derivative` but we will use
`ForwardDiff.gradient` instead, as we typically deal with array inputs and outputs.

```@example nested_ad
function loss_function2(model, t, ps, st)
    smodel = StatefulLuxLayer(model, ps, st)
    ŷ = only(Zygote.gradient(Base.Fix1(sum, abs2) ∘ smodel, t)) # Zygote returns a tuple
    return sum(abs2, ŷ .- cos.(t))
end

model = Chain(Dense(1 => 12,tanh), Dense(12 => 12,tanh), Dense(12 => 12,tanh),
    Dense(12 => 1))
ps, st = Lux.setup(Xoshiro(0), model)
t = rand(Xoshiro(0), Float32, 1, 16)
```

Now the moment of truth:

```@example nested_ad
_, ∂t, ∂ps, _ = Zygote.gradient(loss_function2, model, t, ps, st)
```

Boom that worked! Let's verify the gradient using forward diff:

```@example nested_ad
∂t_fd = ForwardDiff.gradient(t -> loss_function2(model, t, ps, st), t)
∂ps_fd = ForwardDiff.gradient(ps -> loss_function2(model, t, ps, st), ComponentArray(ps))

println("∞-norm(∂t - ∂t_fd): ", norm(∂t .- ∂t_fd, Inf))
println("∞-norm(∂ps - ∂ps_fd): ", norm(ComponentArray(∂ps) .- ∂ps_fd, Inf))
nothing; # hide
```
