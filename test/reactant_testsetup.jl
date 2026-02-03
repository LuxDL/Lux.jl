using Lux, Reactant, Enzyme

sumabs2(x::AbstractArray) = sum(abs2, x)
sumabs2(x::Tuple) = sumabs2(first(x))
sumabs2(model, x, ps, st) = sumabs2(model(x, ps, st))

function ∇sumabs2_enzyme(model, x, ps, st)
    dx = Enzyme.make_zero(x)
    dps = Enzyme.make_zero(ps)
    Enzyme.autodiff(
        Enzyme.Reverse,
        sumabs2,
        Active,
        Const(model),
        Duplicated(x, dx),
        Duplicated(ps, dps),
        Const(st),
    )
    return dx, dps
end

function ∇sumabs2_reactant_fd(model, x, ps, st)
    _, ∂x_fd, ∂ps_fd, _ = @jit Reactant.TestUtils.finite_difference_gradient(
        sumabs2, Const(model), f64(x), f64(ps), Const(f64(st))
    )
    return ∂x_fd, ∂ps_fd
end

function ∇sumabs2_reactant(model, x, ps, st)
    return @jit ∇sumabs2_enzyme(model, x, ps, st)
end
