# This is mostly an internal implementation detail that users shouldn't need to worry about.
# We can remove this once https://github.com/FluxML/Optimisers.jl/issues/205 is resolved.
module ReactantCompatibleOptimisers

using ConcreteStructs: @concrete
using Optimisers: Optimisers, AbstractRule
using Setfield: Setfield, @set!

using ..Lux: Lux, Utils

abstract type ReactantCompatibleOptimisersRule <: AbstractRule end

function make_reactant_compatible(opt::AbstractRule)
    @warn "`make_reactant_compatible` is not defined for $(opt). Returning the original \
           optimizer. This means adjusting learning rate and other parameters won't \
           reflect in the generated MLIR." maxlog=1
    return opt
end
make_reactant_compatible(opt::ReactantCompatibleOptimisersRule) = opt

function setfield_if_present(opt, field::Symbol, nt::NamedTuple)
    if hasfield(typeof(nt), field)
        return Setfield.set(
            opt, Setfield.PropertyLens{field}(),
            convert(typeof(getproperty(opt, field)), getproperty(nt, field))
        )
    end
    return opt
end

function Optimisers._adjust(opt::ReactantCompatibleOptimisersRule, nt::NamedTuple)
    for field in fieldnames(typeof(opt))
        opt = setfield_if_present(opt, field, nt)
    end
    return opt
end

# OptimiserChain
function make_reactant_compatible(opt::Optimisers.OptimiserChain)
    return Optimisers.OptimiserChain(make_reactant_compatible.(opt.opts))
end

# Descent
@concrete struct ReactantDescent <: ReactantCompatibleOptimisersRule
    eta
end

function make_reactant_compatible(opt::Optimisers.Descent)
    return ReactantDescent(Utils.to_rarray(opt.eta; track_numbers=true))
end

Optimisers.init(::ReactantDescent, ::AbstractArray) = nothing

function Optimisers.apply!(opt::ReactantDescent, state, x::AbstractArray{T}, dx) where {T}
    η = T(opt.eta)
    return state, @. dx * η
end

# Momentum
@concrete struct ReactantMomentum <: ReactantCompatibleOptimisersRule
    eta
    rho
end

function make_reactant_compatible(opt::Optimisers.Momentum)
    return ReactantMomentum(
        Utils.to_rarray(opt.eta; track_numbers=true),
        Utils.to_rarray(opt.rho; track_numbers=true)
    )
end

function Optimisers.init(::ReactantMomentum, x::AbstractArray)
    return Optimisers.init(Optimisers.Momentum(0.0, 0.0), x)
end

function Optimisers.apply!(opt::ReactantMomentum, mvel, ::AbstractArray{T}, dx) where {T}
    η, ρ = T(opt.eta), T(opt.rho)
    @. mvel = ρ * mvel + η * dx
    return mvel, mvel
end

# Adam
@concrete struct ReactantAdam <: ReactantCompatibleOptimisersRule
    eta
    beta
    epsilon
end

function make_reactant_compatible(opt::Optimisers.Adam)
    return ReactantAdam(
        Utils.to_rarray(opt.eta; track_numbers=true),
        Utils.to_rarray(opt.beta; track_numbers=true),
        Utils.to_rarray(opt.epsilon; track_numbers=true)
    )
end

function Optimisers.init(opt::ReactantAdam, x::AbstractArray{T}) where {T}
    return (
        zero(x),
        zero(x),
        (Utils.promote_to(T, opt.beta[1]), Utils.promote_to(T, opt.beta[2]))
    )
end

function Optimisers.apply!(o::ReactantAdam, state, ::AbstractArray{T}, dx) where {T}
    η, β, ϵ = T(o.eta), T.(o.beta), T(o.epsilon) # XXX: See Optimisers._eps
    mt, vt, βt = state

    mt = @. β[1] * mt + (1 - β[1]) * dx
    vt = @. β[2] * vt + (1 - β[2]) * abs2(dx)
    dx′ = @. mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ) * η

    return (mt, vt, βt .* β), dx′
end

# AdamW
@concrete struct ReactantAdamW <: ReactantCompatibleOptimisersRule
    eta
    beta
    lambda
    epsilon
    couple::Bool
end

function make_reactant_compatible(opt::Optimisers.AdamW)
    return ReactantAdamW(
        Utils.to_rarray(opt.eta; track_numbers=true),
        Utils.to_rarray(opt.beta; track_numbers=true),
        Utils.to_rarray(opt.lambda; track_numbers=true),
        Utils.to_rarray(opt.epsilon; track_numbers=true),
        opt.couple
    )
end

function Optimisers.init(opt::ReactantAdamW, x::AbstractArray{T}) where {T}
    return (
        zero(x),
        zero(x),
        (Utils.promote_to(T, opt.beta[1]), Utils.promote_to(T, opt.beta[2]))
    )
end

function Optimisers.apply!(o::ReactantAdamW, state, x::AbstractArray{T}, dx) where {T}
    η, β, ϵ, λ = T(o.eta), T.(o.beta), T(o.epsilon), T(o.lambda) # XXX: See Optimisers._eps
    mt, vt, βt = state

    # standard Adam update with learning rate eta=1
    mt = @. β[1] * mt + (1 - β[1]) * dx
    vt = @. β[2] * vt + (1 - β[2]) * abs2(dx)
    dx′ = @. mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ)

    # apply learning rate and weight decay
    if o.couple
        dx′′ = @. η * (dx′ + λ * x)
    else
        dx′′ = @. η * dx′ + λ * x
    end

    return (mt, vt, βt .* β), dx′′
end

end
