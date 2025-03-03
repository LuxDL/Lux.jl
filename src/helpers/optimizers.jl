# This is mostly an internal implementation detail that users shouldn't need to worry about.
# We can remove this once https://github.com/FluxML/Optimisers.jl/issues/205 is resolved.
module ReactantCompatibleOptimisers

using ConcreteStructs: @concrete
using Optimisers: Optimisers, AbstractRule
using Setfield: Setfield, @set!

using ..Lux: Lux, Utils

abstract type ReactantCompatibleOptimisersRule <: AbstractRule end

function make_reactant_compatible(opt::AbstractRule)
    fields = fieldnames(typeof(opt))
    for field in fields
        opt = Setfield.set(
            opt,
            Setfield.PropertyLens{field}(),
            Utils.to_rarray(getfield(opt, field); track_numbers=true)
        )
    end
    return opt
end

function make_reactant_compatible(opt::Optimisers.OptimiserChain)
    return Optimisers.OptimiserChain(make_reactant_compatible.(opt.opts))
end

function make_reactant_compatible(::Optimisers.AccumGrad)
    # The conditional will have to be traced
    error("Reactant + AccumGrad is currently incompatible with the Lux Training API")
end

end
