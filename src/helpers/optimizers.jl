# This is mostly an internal implementation detail that users shouldn't need to worry about.
# We can remove this once https://github.com/FluxML/Optimisers.jl/issues/205 is resolved.
module ReactantCompatibleOptimisers

using ConcreteStructs: @concrete
using Functors: fmap
using Optimisers: Optimisers, AbstractRule

using ..Lux: Lux, Utils

function make_reactant_compatible(leaf::Optimisers.Leaf{<:Optimisers.OptimiserChain})
    res = make_reactant_compatible.(leaf.rule.opts, leaf.state)
    new_opts = first.(res)
    new_state = last.(res)
    rule = Optimisers.OptimiserChain(new_opts...)
    return Optimisers.Leaf(rule, new_state, leaf.frozen)
end

function make_reactant_compatible(leaf::Optimisers.Leaf{<:AbstractRule})
    rule, state = make_reactant_compatible(leaf.rule, leaf.state)
    return Optimisers.Leaf(rule, state, leaf.frozen)
end

function make_reactant_compatible(opt::Optimisers.OptimiserChain, state)
    res = make_reactant_compatible.(opt.opts, state)
    new_opts = first.(res)
    new_state = last.(res)
    return Optimisers.OptimiserChain(new_opts...), new_state
end

function make_reactant_compatible(opt::Optimisers.AbstractRule, state)
    return (
        Utils.to_rarray(opt; track_numbers = AbstractFloat),
        Utils.to_rarray(state; track_numbers = AbstractFloat),
    )
end

function make_reactant_compatible(opt::Optimisers.AccumGrad, state)
    return (
        AccumGrad(Utils.to_rarray(opt.n; track_numbers = Integer)),
        Utils.to_rarray(state; track_numbers = Integer),
    )
end

@concrete struct AccumGrad <: AbstractRule
    n
end

# XXX: the counter needs to match the client / device?
Optimisers.init(::AccumGrad, x) = zero(x), Utils.to_rarray(1; track_numbers = Integer)

end
