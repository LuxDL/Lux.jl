# This is mostly an internal implementation detail that users shouldn't need to worry about.
# We can remove this once https://github.com/FluxML/Optimisers.jl/issues/205 is resolved.
module ReactantCompatibleOptimisers

using Optimisers: Optimisers, AbstractRule

using ..Lux: Lux, Utils

function make_reactant_compatible(opt::Optimisers.OptimiserChain)
    return Optimisers.OptimiserChain(make_reactant_compatible.(opt.opts)...)
end

function make_reactant_compatible(opt::Optimisers.AbstractRule)
    return Utils.to_rarray(opt; track_numbers=AbstractFloat)
end

function make_reactant_compatible(opt::Optimisers.AccumGrad)
    return Utils.to_rarray(opt; track_numbers=Integer)
end

end
