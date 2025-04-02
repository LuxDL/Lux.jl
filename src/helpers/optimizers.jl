# We use this module mostly as a placeholder for patches that should be merged into
# Optimisers.jl for Reactant compatibility.
module ReactantCompatibleOptimisers

using Optimisers: Optimisers

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

function optimisers_setup_with_jit end

end
