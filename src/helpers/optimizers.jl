# We use this module mostly as a placeholder for patches that should be merged into
# Optimisers.jl for Reactant compatibility.
module ReactantCompatibleOptimisers

using Optimisers: Optimisers

using ..Lux: Lux, Utils

# We need to wrap in a "ReactantOptimiser" to correctly update the learning rate as such
# without accidentally making them constants
struct ReactantOptimiser{T} <: Optimisers.AbstractRule
    opt::T
end

function Optimisers.apply!(opt::ReactantOptimiser, state, x, y)
    return Optimisers.apply!(opt.opt, state, x, y)
end

Optimisers.init(opt::ReactantOptimiser, ps) = Optimisers.init(opt.opt, ps)

function Optimisers._adjust(opt::ReactantOptimiser, nt::NamedTuple)
    nt_tracked = Utils.to_rarray(nt; track_numbers=AbstractFloat)
    return ReactantOptimiser(Optimisers._adjust(opt.opt, nt_tracked))
end

function Optimisers._adjust(opt::ReactantOptimiser{<:Optimisers.AccumGrad}, nt::NamedTuple)
    nt_tracked = Utils.to_rarray(nt; track_numbers=Integer)
    return ReactantOptimiser(Optimisers._adjust(opt.opt, nt_tracked))
end

# Convert existing Optimisers.jl rules to ReactantOptimisers
function make_reactant_compatible(opt::Optimisers.OptimiserChain, outermost=Val(true))
    opt_ra = Optimisers.OptimiserChain(make_reactant_compatible.(opt.opts, Val(false))...)
    outermost isa Val{true} && return ReactantOptimiser(opt_ra)
    return opt_ra
end

function make_reactant_compatible(opt::Optimisers.AbstractRule, outermost=Val(true))
    opt_ra = Utils.to_rarray(opt; track_numbers=AbstractFloat)
    outermost isa Val{true} && return ReactantOptimiser(opt_ra)
    return opt_ra
end

function make_reactant_compatible(opt::Optimisers.AccumGrad, outermost=Val(true))
    # ignore outermost we will need to update the fields
    return ReactantOptimiser(Utils.to_rarray(opt; track_numbers=Integer))
end

function optimisers_setup_with_jit end

end
