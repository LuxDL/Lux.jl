# We use this module mostly as a placeholder for patches that should be merged into
# Optimisers.jl for Reactant compatibility.
module ReactantCompatibleOptimisers

using Optimisers: Optimisers
using MLDataDevices: ReactantDevice, get_device, with_track_numbers

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
    dev = with_track_numbers(get_device(opt), AbstractFloat)
    return ReactantOptimiser(Optimisers._adjust(opt.opt, dev(nt)))
end

function Optimisers._adjust(opt::ReactantOptimiser{<:Optimisers.AccumGrad}, nt::NamedTuple)
    dev = with_track_numbers(get_device(opt), Integer)
    return ReactantOptimiser(Optimisers._adjust(opt.opt, dev(nt_tracked)))
end

# Convert existing Optimisers.jl rules to ReactantOptimisers
function make_reactant_compatible(
    opt::Optimisers.OptimiserChain, dev::ReactantDevice, outermost=Val(true)
)
    opt_ra = Optimisers.OptimiserChain(
        make_reactant_compatible.(opt.opts, (dev,), Val(false))...
    )
    outermost isa Val{true} && return ReactantOptimiser(opt_ra)
    return opt_ra
end

function make_reactant_compatible(
    opt::Optimisers.AbstractRule, dev::ReactantDevice, outermost=Val(true)
)
    opt_ra = with_track_numbers(dev, AbstractFloat)(opt)
    outermost isa Val{true} && return ReactantOptimiser(opt_ra)
    return opt_ra
end

function make_reactant_compatible(
    opt::Optimisers.AccumGrad, dev::ReactantDevice, outermost=Val(true)
)
    # ignore outermost we will need to update the fields
    return ReactantOptimiser(with_track_numbers(dev, Integer)(opt))
end

function make_reactant_compatible(
    opt::Optimisers.ClipNorm, dev::ReactantDevice, outermost=Val(true)
)
    opt_ra = Optimisers.ClipNorm(with_track_numbers(dev, Integer)(opt.omega), opt.p, false)
    outermost isa Val{true} && return ReactantOptimiser(opt_ra)
    return opt_ra
end

function optimisers_setup_with_jit end

end
