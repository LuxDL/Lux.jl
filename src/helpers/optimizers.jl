# We use this module mostly as a placeholder for patches that should be merged into
# Optimisers.jl for Reactant compatibility.
module ReactantCompatibleOptimisers

using Optimisers: Optimisers

using ..Lux: Lux, Utils
using MLDataDevices: ReactantDevice, get_device

function _dev_to_kwargs(dev::ReactantDevice)
    kwargs = (;
        client=dev.client === missing ? nothing : dev.client,
        device=dev.device === missing ? nothing : dev.device,
    )
    dev.sharding isa IdDict || (kwargs = (; kwargs..., sharding=dev.sharding))
    return kwargs
end

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
    nt_tracked = Utils.to_rarray(
        nt; track_numbers=AbstractFloat, _dev_to_kwargs(get_device(opt))...
    )
    return ReactantOptimiser(Optimisers._adjust(opt.opt, nt_tracked))
end

function Optimisers._adjust(opt::ReactantOptimiser{<:Optimisers.AccumGrad}, nt::NamedTuple)
    nt_tracked = Utils.to_rarray(
        nt; track_numbers=Integer, _dev_to_kwargs(get_device(opt))...
    )
    return ReactantOptimiser(Optimisers._adjust(opt.opt, nt_tracked))
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
    opt_ra = Utils.to_rarray(opt; track_numbers=AbstractFloat, _dev_to_kwargs(dev)...)
    outermost isa Val{true} && return ReactantOptimiser(opt_ra)
    return opt_ra
end

function make_reactant_compatible(
    opt::Optimisers.AccumGrad, dev::ReactantDevice, outermost=Val(true)
)
    # ignore outermost we will need to update the fields
    return ReactantOptimiser(
        Utils.to_rarray(opt; track_numbers=Integer, _dev_to_kwargs(dev)...)
    )
end

function optimisers_setup_with_jit end

end
