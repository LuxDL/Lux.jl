# We use this module mostly as a placeholder for patches that should be merged into
# Optimisers.jl for Reactant compatibility.
module ReactantCompatibleOptimisers

using ..Lux: Utils

using Optimisers: Optimisers
using MLDataDevices: ReactantDevice, get_device, with_track_numbers

# We need to wrap in a "ReactantOptimiser" to correctly update the learning rate as such
# without accidentally making them constants
struct ReactantOptimiser{T} <: Optimisers.AbstractRule
    opt::T
end

function Base.show(io::IO, opt::ReactantOptimiser)
    print(io, "ReactantOptimiser(", opt.opt, ")")
    return nothing
end

function Optimisers.apply!(opt::ReactantOptimiser, state, x, y)
    return Optimisers.apply!(opt.opt, state, x, y)
end

Optimisers.init(opt::ReactantOptimiser, ps) = Optimisers.init(opt.opt, ps)

# JITing this is not great atm, since it causes issues without implementing result sharding
# annotatations
for common_opt in (:Adam, :AdaMax, :NAdam, :AdamW, :AdaBelief)
    @eval function Optimisers.init(
        opt::ReactantOptimiser{<:Optimisers.$(common_opt)}, x::AbstractArray{T}
    ) where {T}
        return zero(x), zero(x), Utils.convert_eltype.((T,), opt.opt.beta)
    end
end

function Optimisers.init(
    opt::ReactantOptimiser{<:Optimisers.RAdam}, x::AbstractArray{T}
) where {T}
    dev = with_track_numbers(get_device(opt), Integer)
    return (zero(x), zero(x), Utils.convert_eltype.((T,), opt.opt.beta), dev(1))
end

function Optimisers.init(
    opt::ReactantOptimiser{<:Optimisers.OAdam}, x::AbstractArray{T}
) where {T}
    return zero(x), zero(x), Utils.convert_eltype.((T,), opt.opt.beta), zero(x)
end

function Optimisers._adjust(opt::ReactantOptimiser, nt::NamedTuple)
    dev = with_track_numbers(get_device(opt), AbstractFloat)
    return ReactantOptimiser(Optimisers._adjust(opt.opt, dev(nt)))
end

function Optimisers._adjust(opt::ReactantOptimiser{<:Optimisers.AccumGrad}, nt::NamedTuple)
    dev = with_track_numbers(get_device(opt), Integer)
    return ReactantOptimiser(Optimisers._adjust(opt.opt, dev(nt)))
end

# Convert existing Optimisers.jl rules to ReactantOptimisers
function make_reactant_compatible(opt::Optimisers.OptimiserChain, dev::ReactantDevice)
    return ReactantOptimiser(
        Optimisers.OptimiserChain(make_reactant_compatible.(opt.opts, (dev,))...)
    )
end

function make_reactant_compatible(opt::Optimisers.AbstractRule, dev::ReactantDevice)
    return ReactantOptimiser(with_track_numbers(dev, AbstractFloat)(opt))
end

function make_reactant_compatible(opt::Optimisers.AccumGrad, dev::ReactantDevice)
    return ReactantOptimiser(with_track_numbers(dev, Integer)(opt))
end

function make_reactant_compatible(opt::Optimisers.ClipNorm, dev::ReactantDevice)
    return ReactantOptimiser(
        Optimisers.ClipNorm(with_track_numbers(dev, Integer)(opt.omega), opt.p, false)
    )
end

end
