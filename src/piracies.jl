# I know it is bad to have type-piracy. We are in the process of moving these to the correct
# places. We will delete them from here once we are done.

# Base + ChainRules.jl type-piracy
CRC.@non_differentiable Base.printstyled(::Any...)
CRC.@non_differentiable fieldcount(::Any)

# LuxCore type-piracy
# getproperty rrule for AbstractExplicitLayer. needed for type stability of Zygote
# gradients.
function CRC.rrule(::typeof(getproperty), m::AbstractExplicitLayer, name::Symbol)
    res = getproperty(m, name)
    ∇getproperty = Δ -> ntuple(Returns(NoTangent()), 3)
    return res, ∇getproperty
end
