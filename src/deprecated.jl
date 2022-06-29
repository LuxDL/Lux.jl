# Deprecations for v0.5

## Device transfer of AbstractExplicitLayers
function cpu(l::AbstractExplicitLayer)
    Base.depwarn("`cpu` on a layer has been deprecated and will be removed in v0.5. Apply" *
                 "`cpu` on the layer's parameters and states instead.", :cpu)
    return l
end

function gpu(l::AbstractExplicitLayer)
    Base.depwarn("`gpu` on a layer has been deprecated and will be removed in v0.5. Apply" *
                 "`gpu` on the layer's parameters and states instead.", :gpu)
    return l
end

## Trainmode/Testmode with argument
function trainmode(st::NamedTuple, mode::Bool)
    Base.depwarn("Setting `mode` for `trainmode` is deprecated and will be removed in v0.5.",
                 :trainmode)
    return mode ? trainmode(st) : testmode(st)
end

function testmode(st::NamedTuple, mode::Bool)
    Base.depwarn("Setting `mode` for testmode is deprecated and will be removed in v0.5",
                 :testmode)
    return mode ? testmode(st) : trainmode(st)
end

## Fallback `initialparameters` / `initialstates`
function initialparameters(::AbstractRNG, l::Any)
    Base.depwarn("Default fallback for non `AbstractExplicitLayer` types are deprecated" *
                 "and will be removed in v0.5. Define" *
                 " `Lux.initialparameters(::AbstractRNG, ::$(typeof(l)))`",
                 :initialparameters)
    return NamedTuple()
end

function initialstates(::AbstractRNG, l::Any)
    Base.depwarn("Default fallback for non `AbstractExplicitLayer` types are deprecated" *
                 "and will be removed in v0.5. Define" *
                 " `Lux.initialstates(::AbstractRNG, ::$(typeof(l)))`",
                 :initialstates)
    return NamedTuple()
end

## Fallback `parameterlength` / `statelength`
function parameterlength(x::Any)
    Base.depwarn("Fallback for `parameterlength` of type $(typeof(x)) is deprecated." *
                 " This will generate an error from v0.5.",
                 :parameterlength)
    return 0
end

function statelength(x::Any)
    Base.depwarn("Fallback for `statelength` of type $(typeof(x)) is deprecated." *
                 " This will generate an error from v0.5.",
                 :statelength)
    return 0
end
