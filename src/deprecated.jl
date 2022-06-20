# Deprecations for v0.5

## Device transfer of AbstractExplicitLayers
function cpu(::AbstractExplicitLayer)
    @warn "Applying `cpu` on a layer is an invalid operation and will be deprecated in v0.5. Instead apply it on the parameters and states returned by `Lux.setup`." maxlog=1
end

function gpu(::AbstractExplicitLayer)
    @warn "Applying `gpu` on a layer is an invalid operation and will be deprecated in v0.5. Instead apply it on the parameters and states returned by `Lux.setup`." maxlog=1
end