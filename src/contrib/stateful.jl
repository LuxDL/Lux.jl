# Deprecated
function StatefulLuxLayer(args...; kwargs...)
    Base.depwarn(
        "Lux.Experimental.StatefulLuxLayer` has been promoted out of `Lux.Experimental` \
         and is now available in `Lux`. In other words this has been deprecated and will \
         be removed in v0.6. Use `Lux.StatefulLuxLayer` instead.",
        :StatefulLuxLayer)
    return Lux.StatefulLuxLayer(args...; kwargs...)
end
