# Recursive Operations --> Functors.fmap
@deprecate recursive_add!!(x, y) Functors.fmap(
    Utils.add!!, x, y; exclude=MLDataDevices.isleaf
)
@deprecate recursive_make_zero(x) Functors.fmap(Utils.zero, x; exclude=MLDataDevices.isleaf)
@deprecate recursive_make_zero!!(x) Functors.fmap(
    Utils.zero!!, x; exclude=MLDataDevices.isleaf
)
@deprecate recursive_copyto!(x, y) Functors.fmap(
    copyto!, x, y; exclude=MLDataDevices.isleaf
)
@deprecate recursive_map(f, args...) Functors.fmap(f, args...; exclude=MLDataDevices.isleaf)
