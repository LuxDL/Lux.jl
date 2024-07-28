using InteractiveUtils, Hwloc, ReTestItems

@info sprint(io -> versioninfo(io; verbose=true))

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Hwloc.num_physical_cores(), 16))))

ReTestItems.runtests(@__DIR__; nworkers=RETESTITEMS_NWORKERS)
