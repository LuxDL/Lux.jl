using InteractiveUtils, Hwloc, ReTestItems, LuxTestUtils

@info sprint(versioninfo)

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Hwloc.num_physical_cores(), 16))))

ReTestItems.runtests(LuxTestUtils; nworkers=RETESTITEMS_NWORKERS)
