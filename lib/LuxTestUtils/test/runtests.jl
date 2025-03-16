using InteractiveUtils, CPUSummary, ReTestItems, LuxTestUtils

@info sprint(versioninfo)

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Int(CPUSummary.num_cores()), 16)))
)

ReTestItems.runtests(LuxTestUtils; nworkers=RETESTITEMS_NWORKERS)
