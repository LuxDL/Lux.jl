using ReTestItems

const LUXLIB_TEST_GROUP = get(ENV, "LUXLIB_TEST_GROUP", "all")
@info "Running tests for group: $LUXLIB_TEST_GROUP"

if LUXLIB_TEST_GROUP == "all"
    # Instance Normalization Tests causes stalling on CUDA CI
    ReTestItems.runtests(@__DIR__; nworkers=0, tags=[:singleworker])

    ReTestItems.runtests(@__DIR__; tags=[:nworkers])
else
    tag = Symbol(LUXLIB_TEST_GROUP)
    # Instance Normalization Tests causes stalling on CUDA CI
    ReTestItems.runtests(@__DIR__; nworkers=0, tags=[:singleworker, tag])

    ReTestItems.runtests(@__DIR__; tags=[:nworkers, tag])
end
