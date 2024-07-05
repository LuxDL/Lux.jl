using ReTestItems

const LUXLIB_TEST_GROUP = get(ENV, "LUXLIB_TEST_GROUP", "all")
@info "Running tests for group: $LUXLIB_TEST_GROUP"

if LUXLIB_TEST_GROUP == "all"
    ReTestItems.runtests(@__DIR__)
else
    ReTestItems.runtests(@__DIR__; tags=[Symbol(LUXLIB_TEST_GROUP)])
end
