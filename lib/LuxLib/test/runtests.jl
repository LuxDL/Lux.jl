using ReTestItems

# Instance Normalization Tests causes stalling on CUDA CI
ReTestItems.runtests(@__DIR__; nworkers=0, tags=[:singleworker])

ReTestItems.runtests(@__DIR__; tags=[:nworkers])
