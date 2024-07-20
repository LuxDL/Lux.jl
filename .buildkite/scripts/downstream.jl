using Pkg

repo = ARGS[1]
if contains(repo, "#")
    repo, group = split(repo, "#")
else
    group = ARGS[2]
end

println("--- :julia: Instantiating project")
withenv("JULIA_PKG_PRECOMPILE_AUTO" => 0, "GROUP" => group, "BACKEND_GROUP" => group) do
    Pkg.instantiate()

    try
        Pkg.develop(repo)
        println("+++ :julia: Running tests")
        Pkg.test("$(repo)"; coverage="user")
    catch err
        err isa Pkg.Resolve.ResolverError || rethrow()
        @info "Not compatible with this release. No problem." exception=err
        exit(0)
    end
end

println("+++ :julia: Finished Downstream Test")
