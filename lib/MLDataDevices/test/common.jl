using Pkg: Pkg
using Test: @test, @testset

macro test_in_separate_process(testname, expr)
    tmpfile = tempname() * ".jl"
    open(tmpfile, "w") do io
        println(io, "using Pkg, MLDataDevices, Test")
        println(io, expr)
    end
    project_path = dirname(Pkg.project().path)

    run_cmd = `$(Base.julia_cmd()) --color=yes --project=$(project_path) --startup-file=no --code-coverage=user $(tmpfile)`

    return quote
        @testset $(testname) begin
            try
                run($run_cmd)
                @test true
            catch
                @test false
            end
        end
    end
end
