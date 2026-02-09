using Lux, Documenter

# Some of the tests are flaky on prereleases
@testset "doctests: Quality Assurance" begin
    doctestexpr = :(using Adapt, Lux, Random, Optimisers, Zygote, NNlib)

    DocMeta.setdocmeta!(Lux, :DocTestSetup, doctestexpr; recursive=true)
    doctest(Lux; manual=false)
end
