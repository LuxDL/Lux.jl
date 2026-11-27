using LuxTestUtils: check_approx

include("../shared_testsetup.jl")
include("recurrence_testsetup.jl")

@testset "Recurrence: GRUCell" begin
    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        @testset for ordering in (BatchLastIndex(), TimeLastIndex())
            test_recurrence_layer(
                mode, aType, dev, ongpu, ordering, GRUCell, false, true
            )
        end
    end
end
