@testitem "Dynamic Expressions" setup=[SharedTestSetup] tags=[:others] begin
    using DynamicExpressions, ForwardDiff, ComponentArrays, Bumper

    operators = OperatorEnum(; binary_operators=[+, -, *], unary_operators=[cos])

    x1 = Node(; feature=1)
    x2 = Node(; feature=2)

    expr_1 = x1 * cos(x2 - 3.2)
    expr_2 = x2 - x1 * x2 + 2.5 - 1.0 * x1

    for exprs in ((expr_1,), (expr_1, expr_2), ([expr_1, expr_2],)),
        turbo in (Val(false), Val(true)),
        bumper in (Val(false), Val(true))

        layer = DynamicExpressionsLayer(operators, exprs...; turbo, bumper)
        ps, st = Lux.setup(Random.default_rng(), layer)

        x = [1.0f0 2.0f0 3.0f0
             4.0f0 5.0f0 6.0f0]

        y, st_ = layer(x, ps, st)
        @test eltype(y) == Float32
        __f = (x, p) -> sum(abs2, first(layer(x, p, st)))
        test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3, skip_backends=[AutoEnzyme()])

        # Particular ForwardDiff dispatches
        ps_ca = ComponentArray(ps)
        dps_ca = ForwardDiff.gradient(ps_ca) do ps_
            sum(abs2, first(layer(x, ps_, st)))
        end
        dx = ForwardDiff.gradient(x) do x_
            sum(abs2, first(layer(x_, ps, st)))
        end
        dxps = ForwardDiff.gradient(ComponentArray(; x=x, ps=ps)) do ca
            sum(abs2, first(layer(ca.x, ca.ps, st)))
        end

        @test dx≈dxps.x atol=1.0f-3 rtol=1.0f-3
        @test dps_ca≈dxps.ps atol=1.0f-3 rtol=1.0f-3

        x = Float64.(x)
        y, st_ = layer(x, ps, st)
        @test eltype(y) == Float64
        __f = (x, p) -> sum(abs2, first(layer(x, p, st)))
        test_gradients(__f, x, ps; atol=1.0e-3, rtol=1.0e-3, skip_backends=[AutoEnzyme()])
    end

    @testset "$(mode)" for (mode, aType, dev, ongpu) in MODES
        layer = DynamicExpressionsLayer(operators, expr_1)
        ps, st = Lux.setup(Random.default_rng(), layer) |> dev

        x = [1.0f0 2.0f0 3.0f0
             4.0f0 5.0f0 6.0f0] |> aType

        if ongpu
            @test_throws ArgumentError layer(x, ps, st)
        end
    end
end
