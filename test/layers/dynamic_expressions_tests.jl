@testitem "Dynamic Expressions" setup=[SharedTestSetup] tags=[:others] begin
    using DynamicExpressions

    operators = OperatorEnum(; binary_operators=[+, -, *], unary_operators=[cos])

    x1 = Node(; feature=1)
    x2 = Node(; feature=2)

    expr_1 = x1 * cos(x2 - 3.2)
    expr_2 = x2 - x1 * x2 + 2.5 - 1.0 * x1

    for exprs in ((expr_1,), (expr_1, expr_2), ([expr_1, expr_2],))
        layer = DynamicExpressionsLayer(operators, exprs...)
        ps, st = Lux.setup(Random.default_rng(), layer)

        x = [1.0f0 2.0f0 3.0f0
             4.0f0 5.0f0 6.0f0]

        y, st_ = layer(x, ps, st)
        @test eltype(y) == Float32
        __f = (x, p) -> sum(abs2, first(layer(x, p, st)))
        @test_gradients __f x ps atol=1.0f-3 rtol=1.0f-3

        x = Float64.(x)
        y, st_ = layer(x, ps, st)
        @test eltype(y) == Float64
        __f = (x, p) -> sum(abs2, first(layer(x, p, st)))
        @test_gradients __f x ps atol=1.0e-3 rtol=1.0e-3
    end
end
