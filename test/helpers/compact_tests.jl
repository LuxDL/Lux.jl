@testitem "@compact" setup=[SharedTestSetup] tags=[:helpers] begin
    using ComponentArrays, Zygote

    rng = StableRNG(12345)

    function similar_strings(s₁::String, s₂::String)
        if s₁ != s₂
            println(stderr, "s₁: ", s₁)
            println(stderr, "s₂: ", s₂)
        end
        return s₁ == s₂
    end

    function get_model_string(model)
        io = IOBuffer()
        show(io, MIME"text/plain"(), model)
        return String(take!(io))
    end

    @testset "$mode: @compact" for (mode, aType, device, ongpu) in MODES
        @testset "Linear Layer" begin
            r = @compact(w=[1, 5, 10]) do x
                @return sum(w .* x)
            end
            ps, st = Lux.setup(rng, r) |> device

            @test ps.w == ([1, 5, 10] |> device)
            @test st == NamedTuple()

            x = [1, 1, 1] |> aType
            @test first(r(x, ps, st)) == 1 + 5 + 10

            x = [1, 2, 3] |> aType
            @test first(r(x, ps, st)) == 1 + 2 * 5 + 3 * 10

            x = ones(3, 3) |> aType
            @test first(r(x, ps, st)) == 3 * (1 + 5 + 10)

            @jet r(x, ps, st)

            # Test gradients:
            x = [1, 1, 1] |> aType
            @test Zygote.gradient(x -> sum(first(r(x, ps, st))), x)[1] == ps.w
        end

        @testset "Linear Layer with Activation" begin
            d_in = 5
            d_out = 7
            d = @compact(W=randn(d_out, d_in), b=zeros(d_out), act=relu) do x
                y = W * x
                @return act.(y .+ b)
            end

            ps, st = Lux.setup(rng, d) |> device
            @test size(ps.W) == (7, 5)
            @test size(ps.b) == (7,)
            @test st.act == relu

            x = ones(5, 10) |> aType
            @test size(first(d(x, ps, st))) == (7, 10)

            x = randn(rng, 5, 10) |> aType
            @test all(≥(0), first(d(x, ps, st)))

            @jet d(x, ps, st)

            # Test gradients:
            y, ∇ = Zygote.withgradient(ps) do ps
                input = randn(5, 32) |> aType
                desired_output = randn(7, 32) |> aType
                prediction = first(d(input, ps, st))
                return sum(abs2, prediction .- desired_output)
            end

            @test y isa AbstractFloat
            grads = ∇[1]
            @test length(grads) == 2
            @test Set(size.(values(grads))) == Set([(7, 5), (7,)])

            # Test equivalence to Dense layer:
            ps_dense = (; weight=ps.W, bias=ps.b)
            st_dense = NamedTuple()
            dense = Dense(d_in => d_out, relu)

            x = [1, 2, 3, 4, 5] |> aType
            @test first(d(x, ps, st)) ≈ first(dense(x, ps_dense, st_dense))
        end

        @testset "MLP" begin
            n_in = 1
            n_out = 1
            nlayers = 3

            model = @compact(w1=Dense(n_in, 128), w2=[Dense(128, 128) for i in 1:nlayers],
                w3=Dense(128, n_out), act=relu) do x
                embed = act.(w1(x))
                for w in w2
                    embed = act.(w(embed))
                end
                out = w3(embed)
                @return out
            end

            ps, st = Lux.setup(rng, model) |> device

            @test size(ps.w1.weight) == (128, 1)
            @test size(ps.w1.bias) == (128, 1)
            @test length(ps.w2) == nlayers
            for i in 1:nlayers
                @test size(ps.w2[i].weight) == (128, 128)
                @test size(ps.w2[i].bias) == (128, 1)
            end
            @test size(ps.w3.weight) == (1, 128)
            @test size(ps.w3.bias) == (1, 1)

            x = randn(n_in, 32) |> aType

            @test size(first(model(x, ps, st))) == (1, 32)

            ps2 = ps |> cpu_device() |> ComponentArray |> device

            @test size(first(model(x, ps2, st))) == (1, 32)

            @jet model(x, ps, st)

            __f = (x, ps) -> sum(first(model(x, ps, st)))

            @eval @test_gradients $__f $x $ps gpu_testing=$ongpu atol=1.0f-3 rtol=1.0f-3
        end

        @testset "String Representations" begin
            model = @compact(w=Dense(32 => 32)) do (x, y)
                tmp = sum(w(x))
                @return tmp + y
            end
            expected_string = """@compact(
                w = Dense(32 => 32),                # 1_056 parameters
            ) do (x, y) 
                tmp = sum(w(x))
                return tmp + y
            end       # Total: 1_056 parameters,
                      #        plus 0 states."""

            @test similar_strings(get_model_string(model), expected_string)
        end

        @testset "Custom Naming" begin
            model = @compact(w=Dense(32, 32), name="Linear(...)") do (x, y)
                tmp = sum(w(x))
                @return tmp + y
            end
            expected_string = "Linear(...)         # 1_056 parameters"
            @test similar_strings(get_model_string(model), expected_string)
        end

        @testset "Hierarchical Models" begin
            model1 = @compact(w1=Dense(32 => 32, relu), w2=Dense(32 => 32, relu)) do x
                @return w2(w1(x))
            end
            model2 = @compact(w1=model1, w2=Dense(32 => 32, relu)) do x
                @return w2(w1(x))
            end
            expected_string = """@compact(
                w1 = @compact(
                    w1 = Dense(32 => 32, relu),     # 1_056 parameters
                    w2 = Dense(32 => 32, relu),     # 1_056 parameters
                ) do x 
                    return w2(w1(x))
                end,
                w2 = Dense(32 => 32, relu),         # 1_056 parameters
            ) do x 
                return w2(w1(x))
            end       # Total: 3_168 parameters,
                      #        plus 0 states."""
            @test similar_strings(get_model_string(model2), expected_string)
        end

        @testset "Array Parameters" begin
            model = @compact(x=randn(32), w=Dense(32 => 32)) do s
                @return w(x .* s)
            end
            expected_string = """@compact(
                x = 32-element Vector{Float64},
                w = Dense(32 => 32),                # 1_056 parameters
            ) do s 
                return w(x .* s)
            end       # Total: 1_088 parameters,
                      #        plus 0 states."""
            @test similar_strings(get_model_string(model), expected_string)
        end

        @testset "Iterable Inputs" begin
            w1 = randn(rng, 32, 32)
            w2 = randn(rng, 32, 32)

            for x_list in ((w1, w2), [w1, w2])
                model = @compact(x=x_list) do s
                    @return x[2] * (x[1] * s)
                end

                ps, st = Lux.setup(rng, model)
                x = randn(Float32, 32, 2)

                @test ps.x[1] === w1
                @test ps.x[2] === w2

                y, _ = model(x, ps, st)
                @test y ≈ w2 * (w1 * x)
            end

            model = @compact(x=[Dense(32 => 32), Dense(32 => 32)]) do s
                @return x[2](x[1](s))
            end

            ps, st = Lux.setup(rng, model)
            x = randn(Float32, 32, 2)

            y, _ = model(x, ps, st)
            ŷ = ps.x[2].weight * (ps.x[1].weight * x .+ ps.x[1].bias) .+ ps.x[2].bias

            @test y ≈ ŷ
        end

        @testset "Function kwarg" begin
            model = @compact(; f=abs2) do x
                @return f.(x)
            end

            ps, st = Lux.setup(rng, model)
            x = randn(Float32, 32, 2)

            y, _ = model(x, ps, st)
            ŷ = abs2.(x)

            @test y ≈ ŷ
        end

        @testset "Hierarchy with Inner Model Named" begin
            model = @compact(w1=@compact(w1=randn(32, 32), name="Model(32)") do x
                    @return w1 * x
                end, w2=randn(32, 32), w3=randn(32),) do x
                @return w2 * w1(x)
            end
            expected_string = """@compact(
                w1 = Model(32),                     # 1_024 parameters
                w2 = 32×32 Matrix{Float64},
                w3 = 32-element Vector{Float64},
            ) do x 
                return w2 * w1(x)
            end       # Total: 2_080 parameters,
                      #        plus 0 states."""
            @test similar_strings(get_model_string(model), expected_string)
        end

        @testset "Hierarchy with Outer Model Named" begin
            model = @compact(w1=@compact(w1=randn(32, 32)) do x
                    @return w1 * x
                end, w2=randn(32, 32), w3=randn(32), name="Model(32)") do x
                @return w2 * w1(x)
            end
            expected_string = """Model(32)           # 2_080 parameters"""
            @test similar_strings(get_model_string(model), expected_string)
        end

        @testset "Keyword Argument Syntax" begin
            _a = 3
            _b = 4
            c = 5
            model = @compact(a=_a; b=_b, c) do x
                @return a + b * x + c * x^2
            end
            ps, st = Lux.setup(rng, model) |> device
            @test first(model(2, ps, st)) == _a + _b * 2 + c * 2^2
        end

        @testset "Keyword Arguments with Anonymous Function" begin
            model = @compact(x->@return(x+a+b); a=1, b=2)
            ps, st = Lux.setup(rng, model) |> device
            @test first(model(3, ps, st)) == 1 + 2 + 3
            expected_string = """@compact(
                a = 1,
                b = 2,
            ) do x 
                return x + a + b
            end       # Total: 0 parameters,
                      #        plus 2 states."""
            @test similar_strings(get_model_string(model), expected_string)
        end

        @testset "kwarg printing" begin
            model = @compact(; a=1, b=2, z=(; a=3, v=1, k=3, w=2), c=4) do x
                @return x + a + b
            end
            expected_string = """@compact(
                a = 1,
                b = 2,
                z = @NamedTuple{a = 3, v = 1, k = 3, ...},
                c = 4,
            ) do x 
                return x + a + b
            end       # Total: 0 parameters,
                      #        plus 7 states."""
            @test similar_strings(get_model_string(model), expected_string)
        end

        @testset "Scoping of Parameter Arguments" begin
            model = @compact(w1=3, w2=5) do a
                g(w1, w2) = 2 * w1 * w2
                @return (w1 + w2) * g(a, a)
            end
            ps, st = Lux.setup(rng, model) |> device
            @test first(model(2, ps, st)) == (3 + 5) * 2 * 2 * 2
        end

        @testset "Updated State" begin
            function ScaledDense1(; d_in=5, d_out=7, act=relu)
                @compact(W=randn(d_out, d_in), b=zeros(d_out), incr=1) do x
                    y = W * x
                    incr *= 10
                    return act.(y .+ b) .+ incr
                end
            end

            model = ScaledDense1()
            ps, st = Lux.setup(Xoshiro(0), model) |> device
            x = ones(5, 10) |> aType

            @test st.incr == 1
            _, st_new = model(x, ps, st)
            @test st_new.incr == 10
            _, st_new = model(x, ps, st_new)
            @test st_new.incr == 100

            # By default creates a closure so type cannot be inferred
            inf_type = Core.Compiler._return_type(
                model, Tuple{typeof(x), typeof(ps), typeof(st)}).parameters
            @test inf_type[1] === Any
            @test inf_type[2] === NamedTuple

            function ScaledDense2(; d_in=5, d_out=7, act=relu)
                @compact(W=randn(d_out, d_in), b=zeros(d_out), incr=1) do x
                    y = W * x
                    incr *= 10
                    @return act.(y .+ b) .+ incr
                end
            end

            model = ScaledDense2()
            ps, st = Lux.setup(Xoshiro(0), model) |> device
            x = ones(5, 10) |> aType

            @test st.incr == 1
            _, st_new = model(x, ps, st)
            @test st_new.incr == 10
            _, st_new = model(x, ps, st_new)
            @test st_new.incr == 100

            @inferred model(x, ps, st)

            __f = (m, x, ps, st) -> sum(abs2, first(m(x, ps, st)))
            @inferred Zygote.gradient(__f, model, x, ps, st)
        end

        @testset "Multiple @return" begin
            model = @compact(; a=1) do x
                if x > 0
                    a += 1
                    @return x
                end
                a -= 1
                @return -1
            end
            ps, st = Lux.setup(rng, model)

            @test st.a == 1

            y1, st_ = model(2.0, ps, st)
            @test y1 == 2
            @test st_.a == 2

            ∂x1 = only(Zygote.gradient(x -> sum(first(model(x, ps, st))), 2.0))
            @test ∂x1 == 1

            y2, st_ = model(-2.0, ps, st)
            @test y2 == -1
            @test st_.a == 0

            ∂x2 = only(Zygote.gradient(x -> sum(first(model(x, ps, st))), -2.0))
            @test ∂x2 === nothing
        end
    end
end

@testitem "@compact error checks" setup=[SharedTestSetup] tags=[:helpers] begin
    showerror(stdout, Lux.LuxCompactModelParsingException(""))
    println()

    # Test that initialization lines cannot depend on each other
    @test_throws UndefVarError @compact(y₁=3, z=y₁^2) do x
        @return y₁ + z + x
    end

    @test_throws Lux.LuxCompactModelParsingException("expects at least two expressions: a function and at least one keyword") @macroexpand @compact()

    @test_throws Lux.LuxCompactModelParsingException("expects an anonymous function") @macroexpand @compact(;
        a=1)

    @test_throws Lux.LuxCompactModelParsingException("expects only keyword arguments") @macroexpand @compact(2;
        a=1) do x
        @return x + a
    end

    @test_throws Lux.LuxCompactModelParsingException("Encountered a return statement after the last @return statement. This is not supported.") @macroexpand @compact(;
        a=1) do x
        @return x
        return 1
    end

    @test_throws ArgumentError Lux.ValueStorage()(1, 1, 1)

    @test_throws Lux.LuxCompactModelParsingException("A container `x = (var\"1\" = Dense(2 => 3), var\"2\" = 1)` is found which combines Lux layers with non-Lux layers. This is not supported.") @compact(;
        x=(Dense(2 => 3), 1)) do y
        @return x[1](x[2] .* y)
    end
end
