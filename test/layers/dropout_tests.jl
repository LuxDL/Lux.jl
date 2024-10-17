@testitem "Dropout" setup=[SharedTestSetup] tags=[:normalize_layers] begin
    rng=StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        for p in (0.5f0, 0.5)
            layer = Dropout(p)
            display(layer)
            ps, st = Lux.setup(rng, layer) .|> dev
            x = randn(Float32, 5, 2) |> aType

            x_, st_ = layer(x, ps, st)
            x__, st__ = layer(x, ps, st)
            x___, st___ = layer(x_, ps, st_)

            @test st_.rng != st.rng
            @test st_.rng == st__.rng
            @test x_ == x__
            @test x_ != x___

            @jet layer(x, ps, st)
            __f = let layer = layer, ps = ps, st = st
                x -> sum(first(layer(x, ps, st)))
            end
            @test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)

            st = Lux.testmode(st)

            @test first(layer(x, ps, st)) == x
        end
    end
end

@testitem "AlphaDropout" setup=[SharedTestSetup] tags=[:normalize_layers] begin
    rng=StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        for p in (0.5f0, 0.5)
            layer = AlphaDropout(p)
            display(layer)
            ps, st = Lux.setup(rng, layer) .|> dev
            # GPU compilation for mixed types fail atm
            x = randn(typeof(p), 5, 2) |> aType

            x_, st_ = layer(x, ps, st)
            x__, st__ = layer(x, ps, st)
            x___, st___ = layer(x_, ps, st_)

            @test st_.rng != st.rng
            @test st_.rng == st__.rng
            @test x_ == x__
            @test x_ != x___

            @jet layer(x, ps, st)
            __f = let layer = layer, ps = ps, st = st
                x -> sum(first(layer(x, ps, st)))
            end
            @test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)

            st = Lux.testmode(st)

            @test first(layer(x, ps, st)) == x
        end
    end
end

@testitem "VariationalHiddenDropout" setup=[SharedTestSetup] tags=[:normalize_layers] begin
    rng=StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        for p in (0.5f0, 0.5)
            layer = VariationalHiddenDropout(p)
            display(layer)
            ps, st = Lux.setup(rng, layer) .|> dev
            x = randn(Float32, 5, 2) |> aType

            x_, st_ = layer(x, ps, st)
            x__, st__ = layer(x, ps, st)
            x___, st___ = layer(x_, ps, st_)

            @test st_.rng != st.rng
            @test st_.rng == st__.rng
            @test st_.mask == st__.mask
            @test x_ == x__
            @test x_ != x___

            @jet layer(x, ps, st)
            __f = let layer = layer, ps = ps, st = st
                x -> sum(first(layer(x, ps, st)))
            end
            @test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)

            @jet layer(x, ps, st_)
            __f = let layer = layer, ps = ps, st_ = st_
                x -> sum(first(layer(x, ps, st_)))
            end
            @test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)

            st__ = Lux.update_state(st_, :update_mask, Val(true))
            x___, st___ = layer(x, ps, st__)

            @test st___.mask != st__.mask
            @test x___ != x_

            @jet layer(x, ps, st__)
            __f = let layer = layer, ps = ps, st__ = st__
                x -> sum(first(layer(x, ps, st__)))
            end
            @test_gradients(__f, x; atol=1.0f-3, rtol=1.0f-3)
        end
    end
end
