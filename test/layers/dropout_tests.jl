@testitem "Dropout" setup = [SharedTestSetup] tags = [:normalize_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        for p in (0.5f0, 0.5)
            layer = Dropout(p)
            display(layer)
            ps, st = dev.(Lux.setup(rng, layer))
            x = aType(randn(Float32, 5, 2))

            x_, st_ = layer(x, ps, st)
            x__, st__ = layer(x, ps, st)
            x___, st___ = layer(x_, ps, st_)

            @test st_.rng != st.rng
            @test st_.rng == st__.rng
            @test x_ == x__
            @test x_ != x___

            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            st = Lux.testmode(st)
            @test first(layer(x, ps, st)) == x
        end
    end
end

@testitem "AlphaDropout" setup = [SharedTestSetup] tags = [:normalize_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        for p in (0.5f0, 0.5)
            layer = AlphaDropout(p)
            display(layer)
            ps, st = dev.(Lux.setup(rng, layer))
            # GPU compilation for mixed types fail atm
            x = aType(randn(typeof(p), 5, 2))

            x_, st_ = layer(x, ps, st)
            x__, st__ = layer(x, ps, st)
            x___, st___ = layer(x_, ps, st_)

            @test st_.rng != st.rng
            @test st_.rng == st__.rng
            @test x_ == x__
            @test x_ != x___

            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            st = Lux.testmode(st)
            @test first(layer(x, ps, st)) == x
        end
    end
end

@testitem "VariationalHiddenDropout" setup = [SharedTestSetup] tags = [:normalize_layers] begin
    rng = StableRNG(12345)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        for p in (0.5f0, 0.5)
            layer = VariationalHiddenDropout(p)
            display(layer)
            ps, st = dev.(Lux.setup(rng, layer))
            x = aType(randn(Float32, 5, 2))

            x_, st_ = layer(x, ps, st)
            x__, st__ = layer(x, ps, st)
            x___, st___ = layer(x_, ps, st_)

            @test st_.rng != st.rng
            @test st_.rng == st__.rng
            @test st_.mask == st__.mask
            @test x_ == x__
            @test x_ != x___

            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            @jet layer(x, ps, st_)
            @test_gradients(sumabs2first, layer, x, ps, st_; atol=1.0f-3, rtol=1.0f-3)

            st__ = Lux.update_state(st_, :update_mask, Val(true))
            x___, st___ = layer(x, ps, st__)

            @test st___.mask != st__.mask
            @test x___ != x_

            @jet layer(x, ps, st__)
            @test_gradients(sumabs2first, layer, x, ps, st__; atol=1.0f-3, rtol=1.0f-3)
        end
    end
end
