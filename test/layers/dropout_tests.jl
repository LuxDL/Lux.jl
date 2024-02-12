@testitem "Dropout" setup=[SharedTestSetup] begin
    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        for p in (0.5f0, 0.5)
            layer = Dropout(p)
            __display(layer)
            ps, st = Lux.setup(rng, layer) .|> device
            x = randn(Float32, 5, 2) |> aType

            x_, st_ = layer(x, ps, st)
            x__, st__ = layer(x, ps, st)
            x___, st___ = layer(x_, ps, st_)

            @test st_.rng != st.rng
            @test st_.rng == st__.rng
            @test x_ == x__
            @test x_ != x___

            @jet layer(x, ps, st)
            __f = x -> sum(first(layer(x, ps, st)))
            @eval @test_gradients $__f $x atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu

            st = Lux.testmode(st)

            @test first(layer(x, ps, st)) == x
        end
    end
end

@testitem "AlphaDropout" setup=[SharedTestSetup] begin
    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        for p in (0.5f0, 0.5)
            layer = AlphaDropout(p)
            __display(layer)
            ps, st = Lux.setup(rng, layer) .|> device
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
            __f = x -> sum(first(layer(x, ps, st)))
            @eval @test_gradients $__f $x atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu

            st = Lux.testmode(st)

            @test first(layer(x, ps, st)) == x
        end
    end
end

@testitem "VariationalHiddenDropout" setup=[SharedTestSetup] begin
    rng = get_stable_rng(12345)

    @testset "$mode" for (mode, aType, device, ongpu) in MODES
        for p in (0.5f0, 0.5)
            layer = VariationalHiddenDropout(p)
            __display(layer)
            ps, st = Lux.setup(rng, layer) .|> device
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
            __f = x -> sum(first(layer(x, ps, st)))
            @eval @test_gradients $__f $x atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu

            @jet layer(x, ps, st_)
            __f = x -> sum(first(layer(x, ps, st_)))
            @eval @test_gradients $__f $x atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu

            st__ = Lux.update_state(st_, :update_mask, Val(true))
            x___, st___ = layer(x, ps, st__)

            @test st___.mask != st__.mask
            @test x___ != x_

            @jet layer(x, ps, st__)
            __f = x -> sum(first(layer(x, ps, st__)))
            @eval @test_gradients $__f $x atol=1.0f-3 rtol=1.0f-3 gpu_testing=$ongpu
        end
    end
end
