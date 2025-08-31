@testitem "Pooling" setup = [SharedTestSetup] tags = [:core_layers] begin
    rng = StableRNG(12345)

    nnlib_op = Dict(
        :LPPool => (args...) -> lpnormpool(args...; p=2),
        :MeanPool => meanpool,
        :MaxPool => maxpool,
    )

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset for ltype in (:LPPool, :MeanPool, :MaxPool)
            if ongpu && ltype == :LPPool
                @test_broken false
                continue
            end

            adaptive_ltype = Symbol(:Adaptive, ltype)
            global_ltype = Symbol(:Global, ltype)

            x = aType(randn(rng, Float32, 10, 10, 3, 2))
            y = aType(randn(rng, Float32, 20, 20, 3, 2))

            layer = getfield(Lux, adaptive_ltype)((5, 5))
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test size(layer(x, ps, st)[1]) == (5, 5, 3, 2)
            @test layer(x, ps, st)[1] == nnlib_op[ltype](x, PoolDims(x, 2))
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            layer = getfield(Lux, adaptive_ltype)((10, 5))
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test size(layer(y, ps, st)[1]) == (10, 5, 3, 2)
            @test layer(y, ps, st)[1] == nnlib_op[ltype](y, PoolDims(y, (2, 4)))
            @jet layer(y, ps, st)
            @test_gradients(sumabs2first, layer, y, ps, st; atol=1.0f-3, rtol=1.0f-3)

            layer = getfield(Lux, global_ltype)()
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test size(layer(x, ps, st)[1]) == (1, 1, 3, 2)
            @test layer(x, ps, st)[1] == nnlib_op[ltype](x, PoolDims(x, size(x)[1:2]))
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            layer = getfield(Lux, ltype)((2, 2))
            display(layer)
            ps, st = dev(Lux.setup(rng, layer))

            @test layer(x, ps, st)[1] == nnlib_op[ltype](x, PoolDims(x, 2))
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3)

            @testset "SamePad windowsize $k" for k in ((1,), (2,), (3,), (4, 5), (6, 7, 8))
                x = aType(ones(Float32, (k .+ 3)..., 1, 1))

                layer = getfield(Lux, ltype)(k; pad=Lux.SamePad())
                display(layer)
                ps, st = dev(Lux.setup(rng, layer))

                @test size(layer(x, ps, st)[1])[1:(end - 2)] ==
                    cld.(size(x)[1:(end - 2)], k)
                @jet layer(x, ps, st)

                soft_fail = ltype == :MaxPool ? [AutoFiniteDiff()] : []
                @test_gradients(
                    sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3, soft_fail
                )
            end
        end
    end
end
