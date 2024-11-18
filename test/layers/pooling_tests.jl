@testitem "Pooling" setup=[SharedTestSetup] tags=[:core_layers] begin
    rng = StableRNG(12345)

    nnlib_op = Dict(:LPPool => (args...) -> lpnormpool(args...; p=2),
        :MeanPool => meanpool, :MaxPool => maxpool)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset for ltype in (:LPPool, :MeanPool, :MaxPool)
            if ongpu && ltype == :LPPool
                @test_broken false
                continue
            end

            broken_backends = ltype == :LPPool ? Any[AutoTracker()] : []

            adaptive_ltype = Symbol(:Adaptive, ltype)
            global_ltype = Symbol(:Global, ltype)

            x = randn(rng, Float32, 10, 10, 3, 2) |> aType
            y = randn(rng, Float32, 20, 20, 3, 2) |> aType

            layer = getfield(Lux, adaptive_ltype)((5, 5))
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev

            @test size(layer(x, ps, st)[1]) == (5, 5, 3, 2)
            @test layer(x, ps, st)[1] == nnlib_op[ltype](x, PoolDims(x, 2))
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3,
                broken_backends)

            layer = getfield(Lux, adaptive_ltype)((10, 5))
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev

            @test size(layer(y, ps, st)[1]) == (10, 5, 3, 2)
            @test layer(y, ps, st)[1] == nnlib_op[ltype](y, PoolDims(y, (2, 4)))
            @jet layer(y, ps, st)
            @test_gradients(sumabs2first, layer, y, ps, st; atol=1.0f-3, rtol=1.0f-3,
                broken_backends)

            broken_backends2 = broken_backends
            if VERSION ≥ v"1.11-"
                push!(broken_backends2, AutoEnzyme())
            elseif ltype == :LPPool
                push!(broken_backends2, AutoEnzyme())
            end

            layer = getfield(Lux, global_ltype)()
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev

            @test size(layer(x, ps, st)[1]) == (1, 1, 3, 2)
            @test layer(x, ps, st)[1] == nnlib_op[ltype](x, PoolDims(x, size(x)[1:2]))
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3,
                rtol=1.0f-3, broken_backends=broken_backends2)

            layer = getfield(Lux, ltype)((2, 2))
            display(layer)
            ps, st = Lux.setup(rng, layer) |> dev

            @test layer(x, ps, st)[1] == nnlib_op[ltype](x, PoolDims(x, 2))
            @jet layer(x, ps, st)
            @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3,
                broken_backends)

            @testset "SamePad windowsize $k" for k in ((1,), (2,), (3,), (4, 5), (6, 7, 8))
                x = ones(Float32, (k .+ 3)..., 1, 1) |> aType

                layer = getfield(Lux, ltype)(k; pad=Lux.SamePad())
                display(layer)
                ps, st = Lux.setup(rng, layer) |> dev

                @test size(layer(x, ps, st)[1])[1:(end - 2)] ==
                      cld.(size(x)[1:(end - 2)], k)
                @jet layer(x, ps, st)

                soft_fail = ltype == :MaxPool ? [AutoFiniteDiff()] : []
                @test_gradients(sumabs2first, layer, x, ps, st; atol=1.0f-3, rtol=1.0f-3,
                    soft_fail, broken_backends)
            end
        end
    end
end
