@testitem "Type Stability" setup=[SharedTestSetup] tags=[:core_layers] begin
    using Zygote

    rng = StableRNG(12345)

    #! format: off
    MODELS = [
        [
            Dense(3 => 4, gelu),
            Dense(3 => 4, gelu; use_bias=false),
            Dense(3 => 4, relu),
            Dense(3 => 4, relu; use_bias=false),
            Dense(3 => 4),
            Dense(3 => 4; use_bias=false),
        ],
        [Chain(Dense(3 => 4, gelu), Dense(4 => 4))],
        [BatchNorm(3, gelu), BatchNorm(3)],
        [GroupNorm(12, 4, gelu), GroupNorm(12, 4)],
        [InstanceNorm(3, gelu), InstanceNorm(3)],
        [LayerNorm((1, 1, 3), gelu), LayerNorm((1, 1, 3))],
        [
            Conv((3, 3), 3 => 4, relu),
            Conv((3, 3), 3 => 4),
            Conv((3, 3), 3 => 4, gelu; use_bias=false),
            Conv((3, 3), 3 => 4; use_bias=false),
        ],
        [Dropout(0.5f0), AlphaDropout(0.5f0)],
        [
            Chain(Parallel(nothing, Dense(3 => 4, gelu), Dense(3 => 4, gelu)),
                WrappedFunction(first)),
            Parallel(+, Dense(3 => 4, gelu), Dense(3 => 4, gelu))
        ],
        [
            Chain(BranchLayer(Dense(3 => 4, gelu), Dense(3 => 4, gelu)),
                WrappedFunction(first))
        ],
        [RepeatedLayer(Dense(3 => 4, gelu))],
        [PairwiseFusion(+, Dense(3 => 3, gelu), Dense(3 => 3, gelu))],
        [
            MaxPool((2, 2)),
            MeanPool((2, 2)),
            GlobalMaxPool(),
            GlobalMeanPool(),
            AdaptiveMaxPool((2, 2)),
            AdaptiveMeanPool((2, 2)),
        ]
    ]

    INPUTS = [
        [randn(rng, Float32, 3, 4)],
        [randn(rng, Float32, 3, 4)],
        [randn(rng, Float32, 3, 2), randn(rng, Float32, 2, 2, 3, 4)],
        [randn(rng, Float32, 12, 2), randn(rng, Float32, 3, 3, 12, 2)],
        [randn(rng, Float32, 2, 2, 3, 4)],
        [randn(rng, Float32, 3, 3, 3, 2)],
        [randn(rng, Float32, 10, 10, 3, 2)],
        [randn(rng, Float32, 10, 10, 3, 2), randn(rng, Float32, 3, 2)],
        [(randn(rng, Float32, 3, 4), randn(rng, Float32, 3, 4))],
        [randn(rng, Float32, 3, 2)],
        [randn(rng, Float32, 3, 2)],
        [randn(rng, Float32, 3, 2)],
        [randn(rng, Float32, 4, 4, 2, 3)]
    ]
    #! format: on

    loss_function(model, x, ps, st) = sum(abs2, first(model(x, ps, st)))

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "$(nameof(typeof(model)))" for (model_list, inputs) in zip(MODELS, INPUTS),
            model in model_list,
            input in inputs

            model = maybe_rewrite_to_crosscor(mode, model)
            ps, st = Lux.setup(rng, model) |> dev
            x = input |> dev

            @test @inferred(model(x, ps, st)) isa Any
            @test @inferred(loss_function(model, x, ps, st)) isa Any
            if mode == "amdgpu" && (model isa Conv || model isa CrossCor)
                allow_unstable() do
                    @test_broken @inferred(Zygote.gradient(
                        loss_function, model, x, ps, st)) isa Any
                end
            else
                allow_unstable() do
                    @test @inferred(Zygote.gradient(loss_function, model, x, ps, st)) isa
                          Any
                end
            end
        end
    end
end
