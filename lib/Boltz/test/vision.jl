using Boltz, Lux

models_available = Dict(alexnet => [(:alexnet, true), (:alexnet, false)],
                        convmixer => [(:small, false), (:base, false), (:large, false)],
                        densenet => [
                            (:densenet121, false),
                            (:densenet161, false),
                            (:densenet169, false),
                            (:densenet201, false),
                        ], googlenet => [(:googlenet, false)],
                        mobilenet => [
                            (:mobilenet_v1, false),
                            (:mobilenet_v2, false),
                            (:mobilenet_v3_small, false),
                            (:mobilenet_v3_large, false),
                        ],
                        resnet => [
                            (:resnet18, true),
                            (:resnet18, false),
                            (:resnet34, true),
                            (:resnet34, false),
                            (:resnet50, true),
                            (:resnet50, false),
                            (:resnet101, true),
                            (:resnet101, false),
                            (:resnet152, true),
                            (:resnet152, false),
                        ],
                        resnext => [
                            (:resnext50, false),
                            (:resnext101, false),
                            (:resnext152, false),
                        ],
                        vgg => [
                            (:vgg11, false),
                            (:vgg11, true),
                            (:vgg11_bn, false),
                            (:vgg11_bn, true),
                            (:vgg13, false),
                            (:vgg13, true),
                            (:vgg13_bn, false),
                            (:vgg13_bn, true),
                            (:vgg16, false),
                            (:vgg16, true),
                            (:vgg16_bn, false),
                            (:vgg16_bn, true),
                            (:vgg19, false),
                            (:vgg19, true),
                            (:vgg19_bn, false),
                            (:vgg19_bn, true),
                        ],
                        vision_transformer => [
                            (:tiny, false),
                            (:small, false),
                            (:base, false),
                            # CI cant handle these
                            # (:large, false),
                            # (:huge, false),
                            # (:giant, false),
                            # (:gigantic, false),
                        ])

@testset "$model_creator" for (model_creator, config) in pairs(models_available)
    @time begin
        @testset "name = $name & pretrained = $pretrained" for (name, pretrained) in config
            if VERSION <= v"1.7" && pretrained
                @warn "Skipping pretrained models in Julia < 1.7"
                continue
            end
            model, ps, st = model_creator(name; pretrained)
            st = Lux.testmode(st)

            imsize = string(model_creator) == "vision_transformer" ? (256, 256) : (224, 224)
            x = randn(Float32, imsize..., 3, 1)

            @test size(first(model(x, ps, st))) == (1000, 1)

            GC.gc(true)
        end
    end
end
