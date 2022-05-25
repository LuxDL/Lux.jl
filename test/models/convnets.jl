using Lux, Metalhead

include("../utils.jl")

@testset "AlexNet" begin
    m = AlexNet()
    m2 = Lux.transform(m.layers)
    @test size(run_model(m2, rand(Float32, 256, 256, 3, 1))) == (1000, 1)
end

GC.gc(true)

@testset "VGG" begin @testset "VGG($sz, batchnorm=$bn)" for sz in [11, 13, 16, 19],
                                                            bn in [true, false]

    m = VGG(sz, batchnorm=bn)
    m2 = Lux.transform(m.layers)

    @test size(run_model(m2, rand(Float32, 224, 224, 3, 1))) == (1000, 1)

    GC.gc(true)
end end

@testset "ResNet" begin @testset "ResNet($sz)" for sz in [18, 34, 50, 101, 152]
    m = ResNet(sz)
    m2 = Lux.transform(m.layers)

    @test size(run_model(m2, rand(Float32, 256, 256, 3, 1))) == (1000, 1)

    GC.gc(true)
end end

@testset "ResNeXt" begin @testset for depth in [50, 101, 152]
    m = ResNeXt(depth)
    m2 = Lux.transform(m.layers)

    @test size(run_model(m2, rand(Float32, 224, 224, 3, 1))) == (1000, 1)

    GC.gc(true)
end end

@testset "GoogLeNet" begin
    m = GoogLeNet()
    m2 = Lux.transform(m.layers)

    @test size(run_model(m2, rand(Float32, 224, 224, 3, 1))) == (1000, 1)

    GC.gc(true)
end

@testset "Inception3" begin
    m = Inception3()
    m2 = Lux.transform(m.layers)

    @test size(run_model(m2, rand(Float32, 224, 224, 3, 1))) == (1000, 1)

    GC.gc(true)
end

@testset "SqueezeNet" begin
    m = SqueezeNet()
    m2 = Lux.transform(m.layers)

    @test size(run_model(m2, rand(Float32, 224, 224, 3, 1))) == (1000, 1)

    GC.gc(true)
end

@testset "DenseNet" begin @testset for sz in [121, 161, 169, 201]
    m = DenseNet(sz)
    m2 = Lux.transform(m.layers)

    @test size(run_model(m2, rand(Float32, 224, 224, 3, 1))) == (1000, 1)

    GC.gc(true)
end end

@testset "MobileNet" verbose=true begin
    @testset "MobileNetv1" begin
        m = MobileNetv1()
        m2 = Lux.transform(m.layers)

        @test size(run_model(m2, rand(Float32, 224, 224, 3, 1))) == (1000, 1)

        GC.gc(true)
    end

    GC.gc()

    @testset "MobileNetv2" begin
        m = MobileNetv2()
        m2 = Lux.transform(m.layers)

        @test size(run_model(m2, rand(Float32, 224, 224, 3, 1))) == (1000, 1)

        GC.gc(true)
    end

    GC.gc()

    @testset "MobileNetv3" verbose=true begin @testset for mode in [:small, :large]
        m = MobileNetv3(mode)
        m2 = Lux.transform(m.layers)

        @test size(run_model(m2, rand(Float32, 224, 224, 3, 1))) == (1000, 1)

        GC.gc(true)
    end end
end

GC.gc()

# TODO: ConvNeXt requires LayerNorm Implementation

GC.gc()

@testset "ConvMixer" verbose=true begin @testset for mode in [:base, :large, :small]
    m = ConvMixer(mode)
    m2 = Lux.transform(m.layers)

    @test size(run_model(m2, rand(Float32, 224, 224, 3, 1))) == (1000, 1)

    GC.gc(true)
end end
