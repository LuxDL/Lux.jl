function alexnet(name::Symbol; kwargs...)
    assert_name_present_in(name, (:alexnet,))
    model = Lux.transform(AlexNet().layers)
    return initialize_model(name, model; kwargs...)
end

function resnet(name::Symbol; kwargs...)
    assert_name_present_in(name, (:resnet18, :resnet34, :resnet50, :resnet101, :resnet152))
    model = if name == :resnet18
        Lux.transform(ResNet(18).layers)
    elseif name == :resnet34
        Lux.transform(ResNet(34).layers)
    elseif name == :resnet50
        Lux.transform(ResNet(50).layers)
    elseif name == :resnet101
        Lux.transform(ResNet(101).layers)
    elseif name == :resnet152
        Lux.transform(ResNet(152).layers)
    end
    return initialize_model(name, model; kwargs...)
end

function resnext(name::Symbol; kwargs...)
    assert_name_present_in(name, (:resnext50, :resnext101, :resnext152))
    model = if name == :resnext50
        Lux.transform(ResNeXt(50).layers)
    elseif name == :resnext101
        Lux.transform(ResNeXt(101).layers)
    elseif name == :resnext152
        Lux.transform(ResNeXt(152).layers)
    end
    return initialize_model(name, model; kwargs...)
end

function googlenet(name::Symbol; kwargs...)
    assert_name_present_in(name, (:googlenet,))
    model = Lux.transform(GoogLeNet().layers)
    return initialize_model(name, model; kwargs...)
end

function densenet(name::Symbol; kwargs...)
    assert_name_present_in(name, (:densenet121, :densenet161, :densenet169, :densenet201))
    model = if name == :densenet121
        Lux.transform(DenseNet(121).layers)
    elseif name == :densenet161
        Lux.transform(DenseNet(161).layers)
    elseif name == :densenet169
        Lux.transform(DenseNet(169).layers)
    elseif name == :densenet201
        Lux.transform(DenseNet(201).layers)
    end
    return initialize_model(name, model; kwargs...)
end

function mobilenet(name::Symbol; kwargs...)
    assert_name_present_in(name,
                           (:mobilenet_v1, :mobilenet_v2, :mobilenet_v3_small,
                            :mobilenet_v3_large))
    model = if name == :mobilenet_v1
        Lux.transform(MobileNetv1().layers)
    elseif name == :mobilenet_v2
        Lux.transform(MobileNetv2().layers)
    elseif name == :mobilenet_v3_small
        Lux.transform(MobileNetv3(:small).layers)
    elseif name == :mobilenet_v3_large
        Lux.transform(MobileNetv3(:large).layers)
    end
    return initialize_model(name, model; kwargs...)
end

function convmixer(name::Symbol; kwargs...)
    assert_name_present_in(name, (:base, :large, :small))
    model = Lux.transform(ConvMixer(name).layers)
    return initialize_model(name, model; kwargs...)
end
