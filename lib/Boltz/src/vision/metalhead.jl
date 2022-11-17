function alexnet(name::Symbol; pretrained=false, kwargs...)
    assert_name_present_in(name, (:alexnet,))
    model = transform(AlexNet().layers)

    # Compatibility with pretrained weights
    if pretrained
        model = Chain(model[1], model[2])
    end

    return _initialize_model(name, model; pretrained, kwargs...)
end

function resnet(name::Symbol; pretrained=false, kwargs...)
    assert_name_present_in(name, (:resnet18, :resnet34, :resnet50, :resnet101, :resnet152))
    model = if name == :resnet18
        transform(ResNet(18).layers)
    elseif name == :resnet34
        transform(ResNet(34).layers)
    elseif name == :resnet50
        transform(ResNet(50).layers)
    elseif name == :resnet101
        transform(ResNet(101).layers)
    elseif name == :resnet152
        transform(ResNet(152).layers)
    end

    # Compatibility with pretrained weights
    if pretrained
        model = Chain(model[1], model[2])
    end

    return _initialize_model(name, model; pretrained, kwargs...)
end

function resnext(name::Symbol; kwargs...)
    assert_name_present_in(name, (:resnext50, :resnext101, :resnext152))
    model = if name == :resnext50
        transform(ResNeXt(50).layers)
    elseif name == :resnext101
        transform(ResNeXt(101).layers)
    elseif name == :resnext152
        transform(ResNeXt(152).layers)
    end
    return _initialize_model(name, model; kwargs...)
end

function googlenet(name::Symbol; kwargs...)
    assert_name_present_in(name, (:googlenet,))
    model = transform(GoogLeNet().layers)
    return _initialize_model(name, model; kwargs...)
end

function densenet(name::Symbol; kwargs...)
    assert_name_present_in(name, (:densenet121, :densenet161, :densenet169, :densenet201))
    model = if name == :densenet121
        transform(DenseNet(121).layers)
    elseif name == :densenet161
        transform(DenseNet(161).layers)
    elseif name == :densenet169
        transform(DenseNet(169).layers)
    elseif name == :densenet201
        transform(DenseNet(201).layers)
    end
    return _initialize_model(name, model; kwargs...)
end

function mobilenet(name::Symbol; kwargs...)
    assert_name_present_in(name,
                           (:mobilenet_v1, :mobilenet_v2, :mobilenet_v3_small,
                            :mobilenet_v3_large))
    model = if name == :mobilenet_v1
        transform(MobileNetv1().layers)
    elseif name == :mobilenet_v2
        transform(MobileNetv2().layers)
    elseif name == :mobilenet_v3_small
        transform(MobileNetv3(:small).layers)
    elseif name == :mobilenet_v3_large
        transform(MobileNetv3(:large).layers)
    end
    return _initialize_model(name, model; kwargs...)
end

function convmixer(name::Symbol; kwargs...)
    assert_name_present_in(name, (:base, :large, :small))
    model = transform(ConvMixer(name).layers)
    return _initialize_model(name, model; kwargs...)
end
