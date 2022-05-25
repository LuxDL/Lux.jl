function alexnet(name::Symbol; pretrained::Bool=false, kwargs...)
    assert_name_present_in(name, (:default,))
end

function vgg(name::Symbol; pretrained::Bool=false, kwargs...) end

function resnet(name::Symbol; pretrained::Bool=false, kwargs...) end

function resnext(name::Symbol; pretrained::Bool=false, kwargs...) end

function googlenet(name::Symbol; pretrained::Bool=false, kwargs...) end

function densenet(name::Symbol; pretrained::Bool=false, kwargs...) end

function mobilenet(name::Symbol; pretrained::Bool=false, kwargs...) end

function convmixer(name::Symbol; pretrained::Bool=false, kwargs...) end
