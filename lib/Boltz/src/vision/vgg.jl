"""
    vgg_block(input_filters, output_filters, depth, batchnorm)

A VGG block of convolution layers ([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments

  - `input_filters`: number of input feature maps
  - `output_filters`: number of output feature maps
  - `depth`: number of convolution/convolution + batch norm layers
  - `batchnorm`: set to `true` to include batch normalization after each convolution
"""
function vgg_block(input_filters, output_filters, depth, batchnorm)
    k = (3, 3)
    p = (1, 1)
    layers = []
    for _ in 1:depth
        push!(layers,
              Conv(k, input_filters => output_filters, batchnorm ? identity : relu; pad=p))
        if batchnorm
            push!(layers, BatchNorm(output_filters, relu))
        end
        input_filters = output_filters
    end
    return Chain(layers...)
end

"""
    vgg_convolutional_layers(config, batchnorm, inchannels)

Create VGG convolution layers ([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments

  - `config`: vector of tuples `(output_channels, num_convolutions)` for each block
    (see [`Metalhead.vgg_block`](#))
  - `batchnorm`: set to `true` to include batch normalization after each convolution
  - `inchannels`: number of input channels
"""
function vgg_convolutional_layers(config, batchnorm, inchannels)
    layers = []
    input_filters = inchannels
    for c in config
        push!(layers, vgg_block(input_filters, c..., batchnorm))
        push!(layers, MaxPool((2, 2); stride=2))
        input_filters, _ = c
    end
    return Chain(layers...)
end

"""
    vgg_classifier_layers(imsize, nclasses, fcsize, dropout)

Create VGG classifier (fully connected) layers
([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments

  - `imsize`: tuple `(width, height, channels)` indicating the size after the convolution
    layers (see [`Metalhead.vgg_convolutional_layers`](#))
  - `nclasses`: number of output classes
  - `fcsize`: input and output size of the intermediate fully connected layer
  - `dropout`: the dropout level between each fully connected layer
"""
function vgg_classifier_layers(imsize, nclasses, fcsize, dropout)
    return Chain(FlattenLayer(), Dense(Int(prod(imsize)), fcsize, relu), Dropout(dropout),
                 Dense(fcsize, fcsize, relu), Dropout(dropout), Dense(fcsize, nclasses))
end

"""
    vgg(imsize; config, inchannels, batchnorm = false, nclasses, fcsize, dropout)

Create a VGG model ([reference](https://arxiv.org/abs/1409.1556v6)).

# Arguments

  - `imsize`: input image width and height as a tuple
  - `config`: the configuration for the convolution layers
  - `inchannels`: number of input channels
  - `batchnorm`: set to `true` to use batch normalization after each convolution
  - `nclasses`: number of output classes
  - `fcsize`: intermediate fully connected layer size
    (see [`Metalhead.vgg_classifier_layers`](#))
  - `dropout`: dropout level between fully connected layers
"""
function vgg(imsize; config, inchannels, batchnorm=false, nclasses, fcsize, dropout)
    conv = vgg_convolutional_layers(config, batchnorm, inchannels)
    class = vgg_classifier_layers((7, 7, 512), nclasses, fcsize, dropout)
    return Chain(Chain(conv), class)
end

const VGG_CONV_CONFIG = Dict(:A => [(64, 1), (128, 1), (256, 2), (512, 2), (512, 2)],
                             :B => [(64, 2), (128, 2), (256, 2), (512, 2), (512, 2)],
                             :D => [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)],
                             :E => [(64, 2), (128, 2), (256, 4), (512, 4), (512, 4)])

const VGG_CONFIG = Dict(11 => :A, 13 => :B, 16 => :D, 19 => :E)

function vgg(name::Symbol; kwargs...)
    assert_name_present_in(name,
                           (:vgg11, :vgg11_bn, :vgg13, :vgg13_bn, :vgg16, :vgg16_bn, :vgg19,
                            :vgg19_bn))
    model = if name == :vgg11
        vgg((224, 224); config=VGG_CONV_CONFIG[VGG_CONFIG[11]], inchannels=3,
            batchnorm=false, nclasses=1000, fcsize=4096, dropout=0.5f0)
    elseif name == :vgg11_bn
        vgg((224, 224); config=VGG_CONV_CONFIG[VGG_CONFIG[11]], inchannels=3,
            batchnorm=true, nclasses=1000, fcsize=4096, dropout=0.5f0)
    elseif name == :vgg13
        vgg((224, 224); config=VGG_CONV_CONFIG[VGG_CONFIG[13]], inchannels=3,
            batchnorm=false, nclasses=1000, fcsize=4096, dropout=0.5f0)
    elseif name == :vgg13_bn
        vgg((224, 224); config=VGG_CONV_CONFIG[VGG_CONFIG[13]], inchannels=3,
            batchnorm=true, nclasses=1000, fcsize=4096, dropout=0.5f0)
    elseif name == :vgg16
        vgg((224, 224); config=VGG_CONV_CONFIG[VGG_CONFIG[16]], inchannels=3,
            batchnorm=false, nclasses=1000, fcsize=4096, dropout=0.5f0)
    elseif name == :vgg16_bn
        vgg((224, 224); config=VGG_CONV_CONFIG[VGG_CONFIG[16]], inchannels=3,
            batchnorm=true, nclasses=1000, fcsize=4096, dropout=0.5f0)
    elseif name == :vgg19
        vgg((224, 224); config=VGG_CONV_CONFIG[VGG_CONFIG[19]], inchannels=3,
            batchnorm=false, nclasses=1000, fcsize=4096, dropout=0.5f0)
    elseif name == :vgg19_bn
        vgg((224, 224); config=VGG_CONV_CONFIG[VGG_CONFIG[19]], inchannels=3,
            batchnorm=true, nclasses=1000, fcsize=4096, dropout=0.5f0)
    end
    return initialize_model(name, model; kwargs...)
end
