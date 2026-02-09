using LuxCore, Random

# Define some custom layers
struct Dense <: AbstractLuxLayer
    in::Int
    out::Int
end

function LuxCore.initialparameters(rng::AbstractRNG, l::Dense)
    return (w=randn(rng, l.out, l.in), b=randn(rng, l.out))
end

(::Dense)(x, _, st) = x, st  # Dummy Forward Pass

struct DenseWrapper{L} <: AbstractLuxWrapperLayer{:layer}
    layer::L
end

# For checking ambiguities in the dispatch
struct DenseWrapper2{L} <: AbstractLuxWrapperLayer{:layer}
    layer::L
end

(d::DenseWrapper2)(x::AbstractArray, ps, st) = d.layer(x, ps, st)

struct Chain{L} <: AbstractLuxContainerLayer{(:layers,)}
    layers::L
end

function (c::Chain)(x, ps, st)
    y, st1 = c.layers[1](x, ps.layers.layer_1, st.layers.layer_1)
    y, st2 = c.layers[2](y, ps.layers.layer_2, st.layers.layer_2)
    return y, (; layers=(; layer_1=st1, layer_2=st2))
end

struct ChainWrapper{L} <: AbstractLuxWrapperLayer{:layers}
    layers::L
end

function (c::ChainWrapper)(x, ps, st)
    y, st1 = c.layers[1](x, ps.layer_1, st.layer_1)
    y, st2 = c.layers[2](y, ps.layer_2, st.layer_2)
    return y, (; layer_1=st1, layer_2=st2)
end

struct Chain2{L1,L2} <: AbstractLuxContainerLayer{(:layer1, :layer2)}
    layer1::L1
    layer2::L2
end

function (c::Chain2)(x, ps, st)
    y, st1 = c.layer1(x, ps.layer1, st.layer1)
    y, st2 = c.layer2(y, ps.layer2, st.layer2)
    return y, (; layer1=st1, layer2=st2)
end
