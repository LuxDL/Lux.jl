struct MaxPool{N,M} <: AbstractExplicitLayer
    k::NTuple{N,Int}
    pad::NTuple{M,Int}
    stride::NTuple{N,Int}
end
  
function MaxPool(k::NTuple{N,Integer}; pad = 0, stride = k) where N
    stride = expand(Val(N), stride)
    pad = calc_padding(MaxPool, pad, k, 1, stride)
    return MaxPool(k, pad, stride)
end
  
function (m::MaxPool)(x, ::NamedTuple, st::NamedTuple)
    pdims = PoolDims(x, m.k; padding=m.pad, stride=m.stride)
    return maxpool(x, pdims), st
end
  
function Base.show(io::IO, m::MaxPool)
    print(io, "MaxPool(", m.k)
    all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", _maybetuple_string(m.stride))
    print(io, ")")
end


struct MeanPool{N,M} <: AbstractExplicitLayer
    k::NTuple{N,Int}
    pad::NTuple{M,Int}
    stride::NTuple{N,Int}
end
  
function MeanPool(k::NTuple{N,Integer}; pad = 0, stride = k) where N
    stride = expand(Val(N), stride)
    pad = calc_padding(MeanPool, pad, k, 1, stride)
    return MeanPool(k, pad, stride)
end
  
function (m::MeanPool)(x, ::NamedTuple, st::NamedTuple)
    pdims = PoolDims(x, m.k; padding=m.pad, stride=m.stride)
    return meanpool(x, pdims), st
end
  
function Base.show(io::IO, m::MeanPool)
    print(io, "MeanPool(", m.k)
    all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", _maybetuple_string(m.stride))
    print(io, ")")
end

