## Quite a few of these are borrowed from Flux.jl

# Misc
@inline _nfan() = 1, 1 # fan_in, fan_out
@inline _nfan(n) = 1, n # A vector is treated as a n×1 matrix
@inline _nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices
@inline _nfan(dims::Tuple) = _nfan(dims...)
# In case of convolution kernels
@inline _nfan(dims...) = prod(dims[1:(end - 2)]) .* (dims[end - 1], dims[end])

# Neural Network Initialization
# NOTE(@avik-pal): Would be great if these could be moved into its own package and NN
#                  frameworks could just import it.
"""
    zeros32(rng::AbstractRNG, size...) = zeros(Float32, size...)

Return an `Array{Float32}` of zeros of the given `size`. (`rng` is ignored)
"""
zeros32(rng::AbstractRNG, args...; kwargs...) = zeros(rng, Float32, args...; kwargs...)

"""
    ones32(rng::AbstractRNG, size...) = ones(Float32, size...)

Return an `Array{Float32}` of ones of the given `size`. (`rng` is ignored)
"""
ones32(rng::AbstractRNG, args...; kwargs...) = ones(rng, Float32, args...; kwargs...)

Base.zeros(rng::AbstractRNG, args...; kwargs...) = zeros(args...; kwargs...)
Base.ones(rng::AbstractRNG, args...; kwargs...) = ones(args...; kwargs...)

"""
    randn32(rng::AbstractRNG, size...) = randn(rng, Float32, size...)

Return an `Array{Float32}` of random numbers from a standard normal distribution of the
given `size`.
"""
randn32(rng::AbstractRNG, size...) = randn(rng, Float32, size...)

"""
    rand32(rng::AbstractRNG, size...) = rand(rng, Float32, size...)

Return an `Array{Float32}` of random numbers from a uniform distribution of the given
`size`.
"""
rand32(rng::AbstractRNG, size...) = rand(rng, Float32, size...)

"""
    glorot_uniform(rng::AbstractRNG, size...; gain = 1)

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a
uniform distribution on the interval ``[-x, x]``, where
`x = gain * sqrt(6 / (fan_in + fan_out))`. This method is described in [1] and also known as
Xavier initialization.

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep
feedforward neural networks." _Proceedings of the thirteenth international conference on
artificial intelligence and statistics_. 2010.
"""
function glorot_uniform(rng::AbstractRNG, dims::Integer...; gain::Real=1)
    scale = Float32(gain) * sqrt(24.0f0 / sum(_nfan(dims...)))
    return (rand(rng, Float32, dims...) .- 0.5f0) .* scale
end

"""
    glorot_normal(rng::AbstractRNG, size...; gain = 1)

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a normal
distribution with standard deviation `gain * sqrt(2 / (fan_in + fan_out))`. This method is
described in [1] and also known as Xavier initialization.

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep
feedforward neural networks." _Proceedings of the thirteenth international conference on
artificial intelligence and statistics_. 2010.
"""
function glorot_normal(rng::AbstractRNG, dims::Integer...; gain::Real=1)
    std = Float32(gain) * sqrt(2.0f0 / sum(_nfan(dims...)))
    return randn(rng, Float32, dims...) .* std
end

# PRNG Handling
"""
    replicate(rng::AbstractRNG)
    replicate(rng::CUDA.RNG)

Creates a copy of the `rng` state depending on its type.
"""
replicate(rng::AbstractRNG) = copy(rng)
replicate(rng::CUDA.RNG) = deepcopy(rng)

# Training Check
"""
    istraining(::Val{training})
    istraining(st::NamedTuple)

Returns `true` if `training` is `true` or if `st` contains a `training` field with value
`true`. Else returns `false`.

Method undefined if `st.training` is not of type `Val`.
"""
@inline istraining(::Val{training}) where {training} = training
@inline istraining(st::NamedTuple) = hasproperty(st, :training) && istraining(st.training)

# Convolution
function _convfilter(rng::AbstractRNG, filter::NTuple{N, Integer},
                     ch::Pair{<:Integer, <:Integer}; init=glorot_uniform,
                     groups=1) where {N}
    cin, cout = ch
    @assert cin % groups==0 "Input channel dimension must be divisible by groups."
    @assert cout % groups==0 "Output channel dimension must be divisible by groups."
    return init(rng, filter..., cin ÷ groups, cout)
end

_expand(N, i::Tuple) = i
_expand(N, i::Integer) = ntuple(_ -> i, N)

_maybetuple_string(pad) = string(pad)
_maybetuple_string(pad::Tuple) = all(==(pad[1]), pad) ? string(pad[1]) : string(pad)

# Padding
struct SamePad end

function _calc_padding(pad, k::NTuple{N, T}, dilation, stride) where {T, N}
    return _expand(Val(2 * N), pad)
end

function _calc_padding(::SamePad, k::NTuple{N, T}, dilation, stride) where {N, T}
    # Ref: "A guide to convolution arithmetic for deep learning"
    # https://arxiv.org/abs/1603.07285 Effective kernel size, including dilation
    k_eff = @. k + (k - 1) * (dilation - 1)
    # How much total padding needs to be applied?
    pad_amt = @. k_eff - 1
    # In case amount of padding is odd we need to apply different amounts to each side.
    return Tuple(mapfoldl(i -> [cld(i, 2), fld(i, 2)], vcat, pad_amt))
end

# Handling ComponentArrays
function Functors.functor(::Type{<:ComponentArray}, c)
    return NamedTuple{propertynames(c)}(getproperty.((c,), propertynames(c))),
           ComponentArray
end

Optimisers.setup(opt::AbstractRule, ps::ComponentArray) = Optimisers.setup(opt, getdata(ps))

function Optimisers.update(tree, ps::ComponentArray, gs::ComponentArray)
    tree, ps_new = Optimisers.update(tree, getdata(ps), getdata(gs))
    return tree, ComponentArray(ps_new, getaxes(ps))
end

function Optimisers.update!(tree::Optimisers.Leaf, ps::ComponentArray, gs::ComponentArray)
    tree, ps_new = Optimisers.update!(tree, getdata(ps), getdata(gs))
    return tree, ComponentArray(ps_new, getaxes(ps))
end

# Getting typename
get_typename(::T) where {T} = Base.typename(T).wrapper

# RNN Utilities
@inline _gate(h::Int, n::Int) = (1:h) .+ h * (n - 1)
@inline _gate(x::AbstractVector, h::Int, n::Int) = view(x, _gate(h, n))
@inline _gate(x::AbstractMatrix, h::Int, n::Int) = view(x, _gate(h, n), :)

@inline function _init_hidden_state(rng::AbstractRNG, rnn, x::AbstractMatrix)
    return rnn.init_state(rng, rnn.out_dims, size(x, 2))
end

@inline function _init_hidden_state(rng::AbstractRNG, rnn,
                                    x::Union{CUDA.StridedSubCuArray, CuArray})
    return CuArray(rnn.init_state(rng, rnn.out_dims, size(x, 2)))
end

@inline function _init_trainable_hidden_state(hidden_state::AbstractVector,
                                              x::AbstractMatrix)
    return repeat(hidden_state, 1, size(x, 2))
end

"""
    multigate(x::AbstractArray, ::Val{N})

Split up `x` into `N` equally sized chunks (along dimension `1`).
"""
@inline multigate(x::AbstractArray, ::Val{N}) where {N} = _gate.((x,), size(x, 1) ÷ N, 1:N)

# Val utilities
get_known(::Val{T}) where {T} = T

# Indexing into NamedTuple
function _index_namedtuple(nt::NamedTuple{fields}, idxs::AbstractArray) where {fields}
    return NamedTuple{fields[idxs]}(values(nt)[idxs])
end

# If doesn't have a property, return nothing
@generated function _getproperty(x::NamedTuple{names}, ::Val{v}) where {names, v}
    if v in names
        return :(x.$v)
    else
        return :(nothing)
    end
end

@inline function _getproperty(x::ComponentArray, ::Val{prop}) where {prop}
    return prop in propertynames(x) ? getproperty(x, prop) : nothing
end

@inline function _eachslice(x::T, ::Val{dims}) where {T <: AbstractArray, dims}
    return [selectdim(x, dims, i) for i in axes(x, dims)]
end
