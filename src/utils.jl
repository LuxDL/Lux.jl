# PRNG Handling
"""
    replicate(rng::AbstractRNG)
    replicate(rng::CUDA.RNG)

Creates a copy of the `rng` state depending on its type.
"""
replicate(rng::AbstractRNG) = copy(rng)

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
function _convfilter(rng::AbstractRNG,
    filter::NTuple{N, Integer},
    ch::Pair{<:Integer, <:Integer};
    init=glorot_uniform,
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

function _calc_padding(pad, k::NTuple{N}, dilation, stride) where {N}
    return _expand(Val(2 * N), pad)
end

function _calc_padding(::SamePad, k::NTuple, dilation, stride)
    # Ref: "A guide to convolution arithmetic for deep learning"
    # https://arxiv.org/abs/1603.07285 Effective kernel size, including dilation
    k_eff = @. k + (k - 1) * (dilation - 1)
    # How much total padding needs to be applied?
    pad_amt = @. k_eff - 1
    # In case amount of padding is odd we need to apply different amounts to each side.
    return Tuple(mapfoldl(i -> [cld(i, 2), fld(i, 2)], vcat, pad_amt))
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

@inline function _init_trainable_hidden_state(hidden_state::AbstractVector,
    x::AbstractMatrix)
    return repeat(hidden_state, 1, size(x, 2))
end

"""
    multigate(x::AbstractArray, ::Val{N})

Split up `x` into `N` equally sized chunks (along dimension `1`).
"""
@inline function multigate(x::AbstractArray, ::Val{N}) where {N}
    return ntuple(i -> _gate(x, size(x, 1) ÷ N, i), N)
end

# Val utilities
get_known(::Val{T}) where {T} = T

# Indexing into NamedTuple
function _index_namedtuple(nt::NamedTuple{fields}, idxs::AbstractArray) where {fields}
    return NamedTuple{fields[idxs]}(values(nt)[idxs])
end

# If doesn't have a property, return nothing
@generated function _getproperty(x::NamedTuple{names}, ::Val{v}) where {names, v}
    return v ∈ names ? :(x.$v) : :(nothing)
end

## Slow-fallback
@inline function _getproperty(x, ::Val{v}) where {v}
    return v ∈ propertynames(x) ? getproperty(x, v) : nothing
end

@inline function _eachslice(x::AbstractArray, ::Val{dims}) where {dims}
    return [selectdim(x, dims, i) for i in axes(x, dims)]
end

function ∇_eachslice(Δ_raw, x::AbstractArray, ::Val{dims}) where {dims}
    Δs = CRC.unthunk(Δ_raw)
    i1 = findfirst(Δ -> Δ isa AbstractArray, Δs)
    i1 === nothing && zero.(x)  # all slices are Zero!
    Δ = similar(x)
    for i in axes(x, dims)
        Δi = selectdim(Δ, dims, i)
        if Δi isa CRC.AbstractZero
            fill!(Δi, 0)
        else
            copyto!(Δi, Δs[i])
        end
    end
    return CRC.ProjectTo(x)(Δ)
end

# Activation Function
@inline __apply_activation(::typeof(identity), x) = x
@inline __apply_activation(f, x) = f.(x)

# Backend Integration
## Convolution
@inline _conv(x, weight, cdims) = conv(x, weight, cdims)

@inline _conv_transpose(x, weight, cdims) = ∇conv_data(x, weight, cdims)

function _conv_transpose_dims(x::AbstractArray,
    weight::AbstractArray;
    padding,
    stride,
    dilation,
    groups)
    # Calculate size of "input", from ∇conv_data()'s perspective...
    combined_pad = (padding[1:2:end] .+ padding[2:2:end])
    I = (size(x)[1:(end - 2)] .- 1) .* stride .+ 1 .+
        (size(weight)[1:(end - 2)] .- 1) .* dilation .- combined_pad
    C_in = size(weight)[end - 1] * groups
    batch_size = size(x)[end]
    # Create DenseConvDims() that looks like the corresponding conv()
    w_size = size(weight)
    return DenseConvDims((I..., C_in, batch_size),
        w_size;
        stride,
        padding,
        dilation,
        groups)
end

## Adaptive Pooling
@inline function compute_adaptive_pooling_dims(x::AbstractArray, outsize)
    insize = size(x)[1:(end - 2)]
    stride = insize .÷ outsize
    k = insize .- (outsize .- 1) .* stride
    pad = 0
    return PoolDims(x, k; padding=pad, stride=stride)
end

## Foldl with init
"""
    foldl_init(op, x)
    foldl_init(op, x, init)

Exactly same as `foldl(op, x; init)` in the forward pass. But, gives gradients wrt `init`
in the backward pass.
"""
@inline foldl_init(op, x) = foldl_init(op, x, nothing)
@inline foldl_init(op, x, init) = foldl(op, x; init)

# Merging Exotic Types
_merge(nt1::NamedTuple, nt2::NamedTuple) = merge(nt1, nt2)
function _merge(p::AbstractArray, nt::NamedTuple)
    @assert length(p) == 0
    return nt
end
function _merge(nt::NamedTuple, p::AbstractArray)
    @assert length(p) == 0
    return nt
end
