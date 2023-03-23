_div_idx(idx, n) = div(idx - 1, n) + 1
_mod_idx(idx, n) = mod(idx - 1, n) + 1

@static if VERSION >= v"1.7"
    get_device(x) = KA.get_device(x)
else
    # KA.get_device is not present in <= v0.7 but that is what works on julia 1.6
    get_device(x::CuArray) = CUDADevice()
    get_device(x::Array) = CPU()
    get_device(x::SubArray) = CPU()
    function get_device(x)
        throw(ArgumentError("get_device not implemented for $(typeof(x)). This is an" *
                            "undesirable codepath. Please use julia 1.7+ for more " *
                            "meaningful error messages using KA.jl."))
    end
end

_get_device(::Nothing) = nothing
_get_device(d) = hasmethod(get_device, (typeof(d),)) ? get_device(d) : nothing
_get_device(t::Tuple) = filter(!isnothing, _get_device.(t))

CRC.@non_differentiable _get_device(::Any)

function _assert_same_device(args...)
    devs = _get_device(args)
    if !all(devs .== (first(devs),))
        throw(ArgumentError("All arguments must be on the same device. This error is
                             encountered if you are calling a function with a mix of CPU
                             and GPU arrays."))
    end
    return
end

CRC.@non_differentiable _assert_same_device(::Any...)

@inline @generated _vec(x::T) where {T} = hasmethod(vec, (T,)) ? :(vec(x)) : :x

@inline @inbounds function _get_reshape_dims(sx::NTuple{N, <:Int}, ly::Int) where {N}
    if ly == sx[N - 1]
        return ntuple(i -> i == N - 1 ? ly : 1, N)
    elseif N > 2 && ly == sx[N - 1] * sx[N - 2]
        return ntuple(i -> i == (N - 1) || i == (N - 2) ? sx[i] : 1, N)
    else
        throw(ArgumentError("Invalid Dimensions!"))
    end
end

CRC.@non_differentiable _get_reshape_dims(::Any...)

@inline _reshape_into_proper_shape(::Nothing, y) = nothing
@inline _reshape_into_proper_shape(x, y) = reshape(x, _get_reshape_dims(size(y), length(x)))

# Copy and don't allow gradient propagation
_copy_autodiff_barrier(x) = copy(x)
_copy_autodiff_barrier(::Nothing) = nothing

CRC.@non_differentiable _copy_autodiff_barrier(::Any)

_replicate(rng::AbstractRNG) = copy(rng)
_replicate(rng::CUDA.RNG) = deepcopy(rng)

CRC.@non_differentiable _replicate(::Any)

# Var Implementation
## Using the default version from Statistics causes issues with Tracker.jl
function _var(x, ::Val{corrected}, _mean, ::Val{dims}) where {corrected, dims}
    return sum((x .- _mean) .^ 2; dims) ./ (prod(Base.Fix1(size, x), dims) - corrected)
end
