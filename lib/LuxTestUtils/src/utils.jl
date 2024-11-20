# Taken from https://github.com/JuliaLang/julia/pull/54653
struct Fix{N, F, T} <: Function
    f::F
    x::T

    Fix{N}(f::F, x::Constant) where {N, F} = Fix{N}(f, x.val)
    function Fix{N}(f::F, x) where {N, F}
        if N isa Int && N < 1
            throw(ArgumentError("expected `N` in `Fix{N}` to be integer greater than 0, \
                                 but got $N"))
        elseif !(N isa Union{Int, Symbol})
            throw(ArgumentError("expected type parameter in `Fix` to be `Int` or `Symbol`, \
                                 but got `$N::$(typeof(N))`"))
        end
        return new{N, Base._stable_typeof(f), Base._stable_typeof(x)}(f, x)
    end
end
function Fix(f::F; kws...) where {F}
    length(kws) != 1 &&
        throw(ArgumentError("`Fix` expects exactly one argument or keyword argument, but \
                             got keywords `$(keys(kws))`"))
    return Fix{only(keys(kws))}(f, only(values(kws)))
end

function (f::Fix{N})(args::Vararg{Any, M}; kws...) where {N, M}
    if N isa Symbol
        N in keys(kws) &&
            throw(ArgumentError("found duplicate keyword argument `$N` passed to a `Fix` \
                                 function"))
        f_kws = NamedTuple{(N,)}((f.x,))
        return f.f(args...; f_kws..., kws...)
    else # Int
        M < N - 1 &&
            throw(ArgumentError("expected at least $(N-1) arguments to a `Fix` function with `N=$(N)`, but got $M"))
        return f.f(
            args[begin:(begin + (N - 2))]..., f.x, args[(begin + (N - 1)):end]...; kws...)
    end
end

# Special cases for improved constant propagation
(f::Fix{1})(arg; kws...) = f.f(f.x, arg; kws...)
(f::Fix{2})(arg; kws...) = f.f(arg, f.x; kws...)

function partial_function(f::F, idx::Int, args...) where {F}
    partial_f = f
    for (i, arg) in enumerate(args)
        i == idx && continue
        i < idx && (partial_f = Fix{1}(partial_f, arg))
        i > idx && (partial_f = Fix{2}(partial_f, arg))
    end
    return partial_f, args[idx]
end

function flatten_gradient_computable(f, nt)
    if needs_gradient(nt)
        x_flat, re = Optimisers.destructure(nt)
        _f = x -> f(Functors.fmap(ArrayInterface.aos_to_soa, re(x)))
        return _f, x_flat, re
    end
    return nothing, nothing, nothing
end

needs_gradient(::Constant) = false
function needs_gradient(y)
    leaves = Functors.fleaves(y)
    isempty(leaves) && return false
    return all(Fix{2}(isa, AbstractArray{<:AbstractFloat}), leaves)
end

__length(x) = 0
__length(x::AbstractArray) = length(x)
__length(::Number) = 1

# Equality Checks
struct GradientComputationSkipped end

@generated function check_approx(x::X, y::Y; kwargs...) where {X, Y}
    device = cpu_device()
    (X == GradientComputationSkipped || Y == GradientComputationSkipped) && return :(true)
    hasmethod(isapprox, (X, Y)) && return :(isapprox($(device)(x), $(device)(y); kwargs...))
    return :($(device)(x) == $(device)(y))
end

check_approx(x::Tuple, y::Tuple; kwargs...) = all(check_approx.(x, y; kwargs...))

function check_approx(
        nt1::NamedTuple{fields}, nt2::NamedTuple{fields}; kwargs...) where {fields}
    _check_approx(xy) = check_approx(xy[1], xy[2]; kwargs...)
    _check_approx(t::Tuple{Nothing, Nothing}) = true
    return all(_check_approx, zip(values(nt1), values(nt2)))
end

function check_approx(t1::NTuple{N, T}, t2::NTuple{N, T}; kwargs...) where {N, T}
    _check_approx(xy) = check_approx(xy[1], xy[2]; kwargs...)
    _check_approx(t::Tuple{Nothing, Nothing}) = true
    return all(_check_approx, zip(t1, t2))
end

function check_approx(ca::ComponentArray, nt::NamedTuple; kwargs...)
    return check_approx(NamedTuple(ca), nt; kwargs...)
end
function check_approx(nt::NamedTuple, ca::ComponentArray; kwargs...)
    return check_approx(nt, NamedTuple(ca); kwargs...)
end

check_approx(::Nothing, v::AbstractArray; kwargs...) = length(v) == 0
check_approx(v::AbstractArray, ::Nothing; kwargs...) = length(v) == 0
check_approx(v::NamedTuple, ::Nothing; kwargs...) = length(v) == 0
check_approx(::Nothing, v::NamedTuple; kwargs...) = length(v) == 0
check_approx(v::Tuple, ::Nothing; kwargs...) = length(v) == 0
check_approx(::Nothing, v::Tuple; kwargs...) = length(v) == 0
check_approx(x::AbstractArray, y::NamedTuple; kwargs...) = length(x) == 0 && length(y) == 0
check_approx(x::NamedTuple, y::AbstractArray; kwargs...) = length(x) == 0 && length(y) == 0
check_approx(x::AbstractArray, y::Tuple; kwargs...) = length(x) == 0 && length(y) == 0
check_approx(x::Tuple, y::AbstractArray; kwargs...) = length(x) == 0 && length(y) == 0

# Taken from discourse. normalizes the order of keyword arguments in a macro
function reorder_macro_kw_params(exs)
    exs = Any[exs...]
    i = findfirst([(ex isa Expr && ex.head == :parameters) for ex in exs])
    if i !== nothing
        extra_kw_def = exs[i].args
        for ex in extra_kw_def
            push!(exs, ex isa Symbol ? Expr(:kw, ex, ex) : ex)
        end
        deleteat!(exs, i)
    end
    return Tuple(exs)
end

function check_ad_backend_in(backend, backends)
    backends_type = map(ArrayInterface.parameterless_type ∘ typeof, backends)
    backend_type = ArrayInterface.parameterless_type(typeof(backend))
    return backend_type in backends_type
end
