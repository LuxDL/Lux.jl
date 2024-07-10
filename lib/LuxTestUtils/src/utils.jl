# Taken from https://github.com/JuliaLang/julia/pull/54653
struct Fix{N, F, T} <: Function
    f::F
    x::T

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

function flatten_gradient_computable(f, nt::NamedTuple)
    leaves = Functors.fleaves(nt)
    if all(x -> x isa Number || x isa AbstractArray, leaves)
        _f = (x) -> f(NamedTuple(x))
        return _f, nt |> cpu_device() |> ComponentArray |> get_device(nt)
    end
    return nothing, nothing
end

