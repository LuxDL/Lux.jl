struct WeightNorm{which_params,L<:ExplicitLayer,D} <: ExplicitLayer
    layer::L
    dims::D
end

function WeightNorm(layer::ExplicitLayer, which_params::NTuple{N,Symbol}, dims::Union{Tuple,Nothing}=nothing) where {N}
    return WeightNorm{Val{which_params},typeof(layer),typeof(dims)}(layer, dims)
end

function initialparameters(rng::AbstractRNG, wn::WeightNorm{Val{which_params}}) where {which_params}
    ps_layer = initialparameters(rng, wn.layer)
    ps_normalized = []
    ps_unnormalized = []
    i = 1
    for (k, v) ∈ pairs(ps_layer)
        if k ∈ which_params
            dim = wn.dims === nothing ? ndims(v) : wn.dims[i]
            push!(ps_normalized, Symbol(string(k) * "_g") => _norm_except(v, dim))
            push!(ps_normalized, Symbol(string(k) * "_v") => v)
            i += 1
        else
            push!(ps_unnormalized, k => v)
        end
    end
    return (normalized = (; ps_normalized...), unnormalized = (; ps_unnormalized...))
end

initialstates(rng::AbstractRNG, wn::WeightNorm) = initialstates(rng, wn.layer)

function (wn::WeightNorm)(x, ps::NamedTuple, s::NamedTuple)
    _ps = get_normalized_parameters(wn, wn.dims, ps.normalized)
    return wn.layer(x, (; _ps..., ps.unnormalized...), s)
end

function get_normalized_parameters(::WeightNorm{Val{which_params}}, ::Nothing, ps::NamedTuple) where which_params
    wp_s = string.(which_params)
    _vs = getfield.((ps,), Symbol.(wp_s .* "_v"))
    _gs = getfield.((ps,), Symbol.(wp_s .* "_g"))
    return NamedTuple{which_params}(map((v, g) -> (v .* (g ./ _norm_except(v))), _vs, _gs))
end

function get_normalized_parameters(::WeightNorm{Val{which_params}}, dims::Tuple, ps::NamedTuple) where which_params
    wp_s = string.(which_params)
    _vs = getfield.((ps,), Symbol.(wp_s .* "_v"))
    _gs = getfield.((ps,), Symbol.(wp_s .* "_g"))
    return NamedTuple{which_params}(map((v, g, dim) -> (v .* (g ./ _norm_except(v, dim))), _vs, _gs, dims))
end

@inline _norm(x; dims=Colon()) = sqrt.(sum(abs2, x; dims=dims))

# Compute norm over all dimensions except `except_dim`
@inline _norm_except(x::AbstractArray{T,N}, except_dim) where {T,N} = _norm(x; dims=filter(i -> i != except_dim, 1:N))
@inline _norm_except(x::AbstractArray{T,N}) where {T,N} = _norm_except(x, N)
