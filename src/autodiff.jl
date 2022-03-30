Base.zero(s::NamedTuple{(),Tuple{}}) = s

Base.zero(::Symbol) = Symbol()

Base.zero(nt::NamedTuple{fields}) where {fields} = NamedTuple{fields}(zero.(values(nt)))

# Layers are stateless so we can simply return that
Base.zero(l::AbstractExplicitLayer) = l

ChainRulesCore.rrule(::typeof(istraining)) = true, _ -> (NoTangent(),)

ChainRulesCore.@non_differentiable _update_stats!(::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any)

function ChainRulesCore.rrule(::typeof(Val), x)
    valx = Val(x)
    val_pullback(Δ) = NoTangent(), NoTangent()
    return valx, val_pullback
end

# Sparse Arrays
_project(x, y) = x .* one.(y)

function ChainRulesCore.rrule(
    ::typeof(*),
    X::EFLSparseMatrixCSC{<:Union{AbstractSparseMatrixCSC,AbstractCuSparseMatrix}},
    Y::Union{Matrix,CuMatrix},
)
    Z = X * Y
    function sparse_matmul_pullback(Δ)
        Δ = unthunk(Δ)
        return NoTangent(), _project(Δ * Y', X), X.mat' * Δ 
    end
    return Z, sparse_matmul_pullback
end


