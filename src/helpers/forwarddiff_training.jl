using ADTypes: AutoForwardDiff
using DiffResults: DiffResults
using ForwardDiff: ForwardDiff
using Setfield: @set!
using Static: True, False

function Training.compute_gradients_impl(
    ad::AutoForwardDiff, obj_fn::F, data, ts::Training.TrainState
) where {F}
    @assert ts.parameters isa AbstractArray "AutoForwardDiff only supports AbstractArray \
                                             parameters, not $(typeof(ts.parameters)). To \
                                             convert the parameter structure to an array \
                                             use `ComponentArray(ps)`."

    obj_fn_wrap, st_wrap, stats_wrap = Training.wrap_objective_function(
        obj_fn, ts.model, ts.parameters, ts.states, data, True()
    )

    gradient_result = DiffResults.GradientResult(ts.parameters)
    ForwardDiff.gradient!(
        gradient_result, ps -> obj_fn_wrap(ts.model, ps, ts.states, data), ts.parameters
    )

    cache = Training.TrainingBackendCache(
        ad, False(), gradient_result, (; obj_fn=obj_fn_wrap, st_wrap, stats_wrap)
    )
    @set! ts.cache = cache
    @set! ts.objective_function = obj_fn
    @set! ts.states = st_wrap[]
    return (
        DiffResults.gradient(gradient_result),
        DiffResults.value(gradient_result),
        stats_wrap[],
        ts,
    )
end

const FORWARDDIFF_CACHE_TYPE = Training.TrainingBackendCache{
    <:AutoForwardDiff,False,PS,<:NamedTuple{(:obj_fn, :st_wrap, :stats_wrap)}
} where {PS}

function Training.compute_gradients_impl(
    ::AutoForwardDiff, obj_fn::F, data, ts::Training.TrainState{<:FORWARDDIFF_CACHE_TYPE,F}
) where {F}
    gradient_result = ts.cache.dparameters

    ForwardDiff.gradient!(
        gradient_result,
        ps -> ts.cache.extras.obj_fn(ts.model, ps, ts.states, data),
        ts.parameters,
    )

    @set! ts.objective_function = obj_fn
    @set! ts.states = ts.cache.extras.st_wrap[]

    return (
        DiffResults.gradient(gradient_result),
        DiffResults.value(gradient_result),
        ts.cache.extras.stats_wrap[],
        ts,
    )
end

function Training.compute_gradients_impl(
    ::AutoForwardDiff,
    obj_fn::F,
    data,
    ts::Training.TrainState{<:Training.TrainingBackendCache{<:AutoForwardDiff,False}},
) where {F}
    @warn "Detected calls to `compute_gradients(::AutoForwardDiff, ...)` with objective \
           function that is changing across function calls. This can lead to the \
           generation of slow code" maxlog = 1
    gradient_result = ts.cache.dparameters

    # We do exactly same thing as the first case but without caching the function
    obj_fn_wrap, st_wrap, stats_wrap = Training.wrap_objective_function(
        obj_fn, ts.model, ts.parameters, ts.states, data, False()
    )

    ForwardDiff.gradient!(
        gradient_result, ps -> obj_fn_wrap(ts.model, ps, ts.states, data), ts.parameters
    )

    @set! ts.states = st_wrap[]
    return (
        DiffResults.gradient(gradient_result),
        DiffResults.value(gradient_result),
        stats_wrap[],
        ts,
    )
end

# Type Piracy for ForwardDiff GPU Array Support
# This is a workaround for ForwardDiff.jl not supporting GPU arrays post v1.0
# See: https://github.com/JuliaDiff/ForwardDiff.jl/pull/760

using GPUArraysCore: AnyGPUArray

# Helper struct for broadcasting partials extraction
struct PartialsFn{T,D<:ForwardDiff.Dual}
    dual::D
end

PartialsFn{T}(dual::ForwardDiff.Dual) where {T} = PartialsFn{T,typeof(dual)}(dual)

(f::PartialsFn{T})(i) where {T} = ForwardDiff.partials(T, f.dual, i)

# Macro to define ForwardDiff overloads for array types that don't support scalar indexing
macro define_forwarddiff_gpu_overloads(ArrayType)
    return quote
        # Overloaded seed! methods
        function ForwardDiff.seed!(
            duals::$(esc(ArrayType)){ForwardDiff.Dual{T,V,N}},
            x,
            seed::ForwardDiff.Partials{N,V}=zero(ForwardDiff.Partials{N,V}),
        ) where {T,V,N}
            idxs = collect(ForwardDiff.structural_eachindex(duals, x))
            duals[idxs] .= ForwardDiff.Dual{T,V,N}.(view(x, idxs), Ref(seed))
            return duals
        end

        function ForwardDiff.seed!(
            duals::$(esc(ArrayType)){ForwardDiff.Dual{T,V,N}},
            x,
            seeds::NTuple{N,ForwardDiff.Partials{N,V}},
        ) where {T,V,N}
            idxs = collect(Iterators.take(ForwardDiff.structural_eachindex(duals, x), N))
            duals[idxs] .=
                ForwardDiff.Dual{
                    T,V,N
                }.(view(x, idxs), getindex.(Ref(seeds), 1:length(idxs)))
            return duals
        end

        function ForwardDiff.seed!(
            duals::$(esc(ArrayType)){ForwardDiff.Dual{T,V,N}},
            x,
            index,
            seed::ForwardDiff.Partials{N,V}=zero(ForwardDiff.Partials{N,V}),
        ) where {T,V,N}
            idxs = collect(
                Iterators.drop(ForwardDiff.structural_eachindex(duals, x), index - 1)
            )
            duals[idxs] .= ForwardDiff.Dual{T,V,N}.(view(x, idxs), Ref(seed))
            return duals
        end

        function ForwardDiff.seed!(
            duals::$(esc(ArrayType)){ForwardDiff.Dual{T,V,N}},
            x,
            index,
            seeds::NTuple{N,ForwardDiff.Partials{N,V}},
            chunksize=N,
        ) where {T,V,N}
            idxs = collect(
                Iterators.take(
                    Iterators.drop(ForwardDiff.structural_eachindex(duals, x), index - 1),
                    chunksize,
                ),
            )
            duals[idxs] .=
                ForwardDiff.Dual{
                    T,V,N
                }.(view(x, idxs), getindex.(Ref(seeds), 1:length(idxs)))
            return duals
        end

        # Overloaded extract_gradient! methods
        function ForwardDiff.extract_gradient!(
            ::Type{T}, result::$(esc(ArrayType)), dual::ForwardDiff.Dual
        ) where {T}
            fn = PartialsFn{T}(dual)
            idxs = collect(
                Iterators.take(
                    ForwardDiff.structural_eachindex(result), ForwardDiff.npartials(dual)
                ),
            )
            result[idxs] .= fn.(1:length(idxs))
            return result
        end

        function ForwardDiff.extract_gradient_chunk!(
            ::Type{T}, result::$(esc(ArrayType)), dual, index, chunksize
        ) where {T}
            fn = PartialsFn{T}(dual)
            idxs = collect(
                Iterators.take(
                    Iterators.drop(ForwardDiff.structural_eachindex(result), index - 1),
                    chunksize,
                ),
            )
            result[idxs] .= fn.(1:length(idxs))
            return result
        end
    end
end

@static if pkgversion(ForwardDiff) ≥ v"1.0.1"
    # Apply overloads for GPU arrays
    @define_forwarddiff_gpu_overloads AnyGPUArray
end
