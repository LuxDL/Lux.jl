module LuxEnzymeExt

using ADTypes: ADTypes, AutoEnzyme, ForwardMode, ReverseMode
using ArgCheck: @argcheck
using Enzyme: Enzyme, Active, Const, Duplicated, BatchDuplicated
using EnzymeCore: EnzymeCore
using Setfield: @set!
using Static: False, True, StaticBool

using Lux: Lux
using Lux.Training: TrainingBackendCache, TrainState

Lux.is_extension_loaded(::Val{:Enzyme}) = true

normalize_backend(::StaticBool, ad::AutoEnzyme) = ad
function normalize_backend(#=prefer_forward=#::True, ad::AutoEnzyme{Nothing, A}) where {A}
    return AutoEnzyme(; mode=Enzyme.Forward, function_annotation=A)
end
function normalize_backend(#=prefer_forward=#::False, ad::AutoEnzyme{Nothing, A}) where {A}
    return AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=A)
end

annotate_function(::AutoEnzyme{<:Any, Nothing}, f::F) where {F} = f
annotate_function(::AutoEnzyme{<:Any, A}, f::F) where {F, A} = A(f)

include("training.jl")

include("batched_autodiff.jl")

end
