module LuxEnzymeExt

using ADTypes: ADTypes, AutoEnzyme, ForwardMode, ReverseMode
using ArgCheck: @argcheck
using ConcreteStructs: @concrete
using Enzyme: Enzyme, Active, Const, Duplicated, BatchDuplicated
using EnzymeCore: EnzymeCore
using Functors: fmap
using Setfield: @set!, @set
using Static: False, True, StaticBool

using Lux: Lux, Utils
using Lux.Training: TrainingBackendCache, TrainState
using MLDataDevices: isleaf

Lux.is_extension_loaded(::Val{:Enzyme}) = true

normalize_backend(::StaticBool, ad::AutoEnzyme) = ad
normalize_backend(::True, ad::AutoEnzyme{Nothing}) = @set(ad.mode=Enzyme.Forward)
normalize_backend(::False, ad::AutoEnzyme{Nothing}) = @set(ad.mode=Enzyme.Reverse)

annotate_function(::AutoEnzyme{<:Any, Nothing}, f::F) where {F} = f
annotate_function(::AutoEnzyme{<:Any, A}, f::F) where {F, A} = A(f)

include("training.jl")

include("batched_autodiff.jl")

end
