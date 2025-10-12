module LuxEnzymeExt

using ADTypes: ADTypes, AutoEnzyme
using Enzyme: Enzyme, Active, Const, Duplicated
using EnzymeCore: EnzymeCore, Forward, Reverse
using Functors: fmap
using Setfield: @set!, @set
using Static: StaticBool, False, True

using Lux: Lux, Utils, AutoDiffInternalImpl
using Lux.Training: TrainingBackendCache, TrainState
using MLDataDevices: isleaf

Lux.is_extension_loaded(::Val{:Enzyme}) = true

normalize_backend(::StaticBool, ad::AutoEnzyme) = ad
normalize_backend(::True, ad::AutoEnzyme{Nothing}) = @set(ad.mode = Forward)
normalize_backend(::False, ad::AutoEnzyme{Nothing}) = @set(ad.mode = Reverse)

annotate_function(::AutoEnzyme{<:Any,Nothing}, f::F) where {F} = f
annotate_function(::AutoEnzyme{<:Any,A}, f::F) where {F,A} = A(f)

struct OOPFunctionWrapper{F}
    f::F
end

function (f::OOPFunctionWrapper)(y, args...)
    res = f.f(args...)
    fmap(copyto!, y, res; exclude=isleaf)
    return nothing
end

include("training.jl")

include("autodiff.jl")

end
