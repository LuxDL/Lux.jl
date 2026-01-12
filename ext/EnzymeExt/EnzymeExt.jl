module EnzymeExt

using ADTypes: ADTypes, AutoEnzyme
using Enzyme: Enzyme, Active, Const, Duplicated
using EnzymeCore: EnzymeCore, Forward, Reverse
using Functors: fmap
using Setfield: @set!
using Static: False, True

using Lux: Lux, Utils, AutoDiffInternalImpl
using Lux.Training: TrainingBackendCache, TrainState
using MLDataDevices: isleaf

Lux.is_extension_loaded(::Val{:Enzyme}) = true

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
