module LuxEnzymeExt

using ADTypes: AutoEnzyme
using Enzyme: Enzyme
using Lux: Lux
using Setfield: @set!

function Lux.Experimental.compute_gradients(::AutoEnzyme, objective_function::F, data,
        ts::Lux.Experimental.TrainState) where {F}
    dps = Enzyme.make_zero(ts.parameters)
    fwd, rev = Enzyme.autodiff_thunk(
        Enzyme.ReverseSplitWithPrimal, Enzyme.Const{typeof(objective_function)},
        Enzyme.Active, Enzyme.Const{typeof(ts.model)},
        Enzyme.Duplicated{typeof(ts.parameters)},
        Enzyme.Const{typeof(ts.states)}, Enzyme.Const{typeof(data)})
    tape, (loss, st_new, stats), shadow_result = fwd(
        Enzyme.Const(objective_function), Enzyme.Const(ts.model),
        Enzyme.Duplicated(ts.parameters, dps), Enzyme.Const(ts.states), Enzyme.Const(data))
    rev(Enzyme.Const(objective_function), Enzyme.Const(ts.model),
        Enzyme.Duplicated(ts.parameters, dps), Enzyme.Const(ts.states), Enzyme.Const(data),
        (one(loss), Enzyme.make_zero(st_new), Enzyme.make_zero(stats)), tape)
    @set! ts.states = st_new
    return dps, loss, stats, ts
end

end
