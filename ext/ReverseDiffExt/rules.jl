# SimpleChains.jl
@grad_from_chainrules Lux.apply_simple_chain(layer, x::TrackedArray, ps, ::CPUDevice)
@grad_from_chainrules Lux.apply_simple_chain(layer, x, ps::TrackedArray, ::CPUDevice)
@grad_from_chainrules Lux.apply_simple_chain(
    layer, x::TrackedArray, ps::TrackedArray, ::CPUDevice
)

# Nested AD
@grad_from_chainrules Lux.AutoDiffInternalImpl.batched_jacobian_internal(
    f, backend::AbstractADType, x::TrackedArray
)
