module Lux

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ADTypes: AbstractADType, AutoForwardDiff, AutoReverseDiff, AutoTracker, AutoZygote
    using Adapt: Adapt, adapt
    using ArrayInterface: ArrayInterface
    using ChainRulesCore: ChainRulesCore, AbstractZero, HasReverseMode, NoTangent,
                          ProjectTo, RuleConfig, ZeroTangent
    using ConcreteStructs: @concrete
    using FastClosures: @closure
    using Functors: Functors, fmap
    using GPUArraysCore: GPUArraysCore
    using Markdown: @doc_str
    using OhMyThreads: tmapreduce
    using Preferences: @load_preference
    using Random: Random, AbstractRNG
    using Reexport: @reexport

    using LuxCore, LuxLib, LuxDeviceUtils, WeightInitializers
    using LuxLib: __apply_bias_activation
    import LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer,
                    initialparameters, initialstates, parameterlength, statelength,
                    inputsize, outputsize, update_state, trainmode, testmode, setup, apply,
                    display_name, replicate
    using LuxDeviceUtils: get_device

    # @compact specific
    using MacroTools: MacroTools, block, combinedef, splitdef

    # @compact and stateful layers
    using ConstructionBase: ConstructionBase
end

@reexport using LuxCore, LuxLib, LuxDeviceUtils, WeightInitializers

const CRC = ChainRulesCore

const NAME_TYPE = Union{Nothing, String, Symbol}

@inline _is_extension_loaded(::Val) = false

const DISABLE_AUTOMATIC_NESTED_AD_SWITCH = @load_preference("DisableAutomaticNestedADSwitching",
    false)

# Utilities
include("utils.jl")

# Layer Implementations
include("layers/basic.jl")
include("layers/containers.jl")
include("layers/normalize.jl")
include("layers/conv.jl")
include("layers/dropout.jl")
include("layers/recurrent.jl")
include("layers/kan.jl")
include("layers/extension.jl")

# Pretty Printing
include("layers/display.jl")

# AutoDiff
include("chainrules.jl")

# Experimental
include("contrib/contrib.jl")

# Helpful Functionalities
include("helpers/stateful.jl")
include("helpers/compact.jl")
include("helpers/autodiff.jl")
include("helpers/nested_ad.jl")

# Transform to and from other frameworks
include("transform/types.jl")
include("transform/flux.jl")
include("transform/simplechains.jl")

# Distributed Training
include("distributed/backend.jl")
include("distributed/public_api.jl")

# Deprecations
include("deprecated.jl")

# Layers
export cpu, gpu  # deprecated

export Chain, Parallel, SkipConnection, PairwiseFusion, BranchLayer, Maxout, RepeatedLayer
export Bilinear, Dense, Embedding, Scale, PeriodicEmbedding
export Conv, ConvTranspose, CrossCor, MaxPool, MeanPool, GlobalMaxPool, GlobalMeanPool,
       AdaptiveMaxPool, AdaptiveMeanPool, Upsample, PixelShuffle
export AlphaDropout, Dropout, VariationalHiddenDropout
export BatchNorm, GroupNorm, InstanceNorm, LayerNorm
export WeightNorm
export NoOpLayer, ReshapeLayer, SelectDim, FlattenLayer, WrappedFunction
export RNNCell, LSTMCell, GRUCell, Recurrence, StatefulRecurrentCell
export SamePad, TimeLastIndex, BatchLastIndex
export KAN, KANCell, SymbolicKANCell

export StatefulLuxLayer
export @compact, CompactLuxLayer

export jacobian_vector_product, vector_jacobian_product
export batched_jacobian
export AutoForwardDiff, AutoReverseDiff, AutoTracker, AutoZygote

export f16, f32, f64

export transform
export FromFluxAdaptor, FluxLayer
export ToSimpleChainsAdaptor, SimpleChainsLayer
export DynamicExpressionsLayer

export MPIBackend, NCCLBackend, DistributedUtils

end
