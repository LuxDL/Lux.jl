module Lux

using ADTypes: AbstractADType, AutoEnzyme, AutoForwardDiff, AutoReverseDiff, AutoTracker,
               AutoZygote
using Adapt: Adapt, adapt
using ArgCheck: @argcheck
using ArrayInterface: ArrayInterface, fast_scalar_indexing
using ChainRulesCore: ChainRulesCore, AbstractZero, HasReverseMode, NoTangent, ProjectTo,
                      RuleConfig, ZeroTangent, @thunk
using Compat: @compat
using ConcreteStructs: @concrete
using ConstructionBase: ConstructionBase
using EnzymeCore: EnzymeCore, EnzymeRules
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using Functors: Functors, fmap
using GPUArraysCore: GPUArraysCore, @allowscalar
using LossFunctions: LossFunctions
using MacroTools: MacroTools, block, combinedef, splitdef
using Markdown: @doc_str
using NNlib: NNlib
using Optimisers: Optimisers
using Preferences: load_preference, has_preference
using Random: Random, AbstractRNG
using Reexport: @reexport
using Statistics: mean
using UnrolledUtilities: unrolled_map, unrolled_mapreduce

@reexport using LuxCore, LuxLib, LuxDeviceUtils, WeightInitializers
import LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer, initialparameters,
                initialstates, parameterlength, statelength, outputsize, apply,
                display_name, replicate

const CRC = ChainRulesCore

const NAME_TYPE = Union{Nothing, String, Symbol}

@inline _is_extension_loaded(::Val) = false

# Preferences
include("preferences.jl")

# Utilities
include("custom_errors.jl")
include("utils.jl")

# Training Helpers
include("helpers/training.jl")

# Experimental
include("contrib/contrib.jl")

# Layer Implementations
include("layers/basic.jl")
include("layers/containers.jl")
include("layers/normalize.jl")
include("layers/conv.jl")
include("layers/dropout.jl")
include("layers/recurrent.jl")
include("layers/extension.jl")

# Pretty Printing
include("layers/display.jl")

# Helpful Functionalities
include("helpers/stateful.jl")
include("helpers/compact.jl")
include("helpers/autodiff.jl")
include("helpers/nested_ad.jl")
include("helpers/losses.jl")
include("helpers/recursive_ops.jl")
include("helpers/match_eltype.jl")

# AutoDiff
include("chainrules.jl")
include("enzymerules.jl")

# ForwardDiff.jl Integration
include("forwarddiff/jvp.jl")
include("forwarddiff/nested_ad.jl")
include("forwarddiff/batched_ad.jl")

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
export NoOpLayer, ReshapeLayer, SelectDim, FlattenLayer, WrappedFunction, ReverseSequence
export RNNCell, LSTMCell, GRUCell, Recurrence, StatefulRecurrentCell, BidirectionalRNN
export SamePad, TimeLastIndex, BatchLastIndex

export StatefulLuxLayer
export CompactLuxLayer
export @compact, @init_fn, @non_trainable
export Training

export jacobian_vector_product, vector_jacobian_product
export batched_jacobian
export AutoEnzyme, AutoForwardDiff, AutoReverseDiff, AutoTracker, AutoZygote

export BinaryCrossEntropyLoss, BinaryFocalLoss, CrossEntropyLoss, DiceCoeffLoss, FocalLoss,
       HingeLoss, HuberLoss, KLDivergenceLoss, L1Loss, L2Loss, MAELoss, MSELoss, MSLELoss,
       PoissonLoss, SiameseContrastiveLoss, SquaredHingeLoss
export GenericLossFunction

export f16, f32, f64
export match_eltype

export transform
export FromFluxAdaptor, FluxLayer
export ToSimpleChainsAdaptor, SimpleChainsLayer
export DynamicExpressionsLayer

export MPIBackend, NCCLBackend, DistributedUtils

# Unexported functions that are part of the public API
@compat public Experimental
@compat public xlogx, xlogy
@compat(public,
    (recursive_add!!, recursive_copyto!, recursive_eltype,
        recursive_make_zero, recursive_map, recursive_make_zero!!))

end
