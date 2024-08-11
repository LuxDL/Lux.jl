module Lux

using ADTypes: AbstractADType, AutoEnzyme, AutoForwardDiff, AutoReverseDiff, AutoTracker,
               AutoZygote
using Adapt: Adapt, adapt
using ArgCheck: @argcheck
using ArrayInterface: ArrayInterface
using ChainRulesCore: ChainRulesCore, HasReverseMode, NoTangent, RuleConfig, ZeroTangent,
                      @thunk
using Compat: @compat
using ConcreteStructs: @concrete
using ConstructionBase: ConstructionBase
using FastClosures: @closure
using ForwardDiff: ForwardDiff
using Functors: Functors, fmap
using GPUArraysCore: GPUArraysCore, @allowscalar
using LossFunctions: LossFunctions
using Markdown: @doc_str
using NNlib: NNlib
using Optimisers: Optimisers
using Preferences: load_preference, has_preference, set_preferences!
using Random: Random, AbstractRNG
using Static: StaticBool, True, False, static
using Reexport: @reexport
using Statistics: mean
using UnrolledUtilities: unrolled_map, unrolled_mapreduce

@reexport using LuxCore, LuxLib, LuxDeviceUtils, WeightInitializers
import LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer, initialparameters,
                initialstates, parameterlength, statelength, inputsize, outputsize,
                update_state, trainmode, testmode, setup, apply, replicate

const CRC = ChainRulesCore

const NAME_TYPE = Union{Nothing, String, Symbol}
const Optional{T} = Union{T, Nothing}

is_extension_loaded(::Val) = false

# Preferences
include("preferences.jl")

# Utilities
include("piracies.jl")
include("custom_errors.jl")
include("utils.jl")
include("extended_ops.jl")

# Training Helpers
include("helpers/training.jl")

# Experimental
include("contrib/contrib.jl")

# Pretty Printing
include("layers/display.jl")

# Layer Implementations
include("layers/basic.jl")
include("layers/containers.jl")
include("layers/normalize.jl")
include("layers/conv.jl")
include("layers/dropout.jl")
include("layers/recurrent.jl")
include("layers/extension.jl")

# Helpful Functionalities
include("helpers/eltype_conversion.jl")
include("helpers/stateful.jl")
include("helpers/compact.jl")
include("helpers/losses.jl")
include("helpers/recursive_ops.jl")
include("helpers/match_eltype.jl")

# AutoDiff
include("autodiff/api.jl")
include("autodiff/autodiff.jl")

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

export LuxOps

# Unexported functions that are part of the public API
@compat public Experimental
@compat public xlogx, xlogy # TODO: deprecated in v1.0
@compat public set_dispatch_doctor_preferences!

end
