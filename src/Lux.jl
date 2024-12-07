module Lux

using ADTypes: AbstractADType, AutoEnzyme, AutoForwardDiff, AutoReverseDiff, AutoTracker,
               AutoZygote
using Adapt: Adapt, adapt
using ArgCheck: @argcheck
using ArrayInterface: ArrayInterface
using ChainRulesCore: ChainRulesCore, NoTangent, @thunk
using Compat: @compat
using ConcreteStructs: @concrete
using EnzymeCore: EnzymeRules
using FastClosures: @closure
using Functors: Functors, KeyPath, fmap
using GPUArraysCore: @allowscalar
using Markdown: @doc_str
using NNlib: NNlib
using Optimisers: Optimisers
using Random: Random, AbstractRNG
using Static: StaticBool, StaticInt, StaticSymbol, True, False, static, known, dynamic
using Reexport: Reexport, @reexport
using Statistics: mean

import LuxCore: AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer,
                initialparameters, initialstates, parameterlength, statelength, outputsize,
                update_state, trainmode, testmode, setup, apply, replicate

@reexport using LuxCore, LuxLib, MLDataDevices, WeightInitializers
using NNlib: NNlib, DenseConvDims, PoolDims, logsigmoid, logsoftmax, maxpool, meanpool,
             pixel_shuffle, sigmoid_fast, tanh_fast

const CRC = ChainRulesCore

const NAME_TYPE = Union{Nothing, String, Symbol}
const Optional{T} = Union{T, Nothing}

is_extension_loaded(::Val) = false
is_extension_loaded(::Val{:ForwardDiff}) = true

# Preferences
include("preferences.jl")

# Utilities
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
include("layers/pooling.jl")
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
include("helpers/size_propagator.jl")

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

# Deprecations for v2
include("deprecations.jl")

# Layers
export Chain, Parallel, SkipConnection, PairwiseFusion, BranchLayer, Maxout, RepeatedLayer
export Bilinear, Dense, Embedding, Scale
export Conv, ConvTranspose, Upsample, PixelShuffle
export MaxPool, MeanPool, LPPool, GlobalMaxPool, GlobalMeanPool, GlobalLPPool,
       AdaptiveMaxPool, AdaptiveMeanPool, AdaptiveLPPool
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

export f16, f32, f64, bf16
export match_eltype

export FromFluxAdaptor, FluxLayer
export ToSimpleChainsAdaptor, SimpleChainsLayer

export MPIBackend, NCCLBackend, DistributedUtils

export LuxOps

# Unexported functions that are part of the public API
@compat public Experimental
@compat public set_dispatch_doctor_preferences!

# NNlib.jl reexports
## Functional API for common layers. Recommended to use the LuxLib versions
using NNlib: ConvDims, DenseConvDims, PoolDims, batched_adjoint, batched_mul, batched_mul!,
             batched_transpose, batched_vec, bias_act!, conv, conv!, conv_bias_act,
             conv_bias_act!, dot_product_attention, dot_product_attention_scores,
             make_causal_mask, lpnormpool, lpnormpool!, maxpool, maxpool!, meanpool,
             meanpool!, pixel_shuffle, imrotate, ∇conv_data, ∇conv_data!, ∇conv_filter,
             ∇conv_filter!, ∇lpnormpool, ∇lpnormpool!, ∇maxpool, ∇maxpool!, ∇meanpool,
             ∇meanpool!, ∇imrotate
export ConvDims, DenseConvDims, PoolDims, batched_adjoint, batched_mul, batched_mul!,
       batched_transpose, batched_vec, bias_act!, conv, conv!, conv_bias_act,
       conv_bias_act!, dot_product_attention, dot_product_attention_scores,
       make_causal_mask, lpnormpool, lpnormpool!, maxpool, maxpool!, meanpool, meanpool!,
       pixel_shuffle, imrotate, ∇conv_data, ∇conv_data!, ∇conv_filter, ∇conv_filter!,
       ∇lpnormpool, ∇lpnormpool!, ∇maxpool, ∇maxpool!, ∇meanpool, ∇meanpool!, ∇imrotate

## Padding
using NNlib: pad_circular, pad_constant, pad_reflect, pad_repeat, pad_symmetric, pad_zeros
export pad_circular, pad_constant, pad_reflect, pad_repeat, pad_symmetric, pad_zeros

## Upsample
using NNlib: upsample_linear, upsample_bilinear, upsample_trilinear, upsample_nearest,
             ∇upsample_linear, ∇upsample_bilinear, ∇upsample_trilinear, ∇upsample_nearest
export upsample_linear, upsample_bilinear, upsample_trilinear, upsample_nearest,
       ∇upsample_linear, ∇upsample_bilinear, ∇upsample_trilinear, ∇upsample_nearest

## Activation Functions
using NNlib: σ, celu, elu, gelu, glu, hardsigmoid, hardswish, hardtanh, hardσ, leakyrelu,
             lisht, logcosh, logsigmoid, logσ, mish, relu, relu6, rrelu, selu, sigmoid,
             sigmoid_fast, softplus, softshrink, softsign, swish, tanhshrink, tanh_fast,
             thresholdrelu, trelu
export σ, celu, elu, gelu, glu, hardsigmoid, hardswish, hardtanh, hardσ, leakyrelu, lisht,
       logcosh, logsigmoid, logσ, mish, relu, relu6, rrelu, selu, sigmoid, sigmoid_fast,
       softplus, softshrink, softsign, swish, tanhshrink, tanh_fast, thresholdrelu, trelu

using NNlib: softmax, softmax!, logsoftmax, logsoftmax!, logsumexp, ∇logsoftmax,
             ∇logsoftmax!, ∇softmax, ∇softmax!
export softmax, softmax!, logsoftmax, logsoftmax!, logsumexp, ∇logsoftmax, ∇logsoftmax!,
       ∇softmax, ∇softmax!

end
