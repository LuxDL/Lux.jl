module Lux

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using Adapt: Adapt, adapt
    using ArrayInterface: ArrayInterface
    using ChainRulesCore: ChainRulesCore, AbstractZero, HasReverseMode, NoTangent,
                          ProjectTo, RuleConfig, ZeroTangent
    using ConcreteStructs: @concrete
    using FastClosures: @closure
    using Functors: Functors, fmap
    using GPUArraysCore: GPUArraysCore
    using LinearAlgebra: LinearAlgebra
    using Markdown: @doc_str
    using Preferences: @load_preference
    using Random: Random, AbstractRNG
    using Reexport: @reexport
    using Setfield: Setfield, @set!
    using Statistics: Statistics, mean
    using WeightInitializers: WeightInitializers, glorot_uniform, ones32, randn32, zeros32

    using LuxCore, LuxLib, LuxDeviceUtils, WeightInitializers
    import LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer,
                    initialparameters, initialstates, parameterlength, statelength,
                    inputsize, outputsize, update_state, trainmode, testmode, setup, apply,
                    display_name, replicate
    import LuxDeviceUtils: get_device, set_device!
end

@reexport using LuxCore, LuxLib, LuxDeviceUtils, WeightInitializers

const CRC = ChainRulesCore

const NAME_TYPE = Union{Nothing, String, Symbol}

@inline __is_extension_loaded(x) = Val(false)

# Utilities
include("utils.jl")

# Layer Implementations
include("layers/basic.jl")
include("layers/containers.jl")
include("layers/normalize.jl")
include("layers/conv.jl")
include("layers/dropout.jl")
include("layers/recurrent.jl")

# Pretty Printing
include("layers/display.jl")

# AutoDiff
include("chainrules.jl")

# Experimental
include("contrib/contrib.jl")

# Helpful Functionalities
include("helpers/stateful.jl")

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
export cpu, gpu
export Chain, Parallel, SkipConnection, PairwiseFusion, BranchLayer, Maxout, RepeatedLayer
export Bilinear, Dense, Embedding, Scale
export Conv, ConvTranspose, CrossCor, MaxPool, MeanPool, GlobalMaxPool, GlobalMeanPool,
       AdaptiveMaxPool, AdaptiveMeanPool, Upsample, PixelShuffle
export AlphaDropout, Dropout, VariationalHiddenDropout
export BatchNorm, GroupNorm, InstanceNorm, LayerNorm
export WeightNorm
export NoOpLayer, ReshapeLayer, SelectDim, FlattenLayer, WrappedFunction
export RNNCell, LSTMCell, GRUCell, Recurrence, StatefulRecurrentCell
export SamePad, TimeLastIndex, BatchLastIndex

export StatefulLuxLayer

export f16, f32, f64

export transform, FromFluxAdaptor, ToSimpleChainsAdaptor, FluxLayer, SimpleChainsLayer

export MPIBackend, NCCLBackend, DistributedUtils

end
