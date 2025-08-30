@testsetup module ConvSetup
using LuxLib, LuxTestUtils, Random, Test, Zygote, NNlib

expand(_, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

function convfilter(
    gen_f::Function,
    ::Type{wT},
    filter::NTuple{N,Integer},
    ch::Pair{<:Integer,<:Integer};
    groups=1,
) where {wT,N}
    cin, cout = ch
    @assert cin % groups == 0 "Input channel dimension must be divisible by groups."
    @assert cout % groups == 0 "Output channel dimension must be divisible by groups."
    return gen_f(wT, filter..., cin ÷ groups, cout)
end

calc_padding(pad, ::NTuple{N}, dilation, stride) where {N} = expand(Val(2 * N), pad)

sumabs2conv(args...) = sum(abs2, fused_conv_bias_activation(args...))

function run_conv_testing(
    gen_f::Function,
    activation,
    kernel,
    stride,
    padding,
    hasbias,
    groups,
    Tw,
    Tx,
    aType,
    mode,
    ongpu,
)
    weight = aType(convfilter(gen_f, Tw, kernel, 4 => 8; groups))
    x = aType(gen_f(Tx, ntuple(Returns(4), length(kernel))..., 4, 2))
    bias = hasbias ? aType(gen_f(Tx, 8)) : nothing

    cdims = DenseConvDims(
        x,
        weight;
        stride,
        padding=calc_padding(padding, kernel, 1, stride),
        dilation=1,
        groups,
    )

    y = fused_conv_bias_activation(activation, weight, x, bias, cdims)

    generic_testing = !(mode == "amdgpu" && (Tx == Float64 || Tw == Float64))

    atol = 1.0f-3
    rtol = 1.0f-3

    if generic_testing
        y_generic = LuxLib.Impl.conv(x, weight, cdims)
        y_generic = if bias === nothing
            activation.(y_generic)
        else
            activation.(y_generic .+ LuxLib.Impl.reshape_bias(y_generic, bias))
        end
        # Operation reordering has an effect on the accuracy of the results
        @test y ≈ y_generic atol = atol rtol = rtol
    end

    @test eltype(y) == promote_type(Tw, Tx)

    @test @inferred(fused_conv_bias_activation(activation, weight, x, bias, cdims)) isa Any
    @jet fused_conv_bias_activation(activation, weight, x, bias, cdims)

    @test_gradients(sumabs2conv, activation, weight, x, bias, cdims; atol, rtol)
end

anonact = x -> gelu(x)

# const ELTYPES = [(Float32, Float32), (Float32, Float64), (Float64, Float64)]
const ELTYPES = [(Float32, Float32), (Float64, Float64)]
# const ACTIVATIONS = [
#     identity, tanh, tanh_fast, sigmoid, sigmoid_fast, relu, gelu, swish, anonact
# ]
const ACTIVATIONS = [identity, sigmoid, gelu]

const ALL_TEST_CONFIGS = Iterators.product(
    ELTYPES,
    (true, false),
    ACTIVATIONS,
    (
        ((2,), (1,), (1,), 1),
        ((2, 2), (1, 1), (1, 1), 1),
        ((2, 2), (0, 0), (2, 2), 1),
        ((2, 2), (0, 0), (1, 1), 2),
    ),
)

const TEST_BLOCKS = collect(
    Iterators.partition(ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 2))
)

export expand, convfilter, calc_padding, anonact, TEST_BLOCKS, run_conv_testing

end

@testitem "Fused Conv: Group 1" tags = [:common] setup = [SharedTestSetup, ConvSetup] begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "$(Tw) x $(Tx) hasbias: $(hasbias) activation: $(activation) kernel: $(kernel) padding: $(padding) stride: $(stride) groups: $(groups)" for (
            (Tx, Tw), hasbias, activation, (kernel, padding, stride, groups)
        ) in TEST_BLOCKS[1]
            !fp64 && (Tx == Float64 || Tw == Float64) && continue
            run_conv_testing(
                generate_fixed_array,
                activation,
                kernel,
                stride,
                padding,
                hasbias,
                groups,
                Tw,
                Tx,
                aType,
                mode,
                ongpu,
            )
        end
    end
end

@testitem "Fused Conv: Group 2" tags = [:common] setup = [SharedTestSetup, ConvSetup] begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "$(Tw) x $(Tx) hasbias: $(hasbias) activation: $(activation) kernel: $(kernel) padding: $(padding) stride: $(stride) groups: $(groups)" for (
            (Tx, Tw), hasbias, activation, (kernel, padding, stride, groups)
        ) in TEST_BLOCKS[2]
            !fp64 && (Tx == Float64 || Tw == Float64) && continue
            run_conv_testing(
                generate_fixed_array,
                activation,
                kernel,
                stride,
                padding,
                hasbias,
                groups,
                Tw,
                Tx,
                aType,
                mode,
                ongpu,
            )
        end
    end
end
