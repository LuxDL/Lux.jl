@testsetup module ConvSetup
using LuxLib, LuxTestUtils, Random, Test, Zygote, NNlib

_expand(N, i::Tuple) = i
_expand(N, i::Integer) = ntuple(_ -> i, N)

function _convfilter(gen_f::Function, ::Type{wT}, filter::NTuple{N, Integer},
        ch::Pair{<:Integer, <:Integer}; groups=1) where {wT, N}
    cin, cout = ch
    @assert cin % groups==0 "Input channel dimension must be divisible by groups."
    @assert cout % groups==0 "Output channel dimension must be divisible by groups."
    return gen_f(wT, filter..., cin ÷ groups, cout)
end

_calc_padding(pad, ::NTuple{N}, dilation, stride) where {N} = _expand(Val(2 * N), pad)

function run_conv_testing(gen_f::Function, activation, kernel, stride, padding,
        hasbias, groups, Tw, Tx, aType, mode, ongpu)
    weight = _convfilter(gen_f, Tw, kernel, 4 => 8; groups) |> aType
    x = gen_f(Tx, ntuple(Returns(4), length(kernel))..., 4, 2) |> aType
    bias = hasbias ? aType(gen_f(Tx, 8)) : nothing

    cdims = DenseConvDims(
        x, weight; stride, padding=_calc_padding(padding, kernel, 1, stride),
        dilation=1, groups)

    y = fused_conv_bias_activation(activation, weight, x, bias, cdims)

    y_generic = LuxLib._generic_conv_bias_activation(activation, weight, x, bias, cdims)

    fp16 = Tx == Float16 || Tw == Float16
    atol = fp16 ? 1.0f-1 : 1.0f-3
    rtol = fp16 ? 1.0f-1 : 1.0f-3
    # Operation reordering has an effect on the accuracy of the results
    @test y≈y_generic atol=atol rtol=rtol
    @test eltype(y) == promote_type(Tw, Tx)

    @test @inferred(fused_conv_bias_activation(activation, weight, x, bias, cdims)) isa Any
    @jet fused_conv_bias_activation(activation, weight, x, bias, cdims)

    __f = (σ, w, x, b, cdims) -> sum(abs2, fused_conv_bias_activation(σ, w, x, b, cdims))

    if mode != "amdgpu" && activation !== anonact
        @test @inferred(Zygote.gradient(__f, activation, weight, x, bias, cdims)) isa Any
    else
        try
            @inferred(Zygote.gradient(__f, activation, weight, x, bias, cdims))
            @test true
        catch e
            e isa ErrorException || rethrow()
            @test_broken false
        end
    end

    __f_grad = let activation = activation, cdims = cdims
        (w, x, b) -> __f(activation, w, x, b, cdims)
    end

    skip_backends = []
    mp = Tx != Tw
    mp && push!(skip_backends, AutoReverseDiff())
    ((mp && ongpu) || (mode == "amdgpu" && (Tx == Float64 || Tw == Float64))) &&
        push!(skip_backends, AutoTracker())
    test_gradients(__f_grad, weight, x, bias; atol, rtol, skip_backends,
        soft_fail=(fp16 ? [AutoFiniteDiff()] : []))
end

anonact = x -> gelu(x)

const ELTYPES = [(Float16, Float16), (Float32, Float16), (Float32, Float32),
    (Float32, Float64), (Float64, Float64)]
const ACTIVATIONS = [
    identity, tanh, tanh_fast, sigmoid, sigmoid_fast, relu, gelu, swish, anonact]

const ALL_TEST_CONFIGS = Iterators.product(ELTYPES,
    (true, false),
    ACTIVATIONS,
    (((2,), (1,), (1,), 1), ((2, 2), (1, 1), (1, 1), 1),
        ((2, 2), (0, 0), (2, 2), 1), ((2, 2), (0, 0), (1, 1), 2)))

const TEST_BLOCKS = collect(Iterators.partition(
    ALL_TEST_CONFIGS, ceil(Int, length(ALL_TEST_CONFIGS) / 5)))

export _expand, _convfilter, _calc_padding, anonact, TEST_BLOCKS, run_conv_testing

end

@testitem "Fused Conv: Group 1" tags=[:conv] setup=[SharedTestSetup, ConvSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "$(Tw) x $(Tx) hasbias: $(hasbias) activation: $(activation) kernel: $(kernel) padding: $(padding) stride: $(stride) groups: $(groups)" for ((Tx, Tw), hasbias, activation, (kernel, padding, stride, groups)) in TEST_BLOCKS[1]
            run_conv_testing(__generate_fixed_array, activation, kernel, stride,
                padding, hasbias, groups, Tw, Tx, aType, mode, ongpu)
        end
    end
end

@testitem "Fused Conv: Group 2" tags=[:conv] setup=[SharedTestSetup, ConvSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "$(Tw) x $(Tx) hasbias: $(hasbias) activation: $(activation) kernel: $(kernel) padding: $(padding) stride: $(stride) groups: $(groups)" for ((Tx, Tw), hasbias, activation, (kernel, padding, stride, groups)) in TEST_BLOCKS[2]
            run_conv_testing(__generate_fixed_array, activation, kernel, stride,
                padding, hasbias, groups, Tw, Tx, aType, mode, ongpu)
        end
    end
end

@testitem "Fused Conv: Group 3" tags=[:conv] setup=[SharedTestSetup, ConvSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "$(Tw) x $(Tx) hasbias: $(hasbias) activation: $(activation) kernel: $(kernel) padding: $(padding) stride: $(stride) groups: $(groups)" for ((Tx, Tw), hasbias, activation, (kernel, padding, stride, groups)) in TEST_BLOCKS[3]
            run_conv_testing(__generate_fixed_array, activation, kernel, stride,
                padding, hasbias, groups, Tw, Tx, aType, mode, ongpu)
        end
    end
end

@testitem "Fused Conv: Group 4" tags=[:conv] setup=[SharedTestSetup, ConvSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "$(Tw) x $(Tx) hasbias: $(hasbias) activation: $(activation) kernel: $(kernel) padding: $(padding) stride: $(stride) groups: $(groups)" for ((Tx, Tw), hasbias, activation, (kernel, padding, stride, groups)) in TEST_BLOCKS[4]
            run_conv_testing(__generate_fixed_array, activation, kernel, stride,
                padding, hasbias, groups, Tw, Tx, aType, mode, ongpu)
        end
    end
end

@testitem "Fused Conv: Group 5" tags=[:conv] setup=[SharedTestSetup, ConvSetup] begin
    @testset "$mode" for (mode, aType, ongpu) in MODES
        @testset "$(Tw) x $(Tx) hasbias: $(hasbias) activation: $(activation) kernel: $(kernel) padding: $(padding) stride: $(stride) groups: $(groups)" for ((Tx, Tw), hasbias, activation, (kernel, padding, stride, groups)) in TEST_BLOCKS[5]
            run_conv_testing(__generate_fixed_array, activation, kernel, stride,
                padding, hasbias, groups, Tw, Tx, aType, mode, ongpu)
        end
    end
end
