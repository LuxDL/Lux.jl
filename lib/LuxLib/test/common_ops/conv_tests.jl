@testsetup module ConvSetup
using LuxLib, LuxTestUtils, Random, Test, Zygote, Enzyme, NNlib
using LuxTestUtils: @jet, @test_gradients
using DispatchDoctor: allow_unstable

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
        hasbias, groups, Tw, Tx, aType, mode, on_gpu)
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
        catch
            @test_broken false
        end
    end

    if !on_gpu
        _, ∂w_zyg, ∂x_zyg, ∂b_zyg = Zygote.gradient(__f, activation, weight, x, bias, cdims)

        ∂w_enz = Enzyme.make_zero(weight)
        ∂x_enz = Enzyme.make_zero(x)
        ∂b = if hasbias
            Duplicated(bias, Enzyme.make_zero(bias))
        else
            Const(nothing)
        end
        Enzyme.autodiff(Reverse, __f, Active, Const(activation), Duplicated(weight, ∂w_enz),
            Duplicated(x, ∂x_enz), ∂b, Const(cdims))

        @test ∂w_zyg≈∂w_enz rtol=rtol atol=atol
        @test ∂x_zyg≈∂x_enz rtol=rtol atol=atol
        hasbias && @test ∂b_zyg≈∂b.dval rtol=rtol atol=atol
    end

    mp = Tx != Tw
    skipt = (mp && on_gpu) || (mode == "amdgpu" && (Tx == Float64 || Tw == Float64))
    allow_unstable() do
        @eval @test_gradients $__f $activation $weight $x $bias $cdims gpu_testing=$on_gpu atol=$atol rtol=$rtol skip_reverse_diff=$(mp) skip_finite_differences=$(mp) skip_tracker=$(skipt)
    end
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

@testitem "Fused Conv: Group 1" tags=[:common_ops] setup=[SharedTestSetup, ConvSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "$(Tw) x $(Tx) hasbias: $(hasbias) activation: $(activation) kernel: $(kernel) padding: $(padding) stride: $(stride) groups: $(groups)" for ((Tx, Tw), hasbias, activation, (kernel, padding, stride, groups)) in TEST_BLOCKS[1]
            run_conv_testing(__generate_fixed_array, activation, kernel,
                stride, padding, hasbias, groups, Tw, Tx, aType, mode, on_gpu)
        end
    end
end

@testitem "Fused Conv: Group 2" tags=[:common_ops] setup=[SharedTestSetup, ConvSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "$(Tw) x $(Tx) hasbias: $(hasbias) activation: $(activation) kernel: $(kernel) padding: $(padding) stride: $(stride) groups: $(groups)" for ((Tx, Tw), hasbias, activation, (kernel, padding, stride, groups)) in TEST_BLOCKS[2]
            run_conv_testing(__generate_fixed_array, activation, kernel,
                stride, padding, hasbias, groups, Tw, Tx, aType, mode, on_gpu)
        end
    end
end

@testitem "Fused Conv: Group 3" tags=[:common_ops] setup=[SharedTestSetup, ConvSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "$(Tw) x $(Tx) hasbias: $(hasbias) activation: $(activation) kernel: $(kernel) padding: $(padding) stride: $(stride) groups: $(groups)" for ((Tx, Tw), hasbias, activation, (kernel, padding, stride, groups)) in TEST_BLOCKS[3]
            run_conv_testing(__generate_fixed_array, activation, kernel,
                stride, padding, hasbias, groups, Tw, Tx, aType, mode, on_gpu)
        end
    end
end

@testitem "Fused Conv: Group 4" tags=[:common_ops] setup=[SharedTestSetup, ConvSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "$(Tw) x $(Tx) hasbias: $(hasbias) activation: $(activation) kernel: $(kernel) padding: $(padding) stride: $(stride) groups: $(groups)" for ((Tx, Tw), hasbias, activation, (kernel, padding, stride, groups)) in TEST_BLOCKS[4]
            run_conv_testing(__generate_fixed_array, activation, kernel,
                stride, padding, hasbias, groups, Tw, Tx, aType, mode, on_gpu)
        end
    end
end

@testitem "Fused Conv: Group 5" tags=[:common_ops] setup=[SharedTestSetup, ConvSetup] begin
    @testset "$mode" for (mode, aType, on_gpu) in MODES
        @testset "$(Tw) x $(Tx) hasbias: $(hasbias) activation: $(activation) kernel: $(kernel) padding: $(padding) stride: $(stride) groups: $(groups)" for ((Tx, Tw), hasbias, activation, (kernel, padding, stride, groups)) in TEST_BLOCKS[5]
            run_conv_testing(__generate_fixed_array, activation, kernel,
                stride, padding, hasbias, groups, Tw, Tx, aType, mode, on_gpu)
        end
    end
end
