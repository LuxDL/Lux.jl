include("../shared_testsetup.jl")

using LuxLib, LuxTestUtils, Random, Test, NNlib

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

const ELTYPES = [(Float32, Float32), (Float64, Float64)]
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

# The generic-device path (Metal, oneAPI, ...) cannot use NNlib's im2col routines because
# they scalar-index the input arrays. Check that the replacements agree with NNlib.
const FALLBACK_TEST_CONFIGS = [
    ((2,), (1,), (1,), 1),
    ((1,), (0,), (1,), 1),
    ((3,), (1,), (2,), 1),
    ((2, 2), (1, 1), (1, 1), 1),
    ((2, 2), (0, 0), (2, 2), 1),
    ((3, 3), (2, 2), (1, 1), 1),
    ((2, 2, 2), (1, 1, 1), (1, 1, 1), 1),
]

@testset "Slow Conv Fallbacks" begin
    rng = StableRNG(12345)

    @testset "kernel: $(kernel) padding: $(padding) stride: $(stride) flipkernel: $(flipkernel)" for (
            kernel, padding, stride, groups
        ) in FALLBACK_TEST_CONFIGS,
        flipkernel in (true, false)

        weight = convfilter(
            (T, sz...) -> randn(rng, T, sz...), Float32, kernel, 4 => 8; groups
        )
        x = randn(rng, Float32, ntuple(Returns(6), length(kernel))..., 4, 2)

        cdims = DenseConvDims(
            x,
            weight;
            stride,
            padding=calc_padding(padding, kernel, 1, stride),
            dilation=1,
            groups,
            flipkernel,
        )

        y = NNlib.conv(x, weight, cdims)
        dy = randn(rng, Float32, size(y)...)

        # forward, for reference: this path is already exercised on Metal/oneAPI
        @test LuxLib.Impl.fallback_slow_conv(CPUDevice, x, weight, cdims) ≈ y atol = 1.0f-3 rtol =
            1.0f-3

        ∂x = LuxLib.Impl.fallback_slow_∇conv_data(CPUDevice, dy, weight, cdims)
        @test size(∂x) == size(x)
        @test ∂x ≈ NNlib.∇conv_data(dy, weight, cdims) atol = 1.0f-3 rtol = 1.0f-3

        ∂w = LuxLib.Impl.fallback_slow_∇conv_filter(CPUDevice, x, dy, cdims)
        @test size(∂w) == size(weight)
        @test ∂w ≈ NNlib.∇conv_filter(x, dy, cdims) atol = 1.0f-3 rtol = 1.0f-3
    end
end

@testset "Fused Conv" begin
    @testset "$mode" for (mode, aType, ongpu, fp64) in MODES
        @testset "$(Tw) x $(Tx) hasbias: $(hasbias) activation: $(activation) kernel: $(kernel) padding: $(padding) stride: $(stride) groups: $(groups)" for (
                (Tx, Tw), hasbias, activation, (kernel, padding, stride, groups)
            ) in ALL_TEST_CONFIGS

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
