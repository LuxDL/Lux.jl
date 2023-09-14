using LuxLib, ForwardDiff, Zygote, Test
using ComponentArrays

include("test_utils.jl")

struct LuxLibTestTag end

# Computes (∂f/∂x)u
function jvp_forwarddiff(f, x, u)
    uu = reshape(u, axes(x))
    y = ForwardDiff.Dual{typeof(ForwardDiff.Tag(LuxLibTestTag(), eltype(x))), eltype(x),
        1}.(x, ForwardDiff.Partials.(tuple.(uu)))
    return vec(ForwardDiff.partials.(vec(f(y)), 1))
end

function jvp_forwarddiff(f, x::ComponentArray, u)
    xx = getdata(x)
    uu = vec(u)
    y = ComponentArray(ForwardDiff.Dual{typeof(ForwardDiff.Tag(LuxLibTestTag(),
                eltype(x))), eltype(x), 1}.(xx, ForwardDiff.Partials.(tuple.(uu))),
        getaxes(x))
    return vec(ForwardDiff.partials.(vec(f(y)), 1))
end

## This exists exclusively for testing. It has horrifying performance implications
function jvp_forwarddiff_concrete(f, x, u)
    Jₓ = ForwardDiff.jacobian(f, x)
    return Jₓ * vec(u)
end

function jvp_zygote(f, x, u)
    Jₓ = only(Zygote.jacobian(f, x))
    return Jₓ * vec(u)
end

function test_jvp_computation(f, x, u)
    jvp₁ = jvp_forwarddiff(f, x, u)
    if !(x isa ComponentArray)
        # ComponentArray + ForwardDiff on GPU don't play nice
        jvp₂ = jvp_forwarddiff_concrete(f, x, u)
        @test check_approx(jvp₁, jvp₂; atol=1e-5, rtol=1e-5)
    end

    jvp₃ = jvp_zygote(f, x, u)
    @test check_approx(jvp₁, jvp₃; atol=1e-5, rtol=1e-5)
end

@testset "$mode: Jacobian Vector Products" for (mode, aType, on_gpu) in MODES
    @testset "$(op)(; flipped = $flipped)" for flipped in (true, false),
        op in (depthwiseconv, conv)

        op === depthwiseconv && mode == "AMDGPU" && continue

        input_dims = [(2, 4, 2, 1, 3), (4, 4, 1, 3), (4, 4, 3, 2), (4, 1, 3), (4, 3, 2)]
        weight_dims = if op === conv
            [(2, 2, 2, 1, 4), (3, 3, 1, 4), (3, 3, 3, 2), (3, 1, 4), (3, 3, 2)]
        else
            [(2, 2, 2, 1, 1), (3, 3, 1, 1), (3, 3, 3, 3), (3, 1, 1), (3, 3, 3)]
        end

        @testset "Input Dims: $(in_dims) | Weight Dims: $(w_dims)" for (in_dims, w_dims) in zip(input_dims,
            weight_dims)
            x = randn(Float32, in_dims...) |> aType
            w = randn(Float32, w_dims...) |> aType
            ux = randn(Float32, size(x)...) |> aType
            uw = randn(Float32, size(w)...) |> aType
            u = randn(Float32, length(x) + length(w)) |> aType

            test_jvp_computation(x -> op(x, w; flipped), x, ux)
            test_jvp_computation(w -> op(x, w; flipped), w, uw)
            test_jvp_computation(xw -> op(xw.x, xw.w; flipped), ComponentArray(; x, w), u)
        end
    end
end
