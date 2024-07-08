@testitem "Efficient JVPs" tags=[:others] setup=[SharedTestSetup] begin
    using ForwardDiff, Zygote, ComponentArrays

    # Computes (∂f/∂x)u
    function jvp_forwarddiff(f::F, x, u) where {F}
        uu = reshape(u, axes(x))
        y = ForwardDiff.Dual{typeof(ForwardDiff.Tag(f, eltype(x))), eltype(x),
            1}.(x, ForwardDiff.Partials.(tuple.(uu)))
        return vec(ForwardDiff.partials.(vec(f(y)), 1))
    end

    function jvp_forwarddiff(f::F, x::ComponentArray, u) where {F}
        xx = getdata(x)
        uu = vec(u)
        y = ComponentArray(
            ForwardDiff.Dual{typeof(ForwardDiff.Tag(f, eltype(x))), eltype(x),
                1}.(xx, ForwardDiff.Partials.(tuple.(uu))),
            getaxes(x))
        return vec(ForwardDiff.partials.(vec(f(y)), 1))
    end

    ## This exists exclusively for testing. It has horrifying performance implications
    jvp_forwarddiff_concrete(f::F, x, u) where {F} = ForwardDiff.jacobian(f, x) * vec(u)
    jvp_zygote(f::F, x, u) where {F} = only(Zygote.jacobian(f, x)) * vec(u)

    function test_jvp_computation(f::F, x, u, on_gpu, nested=false) where {F}
        jvp₁ = jvp_forwarddiff(f, x, u)
        if !(x isa ComponentArray && on_gpu)
            # ComponentArray + ForwardDiff on GPU don't play nice
            jvp₂ = jvp_forwarddiff_concrete(f, x, u)
            @test check_approx(jvp₁, jvp₂; atol=1e-5, rtol=1e-5)
        end

        if !nested
            jvp₃ = jvp_zygote(f, x, u)
            @test check_approx(jvp₁, jvp₃; atol=1e-5, rtol=1e-5)
        end
    end

    @testset "$(mode): Jacobian Vector Products" for (mode, aType, on_gpu) in MODES
        @testset "$(op)(; flipped = $flipped)" for flipped in (true, false),
            op in (depthwiseconv, conv)

            op === depthwiseconv && on_gpu && continue

            input_dims = [(2, 4, 2, 1, 3), (4, 4, 1, 3), (4, 4, 3, 2), (4, 1, 3), (4, 3, 2)]
            weight_dims = if op === depthwiseconv
                [(2, 2, 2, 1, 1), (3, 3, 1, 1), (3, 3, 3, 3), (3, 1, 1), (3, 3, 3)]
            else
                [(2, 2, 2, 1, 4), (3, 3, 1, 4), (3, 3, 3, 2), (3, 1, 4), (3, 3, 2)]
            end

            @testset "Input Dims: $(in_dims) | Weight Dims: $(w_dims)" for (in_dims, w_dims) in zip(
                input_dims, weight_dims)
                x = randn(Float32, in_dims...) |> aType
                w = randn(Float32, w_dims...) |> aType
                ux = randn(Float32, size(x)...) |> aType
                uw = randn(Float32, size(w)...) |> aType
                u = randn(Float32, length(x) + length(w)) |> aType

                test_jvp_computation(x -> op(x, w; flipped), x, ux, on_gpu)
                test_jvp_computation(w -> op(x, w; flipped), w, uw, on_gpu)
                test_jvp_computation(
                    xw -> op(xw.x, xw.w; flipped), ComponentArray(; x, w), u, on_gpu)

                op === depthwiseconv && continue

                # Zygote.gradient here is used to test the ∇conv_data and ∇conv_filter
                # functions. Also implicitly tests nested AD
                test_jvp_computation(
                    x -> only(Zygote.gradient(w -> sum(abs2, op(x, w; flipped)), w)),
                    x, ux, on_gpu, true)
                test_jvp_computation(
                    x -> only(Zygote.gradient(x -> sum(abs2, op(x, w; flipped)), x)),
                    x, ux, on_gpu, true)
                test_jvp_computation(
                    w -> only(Zygote.gradient(x -> sum(abs2, op(x, w; flipped)), x)),
                    w, uw, on_gpu, true)
                test_jvp_computation(
                    w -> only(Zygote.gradient(w -> sum(abs2, op(x, w; flipped)), w)),
                    w, uw, on_gpu, true)
                test_jvp_computation(
                    xw -> only(Zygote.gradient(
                        xw -> sum(abs2, op(xw.x, xw.w; flipped)), xw)),
                    ComponentArray(; x, w),
                    u,
                    on_gpu,
                    true)
            end
        end
    end
end

@testitem "ForwardDiff dropout" tags=[:common_ops] setup=[SharedTestSetup] begin
    using ForwardDiff

    rng = StableRNG(12345)

    @testset "$mode: dropout" for (mode, aType, on_gpu) in MODES
        x = randn(rng, Float32, 10, 2) |> aType
        x_dual = ForwardDiff.Dual.(x)

        @test_nowarn dropout(rng, x_dual, 0.5f0, Val(true), 2.0f0, :)

        x_dropout = dropout(rng, x, 0.5f0, Val(true); dims=:)[1]
        x_dual_dropout = ForwardDiff.value.(dropout(rng, x_dual, 0.5f0, Val(true); dims=:)[1])

        @test check_approx(x_dropout, x_dual_dropout)
    end
end
