using Lux, Functors, Test, LuxCUDA

include("test_utils.jl")

CUDA.allowscalar(false)

if LuxCUDA.functional()
    @testset "Device Transfer" begin
        ps = (a=(c=zeros(10, 1), d=1),
            b=ones(10, 1),
            e=:c,
            d="string",
            rng=get_stable_rng(12345))

        ps_gpu = ps |> gpu
        @test ps_gpu.a.c isa CuArray
        @test ps_gpu.b isa CuArray
        @test ps_gpu.a.d == ps.a.d
        @test ps_gpu.e == ps.e
        @test ps_gpu.d == ps.d
        @test ps_gpu.rng == ps.rng

        ps_cpu = ps_gpu |> cpu
        @test ps_cpu.a.c isa Array
        @test ps_cpu.b isa Array
        @test ps_cpu.a.c == ps.a.c
        @test ps_cpu.b == ps.b
        @test ps_cpu.a.d == ps.a.d
        @test ps_cpu.e == ps.e
        @test ps_cpu.d == ps.d
        @test ps_cpu.rng == ps.rng
    end
end
