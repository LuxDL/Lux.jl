using Lux, Functors, Random, Test

import CUDA

if CUDA.functional()
  using CUDA  # exports CuArray, etc
  @info "starting CUDA tests"
else
  @info "CUDA not functional, testing via JLArrays"
  using JLArrays
  JLArrays.allowscalar(false)

  # JLArrays provides a fake GPU array, for testing
  using Random, Adapt
  CUDA.cu(x) = jl(x)
  CuArray{T, N} = JLArray{T, N}

  function Lux.gpu(x)
    return fmap(x -> adapt(Lux.LuxCUDAAdaptor(), x), x; exclude=Lux._isleaf)
  end
end

@testset "Device Transfer" begin
  ps = (a=(c=zeros(10, 1), d=1), b=ones(10, 1), e=:c, d="string", rng=Random.default_rng())

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

  # Deprecated Functionality (Remove in v0.5)
  @test_deprecated cpu(Dense(10, 10))
  @test_deprecated gpu(Dense(10, 10))
end
