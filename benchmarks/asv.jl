using BenchmarkTools, Lux, Random, Reactant, MLDataDevices

const SUITE = BenchmarkGroup()

SUITE["basics"] = BenchmarkGroup()

SUITE["basics"]["dense (first run)"] = @benchmarkable begin
    dense(x, ps, st)
end setup=begin
    dense = Dense(2, 3)
    x = ones(Float32, 2, 3)
    ps, st = Lux.setup(Xoshiro(), dense)
end

SUITE["basics"]["dense"] = @benchmarkable begin
    dense(x, ps, st)
end setup=begin
    dense = Dense(2, 3)
    x = ones(Float32, 2, 3)
    ps, st = Lux.setup(Xoshiro(), dense)
    dense(x, ps, st)
end

SUITE["basics"]["conv (first run)"] = @benchmarkable begin
    conv(x, ps, st)
end setup=begin
    conv = Conv((2, 2), 3 => 16)
    x = ones(Float32, 2, 2, 3, 4)
    ps, st = Lux.setup(Xoshiro(), conv)
end

SUITE["basics"]["conv"] = @benchmarkable begin
    conv(x, ps, st)
end setup=begin
    conv = Conv((2, 2), 3 => 16)
    x = ones(Float32, 2, 2, 3, 4)
    ps, st = Lux.setup(Xoshiro(), conv)
    conv(x, ps, st)
end

SUITE["basics"]["MHA (first run)"] = @benchmarkable begin
    mha((q, k, v), ps, st)
end setup=begin
    mha = MultiHeadAttention(4; nheads=2)
    q = ones(Float32, 4, 3, 2)
    k = ones(Float32, 4, 3, 2)
    v = ones(Float32, 4, 3, 2)
    ps, st = Lux.setup(Xoshiro(), mha)
end

SUITE["basics"]["MHA"] = @benchmarkable begin
    mha((q, k, v), ps, st)
end setup=begin
    mha = MultiHeadAttention(4; nheads=2)
    q = ones(Float32, 4, 3, 2)
    k = ones(Float32, 4, 3, 2)
    v = ones(Float32, 4, 3, 2)
    ps, st = Lux.setup(Xoshiro(), mha)
    mha((q, k, v), ps, st)
end

SUITE["basics"]["dense reactant (comp + run)"] = @benchmarkable begin
    @jit dense(x, ps, st)
end setup=begin
    dense = Dense(2, 3)
    x = ones(Float32, 2, 3) |> Reactant.to_rarray
    ps, st = Lux.setup(Xoshiro(), dense) |> Reactant.to_rarray
end

SUITE["basics"]["dense reactant"] = @benchmarkable begin
    compiled_fn(x, ps, st)
end setup=begin
    dense = Dense(2, 3)
    x = ones(Float32, 2, 3) |> Reactant.to_rarray
    ps, st = Lux.setup(Xoshiro(), dense) |> Reactant.to_rarray
    compiled_fn = @compile dense(x, ps, st)
end

SUITE["basics"]["conv reactant (comp + run)"] = @benchmarkable begin
    @jit conv(x, ps, st)
end setup=begin
    conv = Conv((2, 2), 3 => 16)
    x = ones(Float32, 2, 2, 3, 4) |> Reactant.to_rarray
    ps, st = Lux.setup(Xoshiro(), conv) |> Reactant.to_rarray
end

SUITE["basics"]["conv reactant"] = @benchmarkable begin
    compiled_fn(x, ps, st)
end setup=begin
    conv = Conv((2, 2), 3 => 16)
    x = ones(Float32, 2, 2, 3, 4) |> Reactant.to_rarray
    ps, st = Lux.setup(Xoshiro(), conv) |> Reactant.to_rarray
    compiled_fn = @compile conv(x, ps, st)
end

SUITE["basics"]["MHA reactant (comp + run)"] = @benchmarkable begin
    @jit mha((q, k, v), ps, st)
end setup=begin
    mha = MultiHeadAttention(4; nheads=2)
    q = ones(Float32, 4, 3, 2) |> Reactant.to_rarray
    k = ones(Float32, 4, 3, 2) |> Reactant.to_rarray
    v = ones(Float32, 4, 3, 2) |> Reactant.to_rarray
    ps, st = Lux.setup(Xoshiro(), mha) |> Reactant.to_rarray
end

SUITE["basics"]["MHA reactant"] = @benchmarkable begin
    compiled_fn((q, k, v), ps, st)
end setup=begin
    mha = MultiHeadAttention(4; nheads=2)
    q = ones(Float32, 4, 3, 2) |> Reactant.to_rarray
    k = ones(Float32, 4, 3, 2) |> Reactant.to_rarray
    v = ones(Float32, 4, 3, 2) |> Reactant.to_rarray
    ps, st = Lux.setup(Xoshiro(), mha) |> Reactant.to_rarray
    compiled_fn = @compile mha((q, k, v), ps, st)
end
