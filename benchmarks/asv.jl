# This is mostly a stub file. Allows up to use AirSpeedVelocity.jl to test package load
# times

using BenchmarkTools, Lux, Random

const SUITE = BenchmarkGroup()

SUITE["basics"] = BenchmarkGroup()

SUITE["basics"]["overhead"] = @benchmarkable begin dense(x, ps, st) end setup=begin
    dense = Dense(2, 3)
    x = ones(Float32, 2, 3)
    ps, st = Lux.setup(Xoshiro(), dense)
    dense(x, ps, st)
end
