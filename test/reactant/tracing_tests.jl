@testitem "Tracing" tags = [:reactant] begin
    using Reactant, Lux, Random

    model = Chain(Dense(2 => 3, relu), BatchNorm(3), Dense(3 => 2))
    ps, st = Lux.setup(Random.default_rng(), model)

    smodel = StatefulLuxLayer(model, ps, st)
    smodel_ra = Reactant.to_rarray(smodel)

    @test get_device_type(smodel_ra.ps) <: ReactantDevice
    @test get_device_type(smodel_ra.st) <: ReactantDevice
    @test smodel_ra.st_any === nothing
    @test smodel_ra.fixed_state_type == smodel.fixed_state_type

    smodel = StatefulLuxLayer{false}(model, ps, st)
    smodel_ra = Reactant.to_rarray(smodel)

    @test get_device_type(smodel_ra.ps) <: ReactantDevice
    @test get_device_type(smodel_ra.st_any) <: ReactantDevice
    @test smodel_ra.st === nothing
    @test smodel_ra.fixed_state_type == smodel.fixed_state_type
end
