module LuxLibSLEEFPiratesExt

using ChainRulesCore: ChainRulesCore
using NNlib: NNlib
using SLEEFPirates: SLEEFPirates

using LuxLib: Numeric, Impl

const CRC = ChainRulesCore

sigmoid_fast(x::Number) = SLEEFPirates.sigmoid_fast(x)
softplus(x::Number) = SLEEFPirates.softplus(x)
logsigmoid(x::Number) = -softplus(-x)
swish(x::Number) = Base.FastMath.mul_fast(x, sigmoid_fast(x))
lisht(x::Number) = Base.FastMath.mul_fast(x, tanh_fast(x))
tanh(x::Number) = SLEEFPirates.tanh(x)
tanh_fast(x::Number) = SLEEFPirates.tanh_fast(x)

for (f, dfdx) in [
    #! format: off
    (:sigmoid_fast, :(conj(Base.FastMath.mul_fast(Ω, Base.FastMath.sub_fast(1, Ω))))),
    (:softplus, :(sigmoid_fast(x))),
    (:logsigmoid, :(sigmoid_fast(-x))),
    (:swish, :(Base.FastMath.add_fast(Ω, Base.FastMath.mul_fast(sigmoid_fast(x), Base.FastMath.sub_fast(1, Ω))))),
    (:lisht, :(Base.FastMath.add_fast(x, Base.FastMath.mul_fast(tanh_fast(x), Base.FastMath.sub_fast(1, Ω))))),
    (:tanh, :(conj(Base.FastMath.sub_fast(1, Base.FastMath.mul_fast(Ω, Ω))))),
    (:tanh_fast, :(conj(Base.FastMath.sub_fast(1, Base.FastMath.mul_fast(Ω, Ω)))))
    #! format: on
]
    @eval CRC.@scalar_rule($f(x), $(dfdx))

    ∇f = Symbol(:∇broadcasted_, f)
    @eval function CRC.rrule(::typeof(Broadcast.broadcasted), ::typeof($f),
            x::Union{Numeric, Broadcast.Broadcasted})
        Ω = $(f).(x)
        function $(∇f)(dΩ)
            ∂x = CRC.InplaceableThunk(dx -> @.(dx+=dΩ * $(dfdx)), CRC.@thunk @.(dΩ*$(dfdx)))
            return CRC.NoTangent(), CRC.NoTangent(), ∂x
        end
        return Ω, $(∇f)
    end
end

for (fbase, ffast) in [
    #! format: off
    (NNlib.sigmoid_fast, sigmoid_fast),
    (NNlib.softplus, softplus),
    (NNlib.logsigmoid, logsigmoid),
    (NNlib.swish, swish),
    (NNlib.lisht, lisht),
    (Base.tanh, tanh),
    (NNlib.tanh_fast, tanh_fast)
    #! format: on
]
    @eval Impl.sleefpirates_fast_act(::typeof($fbase)) = $ffast
end

end
