# https://github.com/JuliaML/LossFunctions.jl/blob/d07bacab74f4db420833106ebd0b438d22bd2014/src/losses/distance.jl#L213
for fnType in (typeof(sum), Any, typeof(mean))
    @eval function LossFunctionImpl.fused_agg(
            fn::$(fnType), lfn::LossFunctions.HuberLoss{T1}, x::TracedRArray{T2, N},
            y::TracedRArray{T3, N}) where {T1, T2, T3, N}
        T = promote_type(T1, T2, T3)
        delta = T(lfn.d)
        diff = x .- y
        abs_diff = abs.(diff)
        quadratic = abs2.(diff) ./ 2
        linear = (delta .* abs_diff) .- T(0.5) .* abs2(delta)
        return fn(ifelse.(abs_diff .≤ delta, quadratic, linear))
    end
end
