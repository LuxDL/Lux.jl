for fnType in (typeof(sum), Any, typeof(mean))
    @eval begin
        function LossFunctionImpl.fused_agg(
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

        function LossFunctionImpl.fused_agg(
                fn::$(fnType), ::LossFunctions.L2HingeLoss, x::TracedRArray{T1, N},
                y::TracedRArray{T2, N}) where {T1, T2, N}
            T = promote_type(T1, T2)
            agreement = x .* y
            return fn(ifelse.(agreement .≥ ones(T), zero(T), abs2.(one(T) .- agreement)))
        end
    end
end
