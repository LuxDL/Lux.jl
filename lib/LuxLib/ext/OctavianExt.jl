module OctavianExt

using Octavian: Octavian
using LuxLib.Utils: True

using LuxLib: LuxLib, Utils

Utils.is_extension_loaded(::Val{:Octavian}) = True()

@inline function LuxLib.Impl.matmul_octavian!(
    C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α::Number, β::Number
)
    Octavian.matmul!(C, A, B, α, β)
    return nothing
end

end
