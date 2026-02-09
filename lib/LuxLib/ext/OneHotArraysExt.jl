module OneHotArraysExt

using LuxLib: Utils
using OneHotArrays: OneHotLike

Utils.force_3arg_mul!_dispatch(::AbstractMatrix, ::AbstractMatrix, ::OneHotLike) = true

end
