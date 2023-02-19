module LuxComponentArraysZygoteExt

if isdefined(Base, :get_extension)
    using ComponentArrays
    using Zygote
else
    using ..ComponentArrays
    using ..Zygote
end

function Zygote.accum(x::ComponentArray, ys::ComponentArray...)
    return ComponentArray(Zygote.accum(getdata(x), getdata.(ys)...), getaxes(x))
end

end
