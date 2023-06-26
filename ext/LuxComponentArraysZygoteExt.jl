module LuxComponentArraysZygoteExt

using ComponentArrays, Zygote

function Zygote.accum(x::ComponentArray, ys::ComponentArray...)
    return ComponentArray(Zygote.accum(getdata(x), getdata.(ys)...), getaxes(x))
end

end
