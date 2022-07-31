"""
    model_map(f::Function, l::AbstractExplicitLayer, ps, st::NamedTuple,
              name::String="model")

Map the function `f` over the model `l`, with the parameters `ps` and states `st`. This is
different from `Functors.fmap` since it zips the layers, parameters, and states and invokes
the function on all of them together.
"""
function model_map(f::Function, l::AbstractExplicitLayer, ps, st::NamedTuple,
                   name::String="model")
    l_c, l_re = Functors.functor(l)
    ps_c, ps_re = Functors.functor(ps)
    st_c, st_re = Functors.functor(st)

    length(l_c) == 0 && return f(l, ps, st, name)

    l_c_ = l_c isa Tuple ? l_c[1] : l_c
    ks = keys(l_c_)

    l_c_new, ps_c_new, st_c_new = [], [], []
    for k in ks
        l_c_new_, ps_c_new_, st_c_new_ = model_map(f, getproperty(l_c_, k),
                                                   getproperty(ps_c, k),
                                                   getproperty(st_c, k),
                                                   join((name, k), "."))
        push!(l_c_new, k => l_c_new_)
        push!(ps_c_new, k => ps_c_new_)
        push!(st_c_new, k => st_c_new_)
    end
    l_c_new = (; l_c_new...)
    l_c_new = l_c isa Tuple ? (l_c_new,) : l_c_new

    l_new = l_re(l_c_new)
    ps_new = ps_re((; ps_c_new...))
    st_new = st_re((; st_c_new...))

    return l_new, ps_new, st_new
end
