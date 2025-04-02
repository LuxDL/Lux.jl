Utils.vec(x::AnyTracedRArray) = ReactantCore.materialize_traced_array(vec(x))

# XXX: Use PoolDims once EnzymeJAX supports stablehlo.reduce_window adjoint
Lux.calculate_pool_dims(g::Lux.GlobalPoolMode, ::TracedRArray) = g

# Optimisers setup
function Lux.ReactantCompatibleOptimisers.optimisers_setup_with_jit(
    opt::Lux.ReactantCompatibleOptimisers.ReactantOptimiser, ps
)
    @show opt.opt.eta.data

    st_opt = @jit Optimisers.setup(opt, ps)
    # The above will cause the numbers to be unaliased. But this causes too many issues
    # with broadcast_in_dims in the IR. So we realias the numbers here and let reactant
    # do its magic.
    @show objectid(opt.opt.eta)
    @show st_opt.b.rule.opt.eta.data
    @show objectid(st_opt.b.rule.opt.eta)
    @show st_opt.a.rule.opt.eta.data
    @show objectid(st_opt.a.rule.opt.eta)

    st_opt = Functors.fmap_with_path(
        st_opt;
        exclude=(kp, x) ->
            x isa Lux.ReactantCompatibleOptimisers.ReactantOptimiser &&
                !(x.opt isa Optimisers.OptimiserChain),
    ) do kp, x
        rule_idx = findfirst(Base.Fix2(===, :rule), kp.keys)
        @assert rule_idx !== nothing
        opt_part = Functors.getkeypath(opt, kp[(rule_idx + 1):end])
        for k in fieldnames(typeof(opt_part.opt))
            @show getfield(opt_part.opt, k)
            @show objectid(getfield(opt_part.opt, k))
            @show objectid(getfield(x.opt, k))
            zzz = Setfield.set(
                x.opt, Setfield.PropertyLens{k}(), getfield(opt_part.opt, k)
            )
            @show objectid(getfield(zzz, k))
            @set! x.opt = Setfield.set(
                x.opt, Setfield.PropertyLens{k}(), getfield(opt_part.opt, k)
            )
            @show objectid(getfield(x.opt, k))
            # if k == :opt
            #     continue
            # end
            # @assert haskey(x, k)
        end
        return x
        # nt = NamedTuple(
        #     k => getfield(opt_part.opt, k) for k in fieldnames(typeof(opt_part.opt))
        # )
        # if haskey(nt, :eta)
        #     @show objectid(nt.eta)
        #     @show objectid(x.opt.eta)
        # end
        # y = Optimisers.adjust(x; nt...)
        # if haskey(nt, :eta)
        #     @show objectid(y.opt.eta)
        # end
        # return Optimisers.adjust(x; nt...)
    end
    return st_opt
end
