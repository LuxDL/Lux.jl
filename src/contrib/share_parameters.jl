"""
    share_parameters(ps, sharing)
    share_parameters(ps, sharing, new_parameters)

Updates the parameters in `ps` with a common set of parameters `new_parameters` that are
shared between each list in the nested list `sharing`. (That was kind of a mouthful, the
example should make it clear).

## Arguments

  - `ps`: Original parameters.
  - `sharing`: A nested list of lists of accessors of `ps` which need to shate the
    parameters (See the example for details). (Each list in the list must be disjoint)
  - `new_parameters`: If passed the length of `new_parameters` must be equal to the
    length of `sharing`. For each vector in `sharing` the corresponding parameter in
    `new_parameters` will be used. (If not passed, the parameters corresponding to the
    first element of each vector in `sharing` will be used).

## Returns

Updated Parameters having the same structure as `ps`.

## Example

```jldoctest
julia> model = Chain(; d1=Dense(2 => 4, tanh),
           d3=Chain(; l1=Dense(4 => 2), l2=Dense(2 => 4)), d2=Dense(4 => 2))
Chain(
    d1 = Dense(2 => 4, tanh),           # 12 parameters
    d3 = Chain(
        l1 = Dense(4 => 2),             # 10 parameters
        l2 = Dense(2 => 4),             # 12 parameters
    ),
    d2 = Dense(4 => 2),                 # 10 parameters
)         # Total: 44 parameters,
          #        plus 0 states.

julia> ps, st = Lux.setup(Xoshiro(0), model);

julia> # share parameters of (d1 and d3.l1) and (d3.l2 and d2)
       ps = Lux.Experimental.share_parameters(ps, (("d3.l2", "d1"), ("d2", "d3.l1")));

julia> ps.d3.l2.weight === ps.d1.weight &&
           ps.d3.l2.bias === ps.d1.bias &&
           ps.d2.weight === ps.d3.l1.weight &&
           ps.d2.bias === ps.d3.l1.bias
true
```

!!! danger "ComponentArrays"

    ComponentArrays doesn't allow sharing parameters. Converting the returned parameters
    to a ComponentArray will silently cause the parameter sharing to be undone.
"""
function share_parameters(ps, sharing)
    assert_disjoint_sharing_list(sharing)
    lens = map(construct_property_lens, sharing)
    ps_extracted = get.((ps,), first.(lens))
    return unsafe_share_parameters(ps, ps_extracted, lens)
end

function share_parameters(ps, sharing, new_ps)
    assert_disjoint_sharing_list(sharing)
    @argcheck length(sharing)==length(new_ps) "The length of sharing and new_ps must be equal"
    return unsafe_share_parameters(ps, new_ps, map(construct_property_lens, sharing))
end

function unsafe_share_parameters(ps, new_ps, lens)
    for (new_ps, lens_group) in zip(new_ps, lens), cur_lens in lens_group

        ps = _safe_update_parameter(ps, cur_lens, new_ps)
    end
    return ps
end

function _safe_update_parameter(ps, lens, new_ps)
    new_ps_st = Utils.structure(new_ps)
    ps_st = Utils.structure(get(ps, lens))
    if new_ps_st != ps_st
        msg = lazy"The structure of the new parameters must be the same as the old parameters for lens $(lens)!!! The new parameters have a structure: $(new_ps_st) while the old parameters have a structure: $(ps_st). This could potentially be caused since `Lux.Utils.structure` is not appropriately defined for type $(typeof(new_ps))."
        throw(ArgumentError(msg))
    end
    return Setfield.set(ps, lens, new_ps)
end

function assert_disjoint_sharing_list(sharing)
    for i in eachindex(sharing), j in (i+1):length(sharing)

        if !isdisjoint(sharing[i], sharing[j])
            throw(AssertionError("sharing[$i] ($(sharing[i])) and sharing[$j] \
                                  ($(sharing[j])) must be disjoint"))
        end
    end
end

construct_property_lens(x) = construct_property_lens.(x)
function construct_property_lens(x::String)
    return foldr(
        Setfield.ComposedLens, map(x -> Setfield.PropertyLens{Symbol(x)}(), split(x, ".")))
end
