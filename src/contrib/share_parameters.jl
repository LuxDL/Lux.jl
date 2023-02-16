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

```julia
model = Chain(; d1=Dense(2 => 4, tanh), d3=Chain(; l1=Dense(4 => 2), l2=Dense(2 => 4)),
              d2=Dense(4 => 2))

ps, st = Lux.setup(Xoshiro(0), model)

# share parameters of (d1 and d3.l1) and (d3.l2 and d2)
ps = Lux.share_parameters(ps, (("d3.l2", "d1"), ("d2", "d3.l1")))
```
"""
function share_parameters(ps, sharing)
    _assert_disjoint_sharing_list(sharing)
    lens = map(x -> _construct_lens.(x), sharing)
    ps_extracted = get.((ps,), first.(lens))
    return _share_parameters(ps, ps_extracted, lens)
end

function share_parameters(ps, sharing, new_parameters)
    _assert_disjoint_sharing_list(sharing)
    if !(length(sharing) == length(new_parameters))
        throw(ArgumentError("The length of sharing and new_parameters must be equal"))
    end
    return _share_parameters(ps, new_parameters, map(x -> _construct_lens.(x), sharing))
end

function _share_parameters(ps, new_parameters, lens)
    for (new_ps, lens_group) in zip(new_parameters, lens), cur_lens in lens_group
        ps = _safe_update_parameter(ps, cur_lens, new_ps)
    end
    return ps
end

function _safe_update_parameter(ps, lens, new_ps)
    new_ps_st = _parameter_structure(new_ps)
    ps_st = _parameter_structure(get(ps, lens))
    if new_ps_st != ps_st
        msg = "The structure of the new parameters must be the same as the " *
              "old parameters for lens $(lens)!!! The new parameters have a structure: " *
              "$new_ps_st while the old parameters have a structure: $ps_st."
        msg = msg *
              " This could potentially be caused since `_parameter_structure` is not" *
              " appropriately defined for type $(typeof(new_ps))."
        throw(ArgumentError(msg))
    end
    return Setfield.set(ps, lens, new_ps)
end

_parameter_structure(ps::AbstractArray) = size(ps)
_parameter_structure(::Number) = 1
_parameter_structure(ps) = fmap(_parameter_structure, ps)

function _assert_disjoint_sharing_list(sharing)
    for i in 1:length(sharing), j in (i + 1):length(sharing)
        if !isdisjoint(sharing[i], sharing[j])
            throw(ArgumentError("sharing[$i] ($(sharing[i])) and sharing[$j] " *
                                "($(sharing[j])) must be disjoint"))
        end
    end
end

function _construct_lens(x::String)
    return foldr(Setfield.ComposedLens,
                 map(x -> Setfield.PropertyLens{Symbol(x)}(), split(x, ".")))
end
