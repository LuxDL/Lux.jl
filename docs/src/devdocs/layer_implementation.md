# Layer Implementation

## Recurrent Neural Networks

### Cell Implementations

#### Explicit Management on End-User Side

!!! note
    We currently use this implementation

User is responsible for managing the memory and hidden states.

##### Pros

1. Simple Design and Implementation.
2. Hard for the User to mess up, i.e. there is no explicit requirement to call things like
   `Flux.reset!`.
    * In the first call user passes the `input`.
    * In the subsequent calls, the user passes a tuple containing the `input`,
      `hidden_state` and `memory` (if needed).

##### Cons

1. Requires more explicit management from the user which might make it harder to use.
2. Currently the call order convention is not enforced which could lead to sneaky errors.
   (Implementing a check is quite trivial if we store a call counter in the model `state`).


#### Store Hidden State and Memory in Model State

Storing the memory and hidden state in `st` would allow user to just pass `x` without
varying how calls are made at different timesteps.

##### Pros

1. Easier for the end-user.

##### Cons

1. `reset`ing the hidden-state and memory is slightly tricky.
   1. One way would be to store a `initial_hidden_state` and `initial_memory` in the state
      alongside the `hidden_state` and `memory`.


### RNN Blocks

!!! note
    This is currently unimplemented

An example implementation would be

```julia
struct RNN{R} <: Lux.AbstractExplicitContainerLayer{(:recurrent_cell,)}
    recurrent_cell::R
end

function (l::RNN)(x::AbstractArray{T,3}, ps::NamedTuple, st::NamedTuple) where {T}
    x_init, x_rest = Iterators.peel(eachslice(x; dims=2))
    (y, carry), st = l.recurrent_cell(x_init, ps, st)
    for x in x_rest
        (y, carry), st = l.recurrent_cell((x, carry), ps, st)
    end
    return y, st
end
```

We enforce the inputs to be of the format `in_dims × sequence_length × batch_size`.
