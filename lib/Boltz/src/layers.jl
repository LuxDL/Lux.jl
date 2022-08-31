function layer_norm(sz::Integer, activation=identity; affine::Bool=true,
                    epsilon::Real=1.0f-5)
  if affine
    return Chain(WrappedFunction(x -> normalise(x, identity; dims=1, epsilon=epsilon)),
                 Scale(sz, activation))
  else
    return WrappedFunction(x -> normalise(x, activation; dims=1, epsilon=epsilon))
  end
end
