function _normalization(x, running_mean, running_var, scale, bias, reduce_dims, training,
                        momentum, epsilon)
    Base.depwarn("""`LuxLib._normalization` with `reduce_dims` of type
                    $(typeof(reduce_dims)) has been deprecated and will be removed in v0.2.
                    Pass `reduce_dims` as `Val(Tuple(reduce_dims))`""", :_normalization)
    return _normalization(x, running_mean, running_var, scale, bias,
                          Val(Tuple(reduce_dims)), training, momentum, epsilon)
end
