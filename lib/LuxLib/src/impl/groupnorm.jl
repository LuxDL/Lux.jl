# Launch Heuristics
_linear_threads_groupnorm(::CPU) = Threads.nthreads()
_linear_threads_groupnorm(::CUDADevice) = (16, 16)
_linear_threads_groupnorm(::GPU) = 256

_GROUPNORM_IMPL_FLOAT = Union{Float32, Float64}

# Low-Level Kernels
## Original Implementation: https://github.com/pytorch/pytorch/blob/master/caffe2/operators/group_norm_op.cu
@kernel function _compute_fused_params_kernel!(scale, bias, @Const(C), @Const(K),
                                               @Const(mu), @Const(rsig), @Const(gamma),
                                               @Const(beta))
    idx = @index(Global)
    ng = _div_idx(idx, K)
    c = _mod_idx(idx, C)

    @inbounds scale_val = gamma[c] * rsig[ng]
    @inbounds scale[idx] = scale_val
    @inbounds bias[idx] = beta[c] - mu[ng] * scale_val
end

@kernel function _groupnorm_forward_kernel!(Y, @Const(WxH), @Const(X), @Const(scale),
                                            @Const(bias))
    idx = @index(Global)
    nc = _div_idx(idx, WxH)
    @inbounds Y[idx] = X[idx] * scale[nc] + bias[nc]
end

@kernel function _groupnorm_dy_dscale_kernel!(dY_dscale, @Const(C), @Const(K), @Const(rsig),
                                              @Const(gamma))
    idx = @index(Global)
    ng = _div_idx(idx, K)
    c = _mod_idx(idx, C)

    @inbounds dY_dscale[idx] = gamma[c] * rsig[ng]
end

@kernel function _groupnorm_xscale_and_bias_kernel!(X_scale, bias, @Const(alpha),
                                                    @Const(mu), @Const(rsig),
                                                    @Const(ds_sum), @Const(db_sum))
    idx = @index(Global)
    @inbounds x = (db_sum[idx] * mu[idx] - ds_sum[idx]) * (rsig[idx]^3) * alpha
    @inbounds X_scale[idx] = x
    @inbounds bias[idx] = -(x * mu[idx] + db_sum[idx] * rsig[idx] * alpha)
end

@kernel function _groupnorm_dx_kernel!(dX, @Const(WxH), @Const(K), @Const(dY_dscale),
                                       @Const(dY), @Const(X_scale), @Const(X), @Const(bias))
    idx = @index(Global)
    nc = _div_idx(idx, WxH)
    ng = _div_idx(nc, K)
    @inbounds dX[idx] = dY[idx] * dY_dscale[nc] + X_scale[ng] * X[idx] + bias[ng]
end

# High-Level Function (Not User Facing)
@inbounds function _groupnorm(X::AbstractArray{T, 4}, G::Int, gamma::AbstractVector{T},
                              beta::AbstractVector{T}, epsilon::T) where {T}
    W, H, C, N = size(X)
    K = div(C, G)

    X_reshaped = reshape(X, (W, H, K, G, N))
    Y = similar(X)
    mu = mean(X_reshaped; dims=(1, 2, 3))
    rsig = 1 ./ (std(X_reshaped; mean=mu, dims=(1, 2, 3), corrected=false) .+ epsilon)

    _scale = similar(X, (C, N))
    _bias = similar(X, (C, N))

    device = get_device(X)

    n = _linear_threads_groupnorm(device)
    compute_fixed_params! = _compute_fused_params_kernel!(device, n, size(_scale))
    groupnorm_forward! = _groupnorm_forward_kernel!(device, n, size(X))

    wait(compute_fixed_params!(_scale, _bias, C, K, mu, rsig, gamma, beta;
                               ndrange=size(_scale)))
    wait(groupnorm_forward!(Y, W * H, X, _scale, _bias; ndrange=size(Y)))

    return Y, mu, rsig
end

@inbounds function _dgroupnorm(dY::AbstractArray{T, 4}, Y::AbstractArray{T, 4},
                               X::AbstractArray{T, 4}, G::Int, gamma::AbstractVector{T},
                               beta::AbstractVector{T}, mu::AbstractArray{T, 5},
                               rsig::AbstractArray{T, 5}) where {T}
    W, H, C, N = size(X)
    K = div(C, G)
    WxH = W * H
    device = get_device(X)
    n = _linear_threads_groupnorm(device)

    dbias = reshape(sum(dY; dims=(1, 2)), (1, 1, K, G, N))
    dscale = reshape(sum(X .* dY; dims=(1, 2)), (1, 1, K, G, N))

    dY_dscale = similar(X, (C, N))
    groupnorm_dy_dscale! = _groupnorm_dy_dscale_kernel!(device, n, size(dY_dscale))
    ev = groupnorm_dy_dscale!(dY_dscale, C, K, rsig, gamma; ndrange=size(dY_dscale))

    gamma_ = reshape(gamma, (1, 1, K, G, 1))
    db_sum = sum(gamma_ .* dbias; dims=3)
    ds_sum = sum(gamma_ .* dscale; dims=3)
    wait(ev)

    X_scale = similar(X, (G, N))
    bias = similar(X, (G, N))

    groupnorm_xscale_and_bias! = _groupnorm_xscale_and_bias_kernel!(device, n,
                                                                    size(X_scale))
    wait(groupnorm_xscale_and_bias!(X_scale, bias, T(1 / (K * WxH)), mu, rsig, ds_sum,
                                    db_sum; ndrange=size(X_scale)))

    dX = similar(X)
    groupnorm_dx! = _groupnorm_dx_kernel!(device, n, size(dX))
    ev = groupnorm_dx!(dX, WxH, K, dY_dscale, dY, X_scale, X, bias; ndrange=size(dX))
    dgamma = vec(sum((-dbias .* mu .+ dscale) .* rsig; dims=5))
    dbeta = vec(sum(dbias; dims=5))
    wait(ev)

    return dX, dgamma, dbeta
end
