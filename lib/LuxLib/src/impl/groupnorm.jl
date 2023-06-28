# Launch Heuristics
_linear_threads_groupnorm(::CPU) = Threads.nthreads()
_linear_threads_groupnorm(::GPU) = 256

# Low-Level Kernels
## Original Implementation: https://github.com/pytorch/pytorch/blob/master/caffe2/operators/group_norm_op.cu
@kernel function _compute_fused_params_kernel!(scale,
    bias,
    @Const(C),
    @Const(K),
    @Const(μ),
    @Const(σ⁻¹),
    @Const(γ),
    @Const(β))
    idx = @index(Global)
    ng = _div_idx(idx, K)
    c = _mod_idx(idx, C)

    @inbounds scale_val = γ[c] * σ⁻¹[ng]
    @inbounds scale[idx] = scale_val
    @inbounds bias[idx] = β[c] - μ[ng] * scale_val
end

@kernel function _groupnorm_forward_kernel!(Y,
    @Const(WxH),
    @Const(X),
    @Const(scale),
    @Const(bias))
    idx = @index(Global)
    nc = _div_idx(idx, WxH)
    @inbounds Y[idx] = X[idx] * scale[nc] + bias[nc]
end

@kernel function _groupnorm_dy_dscale_kernel!(dY_dscale,
    @Const(C),
    @Const(K),
    @Const(σ⁻¹),
    @Const(γ))
    idx = @index(Global)
    ng = _div_idx(idx, K)
    c = _mod_idx(idx, C)

    @inbounds dY_dscale[idx] = γ[c] * σ⁻¹[ng]
end

@kernel function _groupnorm_xscale_and_bias_kernel!(X_scale,
    bias,
    @Const(alpha),
    @Const(μ),
    @Const(σ⁻¹),
    @Const(ds_sum),
    @Const(db_sum))
    idx = @index(Global)
    @inbounds x = (db_sum[idx] * μ[idx] - ds_sum[idx]) * (σ⁻¹[idx]^3) * alpha
    @inbounds X_scale[idx] = x
    @inbounds bias[idx] = -(x * μ[idx] + db_sum[idx] * σ⁻¹[idx] * alpha)
end

@kernel function _groupnorm_dx_kernel!(dX,
    @Const(WxH),
    @Const(K),
    @Const(dY_dscale),
    @Const(dY),
    @Const(X_scale),
    @Const(X),
    @Const(bias))
    idx = @index(Global)
    nc = _div_idx(idx, WxH)
    ng = _div_idx(nc, K)
    @inbounds dX[idx] = dY[idx] * dY_dscale[nc] + X_scale[ng] * X[idx] + bias[ng]
end

# High-Level Function (Not User Facing)
@inbounds function _groupnorm(X::AA4D, G::Int, γ::AV, β::AV, ϵ)
    W, H, C, N = size(X)
    K = div(C, G)

    X_reshaped = reshape(X, (W, H, K, G, N))
    μ = mean(X_reshaped; dims=(1, 2, 3))
    σ⁻¹ = 1 ./ (std(X_reshaped; mean=μ, dims=(1, 2, 3), corrected=false) .+ ϵ)

    T = promote_type(eltype(μ), eltype(σ⁻¹), eltype(γ), eltype(β))
    _scale = similar(X, T, (C, N))
    _bias = similar(X, T, (C, N))
    Y = similar(X, T)

    backend = KA.get_backend(X)

    n = _linear_threads_groupnorm(backend)
    compute_fixed_params! = _compute_fused_params_kernel!(backend, n, size(_scale))
    groupnorm_forward! = _groupnorm_forward_kernel!(backend, n, size(X))

    compute_fixed_params!(_scale, _bias, C, K, μ, σ⁻¹, γ, β; ndrange=size(_scale))
    KA.synchronize(backend)

    groupnorm_forward!(Y, W * H, X, _scale, _bias; ndrange=size(Y))
    KA.synchronize(backend)

    return Y, μ, σ⁻¹
end

@inbounds function _dgroupnorm(dY::AA4D,
    Y::AA4D,
    X::AA4D,
    G::Int,
    γ::AV,
    β::AV,
    μ::AA5D,
    σ⁻¹::AA5D)
    W, H, C, N = size(X)
    K = div(C, G)
    WxH = W * H
    backend = KA.get_backend(X)
    n = _linear_threads_groupnorm(backend)

    dbias = reshape(sum(dY; dims=(1, 2)), (1, 1, K, G, N))
    dscale = reshape(sum(X .* dY; dims=(1, 2)), (1, 1, K, G, N))

    dY_dscale = similar(X, promote_type(typeof(σ⁻¹), typeof(γ)), (C, N))
    groupnorm_dy_dscale! = _groupnorm_dy_dscale_kernel!(backend, n, size(dY_dscale))
    groupnorm_dy_dscale!(dY_dscale, C, K, σ⁻¹, γ; ndrange=size(dY_dscale))

    γ_ = reshape(γ, (1, 1, K, G, 1))
    db_sum = sum(γ_ .* dbias; dims=3)
    ds_sum = sum(γ_ .* dscale; dims=3)
    KA.synchronize(backend)

    T = promote_type(eltype(μ), eltype(σ⁻¹), eltype(ds_sum), eltype(ds_bias))
    X_scale = similar(X, T, (G, N))
    bias = similar(X, T, (G, N))

    groupnorm_xscale_and_bias! = _groupnorm_xscale_and_bias_kernel!(backend,
        n,
        size(X_scale))
    groupnorm_xscale_and_bias!(X_scale,
        bias,
        T(1 / (K * WxH)),
        μ,
        σ⁻¹,
        ds_sum,
        db_sum;
        ndrange=size(X_scale))
    KA.synchronize(backend)

    dX = similar(X)
    groupnorm_dx! = _groupnorm_dx_kernel!(backend, n, size(dX))
    groupnorm_dx!(dX, WxH, K, dY_dscale, dY, X_scale, X, bias; ndrange=size(dX))
    dγ = vec(sum((-dbias .* μ .+ dscale) .* σ⁻¹; dims=5))
    dβ = vec(sum(dbias; dims=5))
    KA.synchronize(backend)

    return dX, dγ, dβ
end
