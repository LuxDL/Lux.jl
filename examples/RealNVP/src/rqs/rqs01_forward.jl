"""
Rational Quadratic Spline forward transformation (scalar version).

This function implements the forward transformation for a single bin of the RQS bijector.
All operations are elementwise to ensure Reactant compatibility.

Parameters:
- t: normalized position in bin [0,1]
- aL: left slope parameter (must be positive)
- aR: right slope parameter (must be positive)

Returns:
- s: spline value in [0,1]
- ds_dt: derivative of spline with respect to t
"""
function rqs01_forward_scalar(t::Real, aL::Real, aR::Real)
    # Add epsilon to avoid division by zero
    ε = eps(typeof(t))

    # Ensure positive slopes for monotonicity
    aL = max(aL, ε)
    aR = max(aR, ε)

    # Compute spline parameters
    delta = aR - aL

    # Handle linear case (delta ≈ 0) to avoid numerical issues
    if abs(delta) < ε
        return t, one(t)
    end

    # Compute rational quadratic spline
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t

    # Numerator: t * (1 + delta * t * (1-t))
    num = t * (1 + delta * t * (1 - t))

    # Denominator: 1 + delta * t * (1-t)
    den = 1 + delta * t * (1 - t)

    # Spline value
    s = num / den

    # Derivative: ds/dt
    # Using quotient rule: (num' * den - num * den') / den^2
    num_prime = 1 + delta * (2 * t - 3 * t2)
    den_prime = delta * (1 - 2 * t)

    ds_dt = (num_prime * den - num * den_prime) / (den * den)

    # Clamp to ensure output is in [0,1]
    s = clamp(s, zero(t), one(t))

    return s, ds_dt
end

"""
Rational Quadratic Spline forward transformation (broadcasted version).

This function applies the RQS transformation to arrays, handling broadcasting
and bin selection automatically.

Parameters:
- u: input values in [0,1] (any shape)
- x_pos: bin positions (K+1, D, B)
- y_pos: output positions (K+1, D, B)  
- d: derivatives at knots (K+1, D, B)

Returns:
- v: transformed values
- log_det: log determinant of the transformation
"""
function rqs01_forward(u::AbstractArray, x_pos::AbstractArray, y_pos::AbstractArray, d::AbstractArray)
    # Ensure inputs are in [0,1]
    u = clamp.(u, 0, 1)

    # Get dimensions
    K = size(x_pos, 1) - 1
    D = size(x_pos, 2)
    B = size(x_pos, 3)

    # Handle single dimension case
    if D == 1
        # Reshape for single dimension
        u_reshaped = reshape(u, :, B)
        x_pos_reshaped = reshape(x_pos, K + 1, 1, B)
        y_pos_reshaped = reshape(y_pos, K + 1, 1, B)
        d_reshaped = reshape(d, K + 1, 1, B)
    else
        # Reshape for multiple dimensions
        u_reshaped = reshape(u, :, B)
        x_pos_reshaped = reshape(x_pos, K + 1, :, B)
        y_pos_reshaped = reshape(y_pos, K + 1, :, B)
        d_reshaped = reshape(d, K + 1, :, B)
    end

    # Initialize outputs
    v_reshaped = similar(u_reshaped)
    log_det_reshaped = similar(u_reshaped)

    # Process each dimension separately
    for dim in 1:size(u_reshaped, 1)
        for batch in 1:B
            # Extract parameters for this dimension and batch
            # For single dimension case, always use dim=1
            actual_dim = D == 1 ? 1 : dim
            x_pos_d = x_pos_reshaped[:, actual_dim, batch]
            y_pos_d = y_pos_reshaped[:, actual_dim, batch]
            d_d = d_reshaped[:, actual_dim, batch]

            # Find which bin contains this input
            u_val = u_reshaped[dim, batch]

            # Find bin index (vectorized)
            bin_mask = u_val .>= x_pos_d[2:end]
            bin_idx = sum(bin_mask) + 1
            bin_idx = clamp(bin_idx, 1, K)

            # Extract bin parameters
            xL = x_pos_d[bin_idx]
            xR = x_pos_d[bin_idx+1]
            yL = y_pos_d[bin_idx]
            yR = y_pos_d[bin_idx+1]
            dL = d_d[bin_idx]
            dR = d_d[bin_idx+1]

            # Compute normalized position in bin
            w = xR - xL
            h = yR - yL

            # Avoid division by zero
            if w < eps(typeof(u_val))
                t = zero(u_val)
            else
                t = (u_val - xL) / w
            end

            # Clamp t to [0,1]
            t = clamp(t, 0, 1)

            # Compute slopes
            aL = dL * w / h
            aR = dR * w / h

            # Apply spline transformation
            s, ds_dt = rqs01_forward_scalar(t, aL, aR)

            # Rescale to output space
            v_val = yL + h * s

            # Compute log determinant
            log_det_val = log(abs(h * ds_dt / w))

            # Store results
            v_reshaped[dim, batch] = v_val
            log_det_reshaped[dim, batch] = log_det_val
        end
    end

    # Reshape back to original shape
    v = reshape(v_reshaped, size(u))
    log_det = reshape(log_det_reshaped, size(u))

    return v, log_det
end

# Export the main function
export rqs01_forward
