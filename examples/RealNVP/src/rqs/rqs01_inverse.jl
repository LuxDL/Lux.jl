"""
Rational Quadratic Spline inverse transformation (scalar version).

This function implements the inverse transformation for a single bin of the RQS bijector.
Uses fixed-iteration Newton-Raphson method for numerical stability.

Parameters:
- s: spline value in [0,1]
- aL: left slope parameter (must be positive)
- aR: right slope parameter (must be positive)

Returns:
- t: normalized position in bin [0,1]
- dt_ds: derivative of inverse with respect to s
"""
function rqs01_inverse_scalar(s::Real, aL::Real, aR::Real)
    # Add epsilon to avoid division by zero
    ε = eps(typeof(s))

    # Ensure positive slopes for monotonicity
    aL = max(aL, ε)
    aR = max(aR, ε)

    # Clamp input to [0,1]
    s = clamp(s, 0, 1)

    # Handle linear case (delta ≈ 0)
    delta = aR - aL
    if abs(delta) < ε
        return s, one(s)
    end

    # Initial guess: linear approximation
    t = s

    # Fixed-iteration Newton-Raphson (5 iterations)
    for iter in 1:5
        # Compute current spline value and derivative
        t2 = t * t
        t3 = t2 * t

        # Numerator: t * (1 + delta * t * (1-t))
        num = t * (1 + delta * t * (1 - t))

        # Denominator: 1 + delta * t * (1-t)
        den = 1 + delta * t * (1 - t)

        # Current spline value
        s_current = num / den

        # Derivative: ds/dt
        num_prime = 1 + delta * (2 * t - 3 * t2)
        den_prime = delta * (1 - 2 * t)
        ds_dt = (num_prime * den - num * den_prime) / (den * den)

        # Newton step
        residual = s_current - s
        t_new = t - residual / ds_dt

        # Clamp to [0,1] for stability
        t_new = clamp(t_new, 0, 1)

        # Check convergence
        if abs(t_new - t) < ε
            t = t_new
            break
        end

        t = t_new
    end

    # Compute final derivative (inverse of forward derivative)
    t2 = t * t

    # Recompute numerator and denominator for final derivative
    num = t * (1 + delta * t * (1 - t))
    den = 1 + delta * t * (1 - t)

    num_prime = 1 + delta * (2 * t - 3 * t2)
    den_prime = delta * (1 - 2 * t)
    ds_dt = (num_prime * den - num * den_prime) / (den * den)

    dt_ds = 1 / ds_dt

    return t, dt_ds
end

"""
Rational Quadratic Spline inverse transformation (broadcasted version).

This function applies the RQS inverse transformation to arrays, handling broadcasting
and bin selection automatically.

Parameters:
- v: output values in [0,1] (any shape)
- x_pos: bin positions (K+1, D, B)
- y_pos: output positions (K+1, D, B)  
- d: derivatives at knots (K+1, D, B)

Returns:
- u: input values
- log_det: log determinant of the transformation
"""
function rqs01_inverse(v::AbstractArray, x_pos::AbstractArray, y_pos::AbstractArray, d::AbstractArray)
    # Ensure inputs are in [0,1]
    v = clamp.(v, 0, 1)

    # Get dimensions
    K = size(x_pos, 1) - 1
    D = size(x_pos, 2)
    B = size(x_pos, 3)

    # Handle single dimension case
    if D == 1
        # Reshape for single dimension
        v_reshaped = reshape(v, :, B)
        x_pos_reshaped = reshape(x_pos, K + 1, 1, B)
        y_pos_reshaped = reshape(y_pos, K + 1, 1, B)
        d_reshaped = reshape(d, K + 1, 1, B)
    else
        # Reshape for multiple dimensions
        v_reshaped = reshape(v, :, B)
        x_pos_reshaped = reshape(x_pos, K + 1, :, B)
        y_pos_reshaped = reshape(y_pos, K + 1, :, B)
        d_reshaped = reshape(d, K + 1, :, B)
    end

    # Initialize outputs
    u_reshaped = similar(v_reshaped)
    log_det_reshaped = similar(v_reshaped)

    # Process each dimension separately
    for dim in 1:size(v_reshaped, 1)
        for batch in 1:B
            # Extract parameters for this dimension and batch
            # For single dimension case, always use dim=1
            actual_dim = D == 1 ? 1 : dim
            x_pos_d = x_pos_reshaped[:, actual_dim, batch]
            y_pos_d = y_pos_reshaped[:, actual_dim, batch]
            d_d = d_reshaped[:, actual_dim, batch]

            # Find which bin contains this output
            v_val = v_reshaped[dim, batch]

            # Find bin index (vectorized)
            bin_mask = v_val .>= y_pos_d[2:end]
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
            if h < eps(typeof(v_val))
                s = zero(v_val)
            else
                s = (v_val - yL) / h
            end

            # Clamp s to [0,1]
            s = clamp(s, 0, 1)

            # Compute slopes
            aL = dL * w / h
            aR = dR * w / h

            # Apply inverse spline transformation
            t, dt_ds = rqs01_inverse_scalar(s, aL, aR)

            # Rescale to input space
            u_val = xL + w * t

            # Compute log determinant (inverse of forward)
            log_det_val = log(abs(w * dt_ds / h))

            # Store results
            u_reshaped[dim, batch] = u_val
            log_det_reshaped[dim, batch] = log_det_val
        end
    end

    # Reshape back to original shape
    u = reshape(u_reshaped, size(v))
    log_det = reshape(log_det_reshaped, size(v))

    return u, log_det
end

# Export the main function
export rqs01_inverse
