"""
RQS Wrapper Layer with Gather-based Bin Selection

This module provides the main interface for RQS bijector operations, handling
parameterization, bin selection, and integration with Lux/Reactant.

The wrapper layer:
1. Converts conditioner outputs to spline parameters
2. Performs gather-based bin selection
3. Calls the algebra layer for forward/inverse transformations
4. Handles broadcasting and shape management
"""

using NNlib

# Include the algebra layer
include("rqs01_forward.jl")
include("rqs01_inverse.jl")

"""
Convert conditioner outputs to RQS spline parameters.

Parameters:
- logits: conditioner output of shape (3K+1, D, B)
  - logits[1:K, :, :]: width logits
  - logits[K+1:2K, :, :]: height logits  
  - logits[2K+1:end, :, :]: derivative logits

Returns:
- x_pos: bin positions (K+1, D, B)
- y_pos: output positions (K+1, D, B)
- d: derivatives at knots (K+1, D, B)
"""
function parameterize_rqs(logits::AbstractArray)
    # Get dimensions
    K = (size(logits, 1) - 1) ÷ 3
    D = size(logits, 2)
    B = size(logits, 3)

    # Extract logits for each parameter type
    w_logits = logits[1:K, :, :]
    h_logits = logits[K+1:2K, :, :]
    d_logits = logits[2K+1:end, :, :]

    # Convert to positive parameters with numerical stability
    widths = softmax(w_logits; dims = 1)
    heights = softmax(h_logits; dims = 1)
    derivatives = softplus.(d_logits)

    # Add small epsilon to prevent zero derivatives
    ε = eps(Float32)
    derivatives = max.(derivatives, ε)

    # Convert to cumulative positions
    x_pos = vcat(zeros(1, D, B), cumsum(widths; dims = 1))
    y_pos = vcat(zeros(1, D, B), cumsum(heights; dims = 1))

    return x_pos, y_pos, derivatives
end

"""
RQS Forward transformation with parameterization.

This is the main interface function for forward RQS transformation.

Parameters:
- u: input values in [0,1] (any shape)
- logits: conditioner outputs (3K+1, D, B)

Returns:
- v: transformed values
- log_det: log determinant of the transformation
"""
function rqs_forward(u::AbstractArray, logits::AbstractArray)
    # Parameterize the conditioner outputs
    x_pos, y_pos, d = parameterize_rqs(logits)

    # Call the algebra layer
    return rqs01_forward(u, x_pos, y_pos, d)
end

"""
RQS Inverse transformation with parameterization.

This is the main interface function for inverse RQS transformation.

Parameters:
- v: output values in [0,1] (any shape)
- logits: conditioner outputs (3K+1, D, B)

Returns:
- u: input values
- log_det: log determinant of the transformation
"""
function rqs_inverse(v::AbstractArray, logits::AbstractArray)
    # Parameterize the conditioner outputs
    x_pos, y_pos, d = parameterize_rqs(logits)

    # Call the algebra layer
    return rqs01_inverse(v, x_pos, y_pos, d)
end

"""
RQS Bijector for Lux integration.

This struct provides a clean interface for integrating RQS with Lux models.
"""
struct RQSBijector
    K::Int  # Number of bins
end

"""
Forward transformation for RQSBijector.

Parameters:
- bj: RQSBijector instance
- u: input values
- logits: conditioner outputs

Returns:
- v: transformed values
- log_det: log determinant
"""
function forward_and_log_det(bj::RQSBijector, u::AbstractArray, logits::AbstractArray)
    return rqs_forward(u, logits)
end

"""
Inverse transformation for RQSBijector.

Parameters:
- bj: RQSBijector instance
- v: output values
- logits: conditioner outputs

Returns:
- u: input values
- log_det: log determinant
"""
function inverse_and_log_det(bj::RQSBijector, v::AbstractArray, logits::AbstractArray)
    return rqs_inverse(v, logits)
end

# Export main functions
export RQSBijector, rqs_forward, rqs_inverse, parameterize_rqs, forward_and_log_det, inverse_and_log_det
