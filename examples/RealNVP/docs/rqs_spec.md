# RQS (Rational Quadratic Spline) Bijector Specification

## Mathematical Definition

The RQS bijector operates on the unit interval [0,1] and transforms input `u` to output `v` using piecewise rational quadratic functions.

### Forward Transformation

For input `u` in bin `k` (where `x_pos[k] ≤ u < x_pos[k+1]`):

1. **Normalization**: `t = (u - x_pos[k]) / (x_pos[k+1] - x_pos[k])`
2. **Rational quadratic**: `v = y_pos[k] + (y_pos[k+1] - y_pos[k]) * s(t)`
3. **Spline function**: `s(t) = t * (1 + (aR - aL) * t * (1-t)) / (1 + (aR - aL) * t * (1-t))`

Where:
- `aL = d[k] * (x_pos[k+1] - x_pos[k]) / (y_pos[k+1] - y_pos[k])`
- `aR = d[k+1] * (x_pos[k+1] - x_pos[k]) / (y_pos[k+1] - y_pos[k])`

### Inverse Transformation

The inverse is computed using fixed-iteration Newton-Raphson method:
1. Initial guess: `t₀ = (v - y_pos[k]) / (y_pos[k+1] - y_pos[k])`
2. Iterate: `t_{i+1} = t_i - (s(t_i) - t_target) / s'(t_i)`
3. Clamp to [0,1] and rescale: `u = x_pos[k] + t * (x_pos[k+1] - x_pos[k])`

### Log-Determinant

`log|det J| = log|∂v/∂u| = log|(y_pos[k+1] - y_pos[k]) / (x_pos[k+1] - x_pos[k]) * s'(t)|`

## Domain and Constraints

- **Domain**: Input `u ∈ [0,1]`, Output `v ∈ [0,1]`
- **Continuity**: C¹ continuous at bin boundaries
- **Monotonicity**: Strictly increasing when `aL, aR > 0`
- **Boundary**: `v(0) = 0`, `v(1) = 1`

## API Design

### Core Functions

```julia
# Forward transformation
function rqs_forward(u::AbstractArray, x_pos::AbstractArray, y_pos::AbstractArray, d::AbstractArray)
    return v, log_det
end

# Inverse transformation  
function rqs_inverse(v::AbstractArray, x_pos::AbstractArray, y_pos::AbstractArray, d::AbstractArray)
    return u, log_det
end
```

### Parameter Shapes

- `u, v`: `(D, B)` where D = dimensions, B = batch size
- `x_pos, y_pos`: `(K+1, D, B)` where K = number of bins
- `d`: `(K+1, D, B)` derivatives at knot points

### Parameterization

```julia
# Convert conditioner outputs to spline parameters
function parameterize_rqs(logits::AbstractArray)
    # logits shape: (3K+1, D, B)
    K = (size(logits, 1) - 1) ÷ 3
    
    # Width logits → positive widths → cumulative positions
    w_logits = logits[1:K, :, :]
    widths = softmax(w_logits; dims=1)
    x_pos = vcat(zeros(1, size(logits, 2), size(logits, 3)), 
                 cumsum(widths; dims=1))
    
    # Height logits → positive heights → cumulative positions  
    h_logits = logits[K+1:2K, :, :]
    heights = softmax(h_logits; dims=1)
    y_pos = vcat(zeros(1, size(logits, 2), size(logits, 3)),
                 cumsum(heights; dims=1))
    
    # Derivative logits → positive derivatives
    d_logits = logits[2K+1:end, :, :]
    d = softplus.(d_logits)
    
    return x_pos, y_pos, d
end
```

## Implementation Constraints

### Reactant Compatibility

- **No control flow**: Avoid `if`, `while`, `for` loops
- **Elementwise operations**: Use broadcasting and `map`
- **Branchless selection**: Use `ifelse` for conditional logic
- **Type stability**: Support `TrackedReal` for AD

### Memory Management

- **Gather-based bin selection**: `x .>= x_pos[2:end, :, :]` to find active bin
- **Vectorized operations**: Process all elements simultaneously
- **Minimal temporaries**: Reuse arrays where possible

### Numerical Stability

- **Epsilon guards**: Add small constants to avoid division by zero
- **Clamping**: Ensure outputs stay in [0,1]
- **Finite iterations**: Limit Newton-Raphson to 5-10 steps

## Testing Strategy

### Unit Tests
- Forward/inverse round-trip accuracy
- Derivative reciprocity: `(∂v/∂u) * (∂u/∂v) ≈ 1`
- Boundary conditions: `v(0) = 0`, `v(1) = 1`
- Monotonicity for positive slopes
- Broadcasting consistency

### Integration Tests
- Reactant tracing compatibility
- Memory usage profiling
- Performance benchmarks
- Gradient correctness with Enzyme

## Risk Mitigation

1. **Control flow issues**: Start with scalar implementation, then vectorize
2. **Memory tracking**: Test minimal examples before full integration
3. **Numerical instability**: Add comprehensive edge case testing
4. **Performance**: Profile early and optimize bottlenecks 