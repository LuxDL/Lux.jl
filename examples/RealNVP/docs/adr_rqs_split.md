# ADR: RQS Implementation Split - Algebra vs Selector

## Status
Accepted

## Context
The RQS bijector implementation requires complex mathematical operations and efficient bin selection. We need to decide how to structure the code to balance correctness, performance, and Reactant compatibility.

## Decision
Split the RQS implementation into two distinct components:

1. **Algebra Layer** (`rqs01_forward.jl`, `rqs01_inverse.jl`)
   - Pure mathematical functions operating on scalar values
   - No control flow, only elementwise operations
   - Handles the core spline mathematics
   - Testable independently

2. **Selector/Wrapper Layer** (`rqs_wrapper_gather.jl`)
   - Handles bin selection and parameter gathering
   - Vectorized operations for efficiency
   - Manages broadcasting and shape handling
   - Integrates with Lux/Reactant

## Consequences

### Positive
- **Separation of concerns**: Math logic isolated from selection logic
- **Testability**: Algebra functions can be tested independently
- **Reusability**: Algebra functions can be used with different selection strategies
- **Debugging**: Easier to isolate issues in math vs selection
- **Reactant compatibility**: Algebra layer is guaranteed branchless

### Negative
- **Code complexity**: Two layers instead of one
- **Performance overhead**: Potential for redundant computations
- **Interface complexity**: Need to coordinate between layers

## Implementation Strategy

### Phase 1: Algebra Implementation
- Implement scalar forward/inverse functions
- Test with known values and edge cases
- Ensure numerical stability

### Phase 2: Wrapper Implementation  
- Implement gather-based bin selection
- Integrate with algebra functions
- Test broadcasting and shape handling

### Phase 3: Integration
- Connect to Lux conditioner network
- Test with Reactant tracing
- Optimize performance

## Alternatives Considered

1. **Monolithic implementation**: Single function handling everything
   - Rejected: Too complex to test and debug

2. **Template-based approach**: Compile-time specialization
   - Rejected: Not compatible with Reactant's dynamic compilation

3. **Masked backend only**: Use one-hot encoding for bin selection
   - Rejected: Less efficient than gather-based approach

## Metrics for Success
- Round-trip accuracy: `|u - u'| < 1e-6`
- Derivative reciprocity: `|(∂v/∂u) * (∂u/∂v) - 1| < 1e-6`
- Reactant tracing: Successful compilation without control flow errors
- Performance: Within 2x of affine bijector runtime 