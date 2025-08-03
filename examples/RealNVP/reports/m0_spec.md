# M0 Report: RQS Specification & ADR

## Scope
- Created `docs/rqs_spec.md` with mathematical definitions
- Created `docs/adr_rqs_split.md` with architecture decisions
- Established implementation strategy and constraints

## Results
- **Mathematical foundation**: Complete forward/inverse formulas defined
- **API design**: Clear function signatures and parameter shapes
- **Implementation strategy**: Algebra vs selector split decided
- **Reactant constraints**: Identified key compatibility requirements

## Decisions
- **Split architecture**: Algebra layer separate from selector layer
- **Gather-based selection**: Vectorized bin finding with `x .>= x_pos[2:end]`
- **Parameterization**: Softmax + cumsum for positions, softplus for derivatives
- **Numerical stability**: Epsilon guards and clamping strategies

## Open Issues
1. **Newton-Raphson convergence**: Need to determine optimal iteration count
2. **Edge case handling**: Behavior when `aL ≈ aR` (linear case)
3. **Memory efficiency**: Balance between gather vs masked backends
4. **Performance targets**: Define acceptable overhead vs affine bijector

## Next Steps
- **M1**: Implement algebra layer (forward transformation)
- **M2**: Implement algebra layer (inverse transformation)  
- **M3**: Implement wrapper layer with bin selection
- **M4**: Reactant tracing validation

## Risks
- **Control flow detection**: Reactant may reject complex operations
- **Numerical instability**: Edge cases near bin boundaries
- **Performance**: Vectorized operations may be slower than expected
- **Memory usage**: Gather operations may require significant temporary storage

## Attachments
- `docs/rqs_spec.md`: Complete mathematical specification
- `docs/adr_rqs_split.md`: Architecture decision record 