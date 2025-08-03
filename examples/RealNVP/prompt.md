I want to replace the implementation of the bijector by RQS.

It's very challenging because bijectors in the working examples are very simple. Simply lock and lock for the scale and shift. Now we are going to face a more complicated algorithm for the bijector that requires operations like searching for the bin location in which quadratic polynomial needs to be evaluated as well as the inversion of the transformation is non-trivial. The challenge comes from the fact that locks and reactants use tracing of the memory with the control flow algorithm.

The algorithm for the RQS is also non-trivial. In many known libraries it's implemented in the unrestricted form, where the transformation is applied in the range from minus infinity to infinity, while the splines are acting in the interval from minus b to b. In fact, I want to use unconstrained splines in the range from 0 to 1, that where one defines derivative in addition to... so all derivative at all points. 3k plus 1 inputs will be converted into k values for the pin widths, k values for the heights, and k plus 1 values for the derivative. The parameters of the conditioners, they are unrestricted, therefore one needs a softmax and cumulative sum to convert them to positions of the nodes. I also worry about the correctness of this implementation and the memory tracking, therefore I wrote notes for the developer to deal with that. Consider the notes and you might use some of the workflow suggested there.

## ✅ Execution Checklist (for Cursor agent)

### M‑0 — Design spec & ADR

* [ ] Create branch feature/m0-spec.
* [ ] Add docs/rqs_spec.md with:

  * [ ] Forward/inverse formulas and dimensionless slopes aL = dL*w/h, aR = dR*w/h.
  * [ ] Domain, continuity, monotonicity assumptions.
  * [ ] Boundary policy and tie‑break on x==edge.
  * [ ] Public API sketch (function signatures, shapes, dtypes).
  * [ ] K_max padding and K_active mask strategy.
* [ ] Add docs/adr_rqs_split.md capturing *algebra vs selector* split + reasons.
* [ ] Commit with message M0: spec + ADR.
* [ ] Open PR M0: spec & ADR.
* [ ] *Report*: reports/m0_spec.md (1 page) with open questions, risks, decisions.

*Operator review gate*

* [ ] Operator reviewed and approved PR M0.
  Stop if not approved; address comments.

---

### M‑1 — Algebra: forward (scalar/broadcast)

* [ ] Create branch feature/m1-algebra-forward.
* [ ] Implement src/rqs01_forward.jl (no control flow; elementwise ops only).
* [ ] Add safeguards near u≈0/1 to avoid divisions by tiny denominators.
* [ ] Tests in test/test_rqs01_forward.jl:

  * [ ] v(0)=0, v(1)=1.
  * [ ] Slopes via finite differences match aL at 0 and aR at 1 (tolerances per dtype).
  * [ ] Monotonicity for aL,aR>0 on random samples.
  * [ ] Broadcasting consistency over scalar, vector, matrix shapes.
* [ ] CI: run tests on CPU for Float32/64.
* [ ] *Report*: reports/m1_forward.md with table of errors and any stability notes.
* [ ] Open PR M1: rqs01_forward.

*Operator review gate*

* [ ] Operator approved PR M1.

---

### M‑2 — Algebra: inverse (scalar/broadcast)

* [ ] Create branch feature/m2-algebra-inverse.
* [ ] Implement src/rqs01_inverse.jl (closed‑form or fixed‑iteration root solve; still branchless).
* [ ] Tests in test/test_rqs01_inverse.jl:

  * [ ] Round‑trip: u → forward → inverse → û with max error within thresholds.
  * [ ] Derivative reciprocity: (dv/du)*(du/dv) ≈ 1.
  * [ ] Monotonicity and broadcasting parity checks.
* [ ] CI: run tests on CPU for Float32/64.
* [ ] *Report*: reports/m2_inverse.md (include any corner cases near v≈0/1).
* [ ] Open PR M2: rqs01_inverse.

*Operator review gate*

* [ ] Operator approved PR M2.

---

### M‑3 — Selector/Wrapper (gather backend)

* [ ] Create branch feature/m3-wrapper-gather.
* [ ] Implement src/rqs_wrapper_gather.jl:

  * [ ] Index computation w/o control flow:
    non‑uniform bins via vectorized sum(x .≥ edges[2:end]).
  * [ ] *Gather* per‑bin params: xL,xR,yL,yR,dL,dR.
  * [ ] Normalize: w=xR-xL, h=yR-yL, u=(x-xL)/w, aL=dL*w/h, aR=dR*w/h.
  * [ ] Call algebra forward; rescale: y=yL + h*v; derivatives/logdet as needed.
  * [ ] Implement rqs_forward(...) and rqs_inverse(...) (inverse selects in *y‑space*).
  * [ ] Support K_active ≤ K_max with padding + mask.
* [ ] Tests test/test_wrapper_gather.jl:

  * [ ] Correctness vs a simple reference CPU piecewise implementation.
  * [ ] Continuity across bin boundaries (max jump within thresholds).
  * [ ] Float32/64, shapes: N, N×D; several K_active ∈ {4,8,10,12,16}.
* [ ] Tracing check:

  * [ ] Minimal script to compile with Reactant.jl; verify no control‑flow errors.
  * [ ] Ensure device arrays are accepted traced types.
* [ ] *Report*: reports/m3_trace_mem.md with:

  * [ ] Trace success/failure, any unsupported ops.
  * [ ] Peak memory, temporary counts (summary).
* [ ] Open PR M3: gather wrapper.

*Operator review gate*

* [ ] Operator approved PR M3.

---

### M‑4 — Selector/Wrapper (one‑hot masked fallback)

* [ ] Create branch feature/m4-wrapper-masked.
* [ ] Implement src/rqs_wrapper_masked.jl:

  * [ ] Build one‑hot over bins; masked sum to obtain params.
  * [ ] Same forward/inverse interface as gather backend.
* [ ] Tests mirror M‑3; add performance comparison.
* [ ] *Report*: reports/m4_backend_compare.md with speed/memory vs gather.
* [ ] Open PR M4: masked wrapper backend.

*Operator review gate*

* [ ] Operator approved PR M4 and selects default backend per device.

---

### M‑5 — Parameterization & constraints

* [ ] Create branch feature/m5-params.
* [ ] Implement src/rqs_params.jl:

  * [ ] Width logits → positive widths via softmax; cumulative sum → edges ∈ [0,1].
  * [ ] Height logits → monotone y_knots (softmax + cumsum or monotone transform).
  * [ ] Derivative logits → positive via softplus; clamp min slope ε.
* [ ] Tests test/test_params.jl:

  * [ ] Edges and y\_knots strictly increasing; derivatives > 0.
  * [ ] Gradients are non‑zero and stable (finite differences sanity).
* [ ] *Report*: reports/m5_params.md.
* [ ] Open PR M5: parameterization.

*Operator review gate*

* [ ] Operator approved PR M5.

---

### M‑6 — AD/Enzyme gradient checks

* [ ] Create branch feature/m6-grads.
* [ ] Add test/test_grads.jl:

  * [ ] Compare Enzyme gradients vs 5‑point finite differences for:
    wrt x, edges/y_knots parametrization logits, and derivative logits.
  * [ ] Check logabsdet grads (if used in flows).
  * [ ] Boundary behavior documented; grads finite near edges.
* [ ] *Report*: reports/m6_grad_check.md with error tables F32/F64.
* [ ] Open PR M6: gradient checks.

*Operator review gate*

* [ ] Operator approved PR M6.

---

### M‑7 — Lux integration + Reactant.jl compile path

* [ ] Create branch feature/m7-lux-layer.
* [ ] Implement src/lux/RQSLayer.jl:

  * [ ] Params: logits for widths/heights/derivatives; K_max config.
  * [ ] Forward uses selected backend; inverse path exposed as method.
* [ ] Example notebook/script examples/rqs_layer_demo.jl:

  * [ ] Toy loss; single training step; verify grads reach all params.
  * [ ] Reactant.jl compile test; device residency confirmed.
* [ ] *Report*: reports/m7_integration.md with compile logs and throughput snapshot.
* [ ] Open PR M7: Lux integration.

*Operator review gate*

* [ ] Operator approved PR M7.

---

### M‑8 — Performance & memory profiling

* [ ] Create branch feature/m8-bench.
* [ ] Benchmarks bench/bench_rqs.jl:

  * [ ] Vary K_active ∈ {4,8,10,12,16}, batch sizes, F32/F64.
  * [ ] Compare gather vs masked backends on CPU/GPU.
  * [ ] Record compile time, runtime, peak memory.
* [ ] *Report*: reports/m8_perf.md with plots/tables and recommended defaults.
* [ ] Open PR M8: benchmarks.

*Operator review gate*

* [ ] Operator approved PR M8 and confirms defaults.

---

### M‑9 — Robustness & corner cases

* [ ] Create branch feature/m9-robustness.
* [ ] Property tests test/test_property_based.jl:

  * [ ] Random ill‑conditioned bins, extreme slopes, x at/near edges.
  * [ ] No NaNs/Infs; expected behavior at kinks documented.
* [ ] *Report*: reports/m9_robustness.md.
* [ ] Open PR M9: robustness tests.

*Operator review gate*

* [ ] Operator approved PR M9.

---

### M‑10 — API docs & examples

* [ ] Create branch feature/m10-docs.
* [ ] Docstrings + doctests for all public APIs.
* [ ] User guide docs/usage_rqs.md:

  * [ ] Build edges/heights/derivatives from logits.
  * [ ] Forward/inverse usage; logdet; batching & shapes.
  * [ ] Notes on Reactant.jl tracing constraints and backend selection.
* [ ] *Report*: reports/m10_docs.md.
* [ ] Open PR M10: docs & examples.

*Operator review gate*

* [ ] Operator approved PR M10 (API freeze).

---

### M‑11 — Project integration & rollout

* [ ] Create branch feature/m11-rollout.
* [ ] Replace existing spline in model training branch.
* [ ] Add CI jobs:

  * [ ] Unit tests and grad checks on CPU (push/PR).
  * [ ] (Optional) GPU job or manual workflow for Reactant.jl trace.
* [ ] Issue templates: bug_backend.md with minimal reproducer checklist.
* [ ] *Report*: reports/m11_rollout.md summarizing integration status.
* [ ] Open PR M11: rollout.

*Operator review gate*

* [ ] Operator approved PR M11.

---

## Stop conditions & escalation (agent rules)

* [ ] If Reactant.jl rejects *gather* ops → switch to *masked backend* and notify Operator in the milestone report.
* [ ] If dynamic K_active causes shape mismatches → compile with fixed K_max specializations or pad/mask; escalate in report.
* [ ] If numeric checks fail near edges → record reproducible inputs, attach plots/tables in report, and pause for Operator guidance.

---

## Reporting template (paste into each report)

* [ ] *Scope:* milestone ID and files changed.
* [ ] *Results:* key metrics (errors, timings, memory).
* [ ] *Decisions:* defaults chosen, parameters fixed.
* [ ] *Open issues:* blockers, upstream bugs, next steps.
* [ ] *Attachments:* paths to logs, figures, benchmark tables.

---

## How to run (agent quick commands)

* [ ] ] activate . && instantiate (or project’s setup script).
* [ ] julia --project -e 'using Pkg; Pkg.test()'.
* [ ] Benchmarks: julia --project bench/bench_rqs.jl > reports/bench.log.
* [ ] Reactant.jl trace check: run examples/rqs_layer_demo.jl and save logs to reports/trace_log.txt.

---

You do not have to follow all instructions there, but I want your achivements not be fo nothing. Find a way to save reached milestones, and fully completed tasks, e.g. in git main, making development in branches.


I've just coded processing of the inital parameters of the conditioner network for get the nodes of the tranformation.

using NNlib
using Plots
theme(:boxed)

# ╔═╡ 26fcd2f4-6d74-11f0-38b7-b7aed89b5f7f
function transform(pars::AbstractArray)
	_w = pars[1:2,:]
	_h = pars[3:4,:]
	_d = pars[5:end,:]
	x_pos_no0 = cumsum(softmax(_w; dims=1); dims=1)
	y_pos_no0 = cumsum(softmax(_h; dims=1); dims=1)
	d = softplus.(_d)
	x_pos = vcat(zero(x_pos_no0[1:1,:]), x_pos_no0)
	y_pos = vcat(zero(y_pos_no0[1:1,:]), y_pos_no0)
	return RQS(x_pos, y_pos, d)
end

# ╔═╡ 84df93f6-3795-4d48-b1ec-9a333ee07085
struct RQS
	x_pos
	y_pos
	d
end

# ╔═╡ e1813d26-b4b9-45dd-acc2-deffb6bacb06
const nBatch = 9

# ╔═╡ 8a93ae61-b65a-47b9-9d39-b40d7a2c8a38
myRQS = transform(randn(7,nBatch));

# ╔═╡ a8f424b4-d1a0-48f6-99b7-9841f3bdca47
plot(map(1:nBatch) do ib
	xv = myRQS.x_pos[:, ib]
	yv = myRQS.y_pos[:, ib]
	plot(xv, yv, m=(:o, 5))
end..., size=(800,800)) # looks correct


Please achnowledge that that is a very difficult task. There are many steps in it, many challanges related to memory management and implementation of the complex algorithm. Given that the project is extensive set up an infrastructure that will enable efficient development

Make sure not rush out with the final training. Running Lux+Reactant is expensive due to computational cost, make sure not to engauge with a feedback loops with long waiting time. Test processing in components.


