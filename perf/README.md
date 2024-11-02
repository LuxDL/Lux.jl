# Performance Benchmarks

## ResNet

Benchmark was run on a single NVIDIA RTX 4050 GPU with 6GB of memory.

### ResNet18 (Forward Pass)

| Batch Size | Best Timing (Flax) | Best Timing (Lux + Reactant) | Best Timing (Lux) |
| ---------- | ------------------ | ---------------------------- | ----------------- |
| 1          | 0.00249 s          | 0.0003738 s                  | 0.002161114 s     |
| 4          | 0.00381 s          | 0.000595607 s                | 0.003498441 s     |
| 32         | 0.01796 s          | 0.000510855 s                | 0.027250628 s     |
| 128        | 0.06757 s          | 0.000581028 s                | 0.115965297 s     |

### ResNet 34 (Forward Pass)

| Batch Size | Best Timing (Flax) | Best Timing (Lux + Reactant) | Best Timing (Lux) |
| ---------- | ------------------ | ---------------------------- | ----------------- |
| 1          | 0.00462 s          | 0.000799139 s                | 0.003684532 s     |
| 4          | 0.00696 s          | 0.001039750 s                | 0.006234771 s     |
| 32         | 0.03169 s          | 0.000530302 s                | 0.046339233 s     |
| 128        | 0.12129 s          | 0.001188182 s                | 0.640747518 s     |

### ResNet50 (Forward Pass)

| Batch Size | Best Timing (Flax) | Best Timing (Lux + Reactant) | Best Timing (Lux) |
| ---------- | ------------------ | ---------------------------- | ----------------- |
| 1          | 0.00403 s          | 0.000575870 s                | 0.004382536 s     |
| 4          | 0.00788 s          | 0.000712372 s                | 0.011562075 s     |
| 32         | 0.05146 s          | 0.000810471 s                | 0.103826668 s     |
| 128        | 0.20071 s          | 0.009914158 s                | 0.430018518 s     |
