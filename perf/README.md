# Performance Benchmarks

## ResNet

### ResNet50 (Forward Pass)

Benchmark was run on a single NVIDIA RTX 4050 GPU with 6GB of memory.

| Batch Size | Best Timing (Flax) | Best Timing (Lux + Reactant) | Best Timing (Lux) |
| ---------- | ------------------ | ---------------------------- | ----------------- |
| 1          | 0.00403 s          | 0.000575870 s                | 0.004382536 s     |
| 4          | 0.00788 s          | 0.000712372 s                | 0.011562075 s     |
| 32         | 0.05146 s          | 0.000810471 s                | 0.103826668 s     |
| 128        | 0.20071 s          | 0.009914158 s                | 0.430018518 s     |
