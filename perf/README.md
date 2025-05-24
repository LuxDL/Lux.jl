# Performance Benchmarks

## ResNet

Benchmark was run on a single NVIDIA RTX 3060 Ti GPU with 12GB of memory.

### ResNet18 (Forward Pass)

| Batch Size | Best Timing (Lux + Reactant) | Best Timing (Flax) | Best Timing (Lux) | Speedup (Lux vs Lux + Reactant) | Speedup (Flax vs Lux + Reactant) |
| ---------- | ---------------------------- | ------------------ | ----------------- | ------------------------------- | -------------------------------- |
| 1          | 0.00125634 s                 | 0.002330780 s      | 0.00196218 s      | 1.56x                           | 1.86x                            |
| 4          | 0.00295247 s                 | 0.003750324 s      | 0.0042437 s       | 1.44x                           | 1.27x                            |
| 32         | 0.0134202 s                  | 0.015712499 s      | 0.0253424 s       | 1.89x                           | 1.17x                            |
| 128        | 0.047055 s                   | 0.05045008 s       | 0.097789 s        | 2.08x                           | 1.07x                            |

### ResNet 34 (Forward Pass)

| Batch Size | Best Timing (Lux + Reactant) | Best Timing (Flax) | Best Timing (Lux) | Speedup (Lux vs Lux + Reactant) | Speedup (Flax vs Lux + Reactant) |
| ---------- | ---------------------------- | ------------------ | ----------------- | ------------------------------- | -------------------------------- |
| 1          | 0.00239355 s                 | 0.004539012 s      | 0.00364651 s      | 1.52x                           | 1.90x                            |
| 4          | 0.00548421 s                 | 0.00700831 s       | 0.00725198 s      | 1.32x                           | 1.28x                            |
| 32         | 0.0241452 s                  | 0.028824806 s      | 0.042178 s        | 1.75x                           | 1.19x                            |
| 128        | 0.0845379 s                  | 0.08936405 s       | 0.159878 s        | 1.89x                           | 1.06x                            |

### ResNet50 (Forward Pass)

| Batch Size | Best Timing (Lux + Reactant) | Best Timing (Flax) | Best Timing (Lux) | Speedup (Lux vs Lux + Reactant) | Speedup (Flax vs Lux + Reactant) |
| ---------- | ---------------------------- | ------------------ | ----------------- | ------------------------------- | -------------------------------- |
| 1          | 0.00265636 s                 | 0.004221916 s      | 0.00532997 s      | 2.01x                           | 1.59x                            |
| 4          | 0.00690339 s                 | 0.008177042 s      | 0.0163181 s       | 2.36x                           | 1.18x                            |
| 32         | 0.0388969 s                  | 0.04150199 s       | 0.111284 s        | 2.86x                           | 1.07x                            |
| 128        | 0.141286 s                   | 0.14333295 s       | 0.436758 s        | 3.09x                           | 1.01x                            |
