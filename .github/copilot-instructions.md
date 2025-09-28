# Lux.jl Deep Learning Framework

Lux.jl is a Julia deep learning framework focused on explicit parameterization of neural networks. Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

- Bootstrap, build, and test the repository:
  - `julia --startup-file=no --project=. -e "using Pkg; Pkg.instantiate()"` -- takes 45-60 seconds. NEVER CANCEL. Set timeout to 120+ seconds.
  - `julia --startup-file=no --project=. -e "using Lux; println(\"Lux loaded successfully\")"` -- takes 3-5 seconds. Test basic functionality.
  - Use temporary environment for development dependencies: `julia --startup-file=no --project=@temp -e "using Pkg; Pkg.add(Pkg.PackageSpec(name=\"JuliaFormatter\", version=\"1\")); using JuliaFormatter; format(\".\")"`-- code formatting with JuliaFormatter v1. Required before commits.
- Run examples:
  - `cd examples/Basics && julia --startup-file=no --project=. -e "using Pkg; Pkg.instantiate()"` -- takes 50-60 seconds. NEVER CANCEL.
  - `cd examples/Basics && julia --startup-file=no --project=. --threads=auto main.jl` -- takes 25-30 seconds. Test neural network training.
- Test neural network functionality:
  - Basic model creation: `model = Chain(Dense(4, 8, tanh), Dense(8, 2))`
  - Setup parameters: `ps, st = Lux.setup(Random.default_rng(), model)`. Remember to load `Random` before using this function.
  - Forward pass: `y, st_new = Lux.apply(model, x, ps, st)`
  - Training: Use `Lux.Training.TrainState` with optimizers from `Optimisers.jl`
- Adding or deleting dependencies:
  - Whenever adding or deleting dependencies, use Pkg.jl APIs instead of directly editing Project.toml files
- Run tests:
  - Full test suite requires external package dependencies that may fail due to network restrictions
  - Use test groups: `LUX_TEST_GROUP=core_layers`, `LUX_TEST_GROUP=normalize_layers`, `LUX_TEST_GROUP=autodiff`, `LUX_TEST_GROUP=recurrent_layers`, `LUX_TEST_GROUP=misc`, `LUX_TEST_GROUP=reactant`
  - Tests use ReTestItems framework with tags

## Validation

- ALWAYS manually validate any new code via the Basics example when changing core functionality.
- ALWAYS run through at least one complete end-to-end scenario after making changes.
- You can build and run the examples successfully, which provides good validation coverage.
- Always run `julia --startup-file=no --project=@temp -e "using Pkg; Pkg.add(Pkg.PackageSpec(name=\"JuliaFormatter\", version=\"1\")); using JuliaFormatter; format(\".\")"` before you are done or the CI (.github/workflows/CI.yml) will fail.

## Development Environment Guidelines

- **NEVER** add development dependencies to the main Project.toml file
- **ALWAYS** use temporary environments for development tools: `julia --startup-file=no --project=@temp`
- **ALWAYS** use JuliaFormatter v1 for code formatting: `Pkg.add(Pkg.PackageSpec(name="JuliaFormatter", version="1"))`
- Main project dependencies should only include runtime dependencies for Lux.jl users
- Development tools like JuliaFormatter, testing utilities, and documentation tools should be in temporary environments

### Quick Validation Commands

```bash
# Validate basic loading (3-5 seconds)
julia --startup-file=no --project=. -e "using Lux; println(\"✅ Lux loads\")"

# Validate neural network functionality (30 seconds)
julia --startup-file=no --project=@temp -e "
using Pkg; Pkg.add([\"Optimisers\", \"Zygote\"]); using Lux, Random, Optimisers, Zygote
rng = Random.default_rng()
Random.seed!(rng, 0)
model = Chain(Dense(4, 8, tanh), Dense(8, 2))
ps, st = Lux.setup(rng, model) |> cpu_device()
x = rand(rng, Float32, 4, 10) |> cpu_device()
y, st_new = Lux.apply(model, x, ps, st)
println(\"✅ Forward pass: \", size(x), \" -> \", size(y))
opt = Adam(0.01f0)
train_state = Lux.Training.TrainState(model, ps, st, opt)
_, loss, _, _ = Lux.Training.single_train_step!(AutoZygote(), MSELoss(), (x, rand(rng, Float32, 2, 10)), train_state)
println(\"✅ Training step works\")
"

# Run full Basics example for comprehensive validation (25-30 seconds)
cd examples/Basics && julia --startup-file=no --project=. --threads=auto main.jl
```

## Common tasks

The following are outputs from frequently run commands. Reference them instead of viewing, searching, or running bash commands to save time.

### Repository root

```
ls -la /home/runner/work/Lux.jl/Lux.jl
.JuliaFormatter.toml
.buildkite/
.git-blame-ignore-revs
.github/
.gitignore
.typos.toml
CITATION.cff
LICENSE
Project.toml
README.md
assets/
benchmarks/
docs/
examples/
ext/
lib/
perf/
src/
test/
```

### Key project structure

- **Main package**: `Project.toml`, `src/` - Core Lux.jl functionality
- **Sub-packages**: `lib/LuxCore/`, `lib/LuxLib/`, `lib/MLDataDevices/`, `lib/WeightInitializers/`, `lib/LuxTestUtils/`, `lib/LuxCUDA/`
- **Examples**: `examples/` - Self-contained usage examples (Basics, PolynomialFitting, CIFAR10, etc.)
- **Documentation**: `docs/` - Uses Documenter.jl with DocumenterVitepress
- **Tests**: `test/` - Uses ReTestItems with tagged test groups
- **Extensions**: `ext/` - Optional package extensions for Flux, Zygote, Enzyme, etc.

### Package instantiation timing

- Main package: ~45-60 seconds (NEVER CANCEL - set 120+ second timeout)
- Examples: ~50-60 seconds each (NEVER CANCEL - set 120+ second timeout)
- Test environment

### Julia version requirements

- Requires Julia 1.10+
- Current environment: Julia 1.11.6
- Uses Package sources for sub-packages in Julia 1.11+

### Test structure

Test groups available via `LUX_TEST_GROUP` environment variable:

- `core_layers` - Dense, Conv, etc.
- `normalize_layers` - BatchNorm, LayerNorm, GroupNorm
- `autodiff` - Automatic differentiation functionality
- `recurrent_layers` - RNN, LSTM, GRU layers
- `misc` - Miscellaneous functionality
- `reactant` - Reactant.jl integration (not on Windows)

### Key commands that work

```bash
# Basic setup (NEVER CANCEL - takes 45-60 seconds)
julia --startup-file=no --project=. -e "using Pkg; Pkg.instantiate()"

# Test loading (takes 3-5 seconds)
julia --startup-file=no --project=. -e "using Lux; println(\"Success\")"

# Add commonly needed packages to temporary environment
julia --startup-file=no --project=@temp -e "using Pkg; Pkg.add([Pkg.PackageSpec(name=\"JuliaFormatter\", version=\"1\"), \"Optimisers\", \"Zygote\"])"

# Format code (required before commits - use JuliaFormatter v1)
julia --startup-file=no --project=@temp -e "using Pkg; Pkg.add(Pkg.PackageSpec(name=\"JuliaFormatter\", version=\"1\")); using JuliaFormatter; format(\".\")"

# Run basic example (works reliably)
cd examples/Basics
julia --startup-file=no --project=. -e "using Pkg; Pkg.instantiate()"  # 50-60 seconds
julia --startup-file=no --project=. --threads=auto main.jl              # 25-30 seconds

# Quick validation of neural network functionality (takes 7 seconds)
julia --startup-file=no --project=@temp --threads=auto -e "
using Pkg; Pkg.add([\"Optimisers\", \"Zygote\"]); using Lux, Random, Optimisers, Zygote
rng = Random.default_rng()
Random.seed!(rng, 0)
model = Chain(Dense(4, 8, tanh), Dense(8, 2))
ps, st = Lux.setup(rng, model) |> cpu_device()
x = rand(rng, Float32, 4, 10) |> cpu_device()
y, st_new = Lux.apply(model, x, ps, st)
println(\"✅ Neural network works: \", size(x), \" -> \", size(y))
"

# Check repo structure
ls -la src/
ls -la lib/
ls -la examples/
```

### Known limitations

- **Test suite**: Full test suite cannot run due to external dependency issues
- **Large examples**: Some examples (like PolynomialFitting) have heavy dependencies that may timeout
- **Documentation building**: Requires many dependencies and may fail in restricted environments

### Development workflow

1. Make changes to source code in `src/` or relevant sub-package in `lib/`
2. Test with basic example: `cd examples/Basics && julia --startup-file=no --project=. --threads=auto main.jl`
3. Format code: `julia --startup-file=no --project=@temp -e "using Pkg; Pkg.add(Pkg.PackageSpec(name=\"JuliaFormatter\", version=\"1\")); using JuliaFormatter; format(\".\")`
4. Validate specific functionality by modifying and running appropriate example
5. Check that basic loading still works: `julia --startup-file=no --project=. -e "using Lux"`
6. For all commit messages use conventional commits
7. Once development is completed, ensure that the versions in Project.toml for each of the relevant packages are updated in accordance with semantic versioning

### CI/CD information

- Uses GitHub Actions with `.github/workflows/CI.yml`
- Tests run across multiple Julia versions (1.10, 1)
- Tests are parallelized by test groups
- Uses CommonCI.yml for shared workflow logic
- Requires code formatting with JuliaFormatter v1
- Has extensive downstream testing for dependent packages

### Example descriptions

- **Basics**: Introduction to Lux.jl with arrays, layers, and training (reliable)
- **PolynomialFitting**: Polynomial regression example (heavy dependencies)
- **CIFAR10**: Image classification example (heavy dependencies)
- **NeuralODE**: Neural ordinary differential equations (complex)
- **SimpleRNN**: Recurrent neural network example (moderate)

Always validate your changes by running the Basics example to ensure core functionality remains intact.
