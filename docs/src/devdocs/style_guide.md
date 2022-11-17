# Style Guide

We strictly enforce a style guide across the repository. For the most part we rely on
[SciMLStyle](https://github.com/SciML/SciMLStyle). However, any additional guideline
mentioned in this document takes precedence.

**How to auto-format your code?**

Firstly, install `JuliaFormatter` by running
`julia -e 'using Pkg; Pkg.add(PackageSpec(name="JuliaFormatter"))'`. Next, from the root
directory of the project, simply run `julia -e 'using JuliaFormatter; format(".")'`.

We do have automatic formatter, which opens PR after fixing common style issues, however, we
**strictly** don't merge PRs without a green style check.

!!! note
    If you find any existing code which doesn't adhere to these guidelines, open an issue
    so that we can fix that.

## Code Styling

* Keyword Arguments must be separated using a semicolon `;`
* Functions must use `return`. Returning the last value is quite ambiguous -- did the author
  actually want it returned?
* Format docstrings as you would format regular code. If the docstring constains LaTeX in
  multiple lines, use `math` block.
* No avoiding multiply symbol -- so `2x` is invalid instead do it like other languages
  `2 * x`.

## Testing

!!! note
    Unfortunately we haven't yet tested all the functionality in the base library using
    these guidelines.

* The file structure of the `test` folder should mirror that of the `src` folder. Every file
  in src should have a complementary file in the test folder, containing tests relevant to
  that file's contents.

* Add generic utilities for testing in `test/test_utils.jl` and include them in the relevant
  files.

* Use [JET.jl](https://aviatesk.github.io/JET.jl/dev/) to test for dynamic dispatch in the
  functionality you added, specifically use `run_JET_tests` from `test/test_utils.jl`.

* **Always** test for gradient correctness. Zygote can be notorious for incorrect gradients,
  so add tests using `test_gradient_correctness_fdm` for finite differencing or use any
  other AD framework and tally the results.


## Try adding to backend packages

Lux is mostly a frontend for defining Neural Networks. As such, if an optimization needs to
be applied to lets say `NNlib.jl`, it is better to open a PR there since all frameworks
using `NNlib.jl` get to benefit from these fixes.

Similarly, if a bug comes to the forefront from one of the backend packages, make sure to 
open a corresponding issue there to ensure they are appropriately tracked.


## Mutability

This is strictly enforced, i.e. all layers/functions provided as part of the external API
must be pure functions, even if they come with a performance penalty.

## Branching -- Generated Functions

Zygote doesn't like branches in code. Like it or not, we are stuck with it for the near
future. Even if julia is able to optimize branches away, Zygote will most certainly throw
away those optimizations (these can be tested via `Zygote.@code_ir`).

### Writing efficient non-branching code to make Zygote happy

* Rely on `@generated` functions to remove **most** runtime branching. Certain examples:
  * Layers behaving differently during training and inference -- we know at compile-time
    whether a layer is being run in training/inference mode via `istraining(st)`.
  * Composite Layers relying on a variable number of internal layers -- Again we know the
    length of the number of internal layers at compile time. Hence we can manually unroll
    the loops. See [`Parallel`](@ref), [`Chain`](@ref), etc.
* Pass around `Val` in state. `Flux.jl` sets `training` to be `(:auto, true, false)`. Hence,
  which branch will be evaluated, will have to be determined at runtime time (*bad*).
  Instead if we pass `Val(true)`, we will be able to specialize functions directly based on
  `true`, `false`, etc. ensuring there is no runtime cost for these operations.
  See [`BatchNorm`](@ref), [`Dropout`](@ref), etc.

## Deprecation

Deprecations should be handled according to SemVer recommendations, i.e. there should be
atleast one version where we throw a deprecation warning. This ensures users know how to
modify their code for upcoming releases.

[This blog](https://invenia.github.io/blog/2022/06/17/deprecating-in-julia/) details the
process of deprecating functionalities in Julia packages. We follow the same process. Some
additional guidelines are:

  - Add tests using `Test.@test_deprecated` to ensure that deprecations are indeed working
    as expected.
  - Add a warning to the documentation about deprecations (and how to use the new
    recommended functionality).
  - Add `# Deprecated Functionality (Remove in <VERSION NUMBER>)` before the tests and
    deprecated functionality not placed in `src/deprecated.jl` (like kwarg deprecations).
    This makes it easier to search and delete the functionalities before making a breaking
    release.

## Documentation

We use `Documenter.jl` + `mkdocs` for our documentation.

### Adding Tutorials

Add tutorials must be added to the `examples` directory. Then add an entry for the path and
tutorial name in `docs/make.jl`. Finally, update the navigation `nav` in `docs/mkdocs.yml`

### Documentation for Layers

The first line must be indented by 4 spaces and should contain the possible ways to
construct the layer. This should be followed up with a description about what the layer
does. If mathematical equations are needed to explain what the layer does, go for it. Often
times we fuse parameters to make computation faster, this should be reflected in the
equations being used, i.e. equations and the internal code must be consistent.
(See [`LSTMCell`](@ref), [`GRUCell`](@ref) for some examples.)

!!! note
    There is no need to document how the layers are being called since they **must** adhere
    to `layer(x, ps, st)`. Any deviation from that and the PR will not be accepted.

Next, we will have certain subsections (though all of them might not be necessary for all
layers).

* **Arguments**: This section should be present unless the layer is constructed without any
  arguments (See [`NoOpLayer`](@ref)). All the arguments and their explicit constraints must
  be explained.
  * It is recommended to separate out the Keyword Arguments in their own section.
* **Inputs**: This section should always be present. List out the requirements `x` needs to
  satisfy. (Don't write about `ps` and `st` since that is expected by default.)
* **Returns**: What will the layer return? We know the second element will be a state but is
  that updated in any form or not? 
* **Parameters**: What are the properties of the NamedTuple returned from
  `initialparameters`? Omit if the layer is parameterless.
* **States**: What are the properties of the NamedTuple returned from `initialstates`? Omit
  if the layer is stateless.
