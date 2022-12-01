# SubPackages

`Lux.jl` operates somewhat like a monorepo having a weird inter-dependency structure. So
adding new subpackages in `lib` can be somewhat complicated. Here are the guidelines that
need to be followed:

## Package Structure

  * Each subpackage should be in its own directory in `lib`.
  * Don't have a `docs` directory (see the [Documentation Section](#documentation) for
    details).
  * Add a `LICENSE` file (needed to register the package independently).

## Workflows

  * All workflows should go in the `.github/workflows` directory (in the project root).
  * For `CI.yml` and `CINightly.yml`
    - add the project name to `group` matrix.
    - Under `directories` for `julia-actions/julia-processcoverage@v1` add
      `lib/<project name>/src`.
  * For `CompatHelper.yml` add `lib/<project name>` to the list of `subdirs`.

## Documentation

  * Create a directory for the package: `docs/src/lib/<project name>`.
  * Optionally, if you want the `index.md` page for your subpackage to be same as the
    `README.md` file, add the package name to `_setup_subdir_pkgs_index_file.([...])` in
    `docs/make.jl`.
  * For generating documentation (say from docstrings) for your package, you need to update
    the documentation pipeline:
    - In `.github/workflows/Documentation.yml` under `Install documentation dependencies`
      install the package using
      `Pkg.develop(PackageSpec(path=joinpath(pwd(), "lib/<project name>")))`.
    - Add a dependency in `docs/Project.toml` (don't add compat entries).
    - Add `using <Project Name>` in `docs/make.jl`.
    - Add the package name to `modules` in `docs/make.jl`.
  * For every new page that you have added (including `index.md` if using `README.md` file)
    update `docs/mkdocs.yml` `nav` field.

## Testing

  * For testing, always use `Project.toml` in the `lib/<project name>/test` directory.
  * Write the tests as you would for a normal package.
  * In `test/runtests.jl` add the package name to `groups` if `GROUP == "All"`.
  * Next list any cross-dependency. When CI is run, it uses the local version of the package
    instead of the registered version.
    - For example, `Lux` depends on `LuxLib` so `"Lux" => [_get_lib_path("LuxLib")]`
    - `Boltz` depends on both `Lux` and `LuxLib` (via `Lux`) so
      `"Boltz" => [_get_lib_path("LuxLib"), dirname(@__DIR__)]`
    - If there are no cross-dependencies, remember to add an empty vector.

## Registration

Registration is simply, just run `@JuliaRegistrator register subdir="lib/<project name>"`

## Code Coverage

Add the following to `.codecov.yml`:

```yaml
    - name: <package name>
      paths: 
      - lib/<package name>/**
```

If you have performed all the steps correctly the code coverage for the subpackage will
be available under the flag `<project name>`.
