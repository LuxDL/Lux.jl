# Testing using JET.jl
const JET_TARGET_MODULES = Ref{Union{Nothing, Vector{String}}}(nothing)

"""
    jet_target_modules!(list::Vector{String})

This sets `target_modules` for all JET tests when using [`@jet`](@ref).
"""
function jet_target_modules!(list::Vector{String})
    JET_TARGET_MODULES[] = list
    @info "JET_TARGET_MODULES set to $list"
    return list
end

"""
    @jet f(args...) call_broken=false opt_broken=false

Run JET tests on the function `f` with the arguments `args...`. If `JET.jl` fails to
compile, then the macro will be a no-op.

## Keyword Arguments

  - `call_broken`: Marks the test_call as broken.
  - `opt_broken`: Marks the test_opt as broken.

All additional arguments will be forwarded to `JET.@test_call` and `JET.@test_opt`.

!!! tip

    Instead of specifying `target_modules` with every call, you can set global target
    modules using [`jet_target_modules!`](@ref).

    ```julia
    using LuxTestUtils

    jet_target_modules!(["Lux", "LuxLib"]) # Expects Lux and LuxLib to be present in the module calling `@jet`
    ```

## Example

```jldoctest
julia> @jet sum([1, 2, 3]) target_modules=(Base, Core)
Test Passed

julia> @jet sum(1, 1) target_modules=(Base, Core) opt_broken=true call_broken=true
Test Broken
  Expression: #= REPL[21]:1 =# JET.@test_opt target_modules = (Base, Core) sum(1, 1)
```
"""
macro jet(expr, args...)
    !JET_TESTING_ENABLED && return :()

    all_args, call_extras, opt_extras = [], [], []
    target_modules_set = false
    for kwexpr in args
        if Meta.isexpr(kwexpr, :(=))
            if kwexpr.args[1] == :call_broken
                push!(call_extras, :(broken = $(kwexpr.args[2])))
            elseif kwexpr.args[1] == :opt_broken
                push!(opt_extras, :(broken = $(kwexpr.args[2])))
            elseif kwexpr.args[1] == :broken
                throw(ArgumentError("`broken` keyword argument is ambiguous. Use `call_broken` or `opt_broken` instead."))
            else
                kwexpr.args[1] == :target_modules && (target_modules_set = true)
                push!(all_args, kwexpr)
            end
        else
            push!(all_args, kwexpr)
        end
    end

    if !target_modules_set && JET_TARGET_MODULES[] !== nothing
        target_modules = getproperty.((__module__,), Tuple(Symbol.(JET_TARGET_MODULES[])))
        push!(all_args, :(target_modules = $target_modules))
    end

    push!(all_args, expr)

    ex_call = JET.call_test_ex(:report_call, Symbol("@test_call"),
        vcat(call_extras, all_args), __module__, __source__)
    ex_opt = JET.call_test_ex(:report_opt, Symbol("@test_opt"),
        vcat(opt_extras, all_args), __module__, __source__)

    return Expr(:block, ex_call, ex_opt)
end
