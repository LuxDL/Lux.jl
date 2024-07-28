# Based off of the official `@test` macro
"""
    @test_softfail expr

Evaluate `expr` and record a test result. If `expr` throws an exception, the test
result will be recorded as an error. If `expr` returns a value, and it is not a boolean,
the test result will be recorded as an error.

If the test result is false then the test will be recorded as a broken test, else it will be
recorded as a pass.
"""
macro test_softfail(ex)
    # Build the test expression
    Test.test_expr!("@test_softfail", ex)

    result = Test.get_test_result(ex, __source__)

    ex = Expr(:inert, ex)
    result = quote
        do_softfail_test($result, $ex)
    end
    return result
end

function do_softfail_test(result, orig_expr)
    if isa(result, Test.Returned)
        value = result.value
        testres = if isa(value, Bool)
            if value
                Pass(:test, orig_expr, result.data, value, result.source)
            else
                Broken(:test, orig_expr)
            end
        else
            Error(:test_nonbool, orig_expr, value, nothing, result.source)
        end
    else
        @assert isa(result, Threw)
        testres = Error(:test_throws, orig_expr, result.exception,
            result.backtrace::Vector{Any}, result.source)
    end
    Test.record(get_testset(), testres)
end
