using FiniteDifferences, LuxLib, Test
using LuxCUDA  # CUDA Support
using ReverseDiff, Tracker, Zygote  # AD Packages

const GROUP = get(ENV, "GROUP", "All")

cpu_testing() = GROUP == "All" || GROUP == "CPU"
cuda_testing() = (GROUP == "All" || GROUP == "CUDA") && LuxCUDA.functional()
amdgpu_testing() = (GROUP == "All" || GROUP == "AMDGPU") # && LuxAMDGPU.functional()

const MODES = begin
    # Mode, Array Type, GPU?
    cpu_mode = ("CPU", Array, false)
    cuda_mode = ("CUDA", CuArray, true)

    if GROUP == "All"
        [cpu_mode, cuda_mode]
    else
        modes = []
        cpu_testing() && push!(modes, cpu_mode)
        cuda_testing() && push!(modes, cuda_mode)
        modes
    end
end

try
    using JET
catch
    @warn "JET not not precompiling. All JET tests will be skipped." maxlog=1
    global test_call(args...; kwargs...) = nothing
    global test_opt(args...; kwargs...) = nothing
end

function Base.isapprox(x, y; kwargs...)
    @warn "`isapprox` is not defined for ($(typeof(x)), $(typeof(y))). Using `==` instead."
    return x == y
end

function Base.isapprox(x::Tuple, y::Tuple; kwargs...)
    return all(isapprox.(x, y; kwargs...))
end

function Base.isapprox(nt1::NamedTuple{fields}, nt2::NamedTuple{fields};
                       kwargs...) where {fields}
    checkapprox(xy) = isapprox(xy[1], xy[2]; kwargs...)
    checkapprox(t::Tuple{Nothing, Nothing}) = true
    return all(checkapprox, zip(values(nt1), values(nt2)))
end

function Base.isapprox(t1::NTuple{N, T}, t2::NTuple{N, T}; kwargs...) where {N, T}
    checkapprox(xy) = isapprox(xy[1], xy[2]; kwargs...)
    checkapprox(t::Tuple{Nothing, Nothing}) = true
    return all(checkapprox, zip(t1, t2))
end

Base.isapprox(::Nothing, v::AbstractArray; kwargs...) = length(v) == 0
Base.isapprox(v::AbstractArray, ::Nothing; kwargs...) = length(v) == 0
Base.isapprox(v::NamedTuple, ::Nothing; kwargs...) = length(v) == 0
Base.isapprox(::Nothing, v::NamedTuple; kwargs...) = length(v) == 0
Base.isapprox(v::Tuple, ::Nothing; kwargs...) = length(v) == 0
Base.isapprox(::Nothing, v::Tuple; kwargs...) = length(v) == 0
Base.isapprox(x::AbstractArray, y::NamedTuple; kwargs...) = length(x) == 0 && length(y) == 0
Base.isapprox(x::NamedTuple, y::AbstractArray; kwargs...) = length(x) == 0 && length(y) == 0
Base.isapprox(x::AbstractArray, y::Tuple; kwargs...) = length(x) == 0 && length(y) == 0
Base.isapprox(x::Tuple, y::AbstractArray; kwargs...) = length(x) == 0 && length(y) == 0

# JET Tests
function run_JET_tests(f, args...; call_broken=false, opt_broken=false, kwargs...)
    @static if VERSION >= v"1.7"
        test_call(f, typeof.(args); broken=call_broken, target_modules=(LuxLib,))
        test_opt(f, typeof.(args); broken=opt_broken, target_modules=(LuxLib,))
    end
end

__istraining(::Val{training}) where {training} = training

# Test the gradients across AD Frameworks and FiniteDifferences
# TODO: Implement it as a macro so that we get correct line numbers for `@test` failures.
function test_gradient_correctness(f::Function, args...; gpu_testing::Bool=false,
                                   skip_fdm::Bool=false, skip_fdm_override::Bool=false,
                                   soft_fail::Bool=false, kwargs...)
    gs_ad_zygote = Zygote.gradient(f, args...)
    gs_ad_tracker = Tracker.gradient(f, args...)
    gs_ad_reversediff = gpu_testing ? nothing : ReverseDiff.gradient(f, args)

    if !skip_fdm_override
        arr_len = length.(args)
        if any(x -> x >= 25, arr_len) || sum(arr_len) >= 100
            @warn "Skipping FiniteDifferences test for large arrays: $(arr_len)."
            skip_fdm = true
        end
    end

    gs_fdm = gpu_testing || skip_fdm ? nothing :
             FiniteDifferences.grad(FiniteDifferences.central_fdm(8, 1), f, args...)
    for idx in 1:length(gs_ad_zygote)
        _c1 = isapprox(Tracker.data(gs_ad_tracker[idx]), gs_ad_zygote[idx]; kwargs...)
        if soft_fail && !_c1
            @test_broken isapprox(Tracker.data(gs_ad_tracker[idx]), gs_ad_zygote[idx];
                                  kwargs...)
        else
            @test isapprox(Tracker.data(gs_ad_tracker[idx]), gs_ad_zygote[idx]; kwargs...)
        end

        if !gpu_testing
            if !skip_fdm
                _c2 = isapprox(gs_ad_zygote[idx], gs_fdm[idx]; kwargs...)
                if soft_fail && !_c2
                    @test_broken isapprox(gs_ad_zygote[idx], gs_fdm[idx]; kwargs...)
                else
                    @test isapprox(gs_ad_zygote[idx], gs_fdm[idx]; kwargs...)
                end
            end

            _c3 = isapprox(ReverseDiff.value(gs_ad_reversediff[idx]), gs_ad_zygote[idx];
                           kwargs...)
            if soft_fail && !_c3
                @test_broken isapprox(ReverseDiff.value(gs_ad_reversediff[idx]),
                                      gs_ad_zygote[idx]; kwargs...)
            else
                @test isapprox(ReverseDiff.value(gs_ad_reversediff[idx]), gs_ad_zygote[idx];
                               kwargs...)
            end
        end
    end
    return
end
