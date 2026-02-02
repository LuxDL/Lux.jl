# Most of the tests in this file were derived from https://github.com/FluxML/NNlib.jl/blob/master/test/batchedmul.jl
using Test, NNlib, StableRNGs, LuxLib

include("../../shared_testsetup.jl")

function bmm_test(a, b; transA=false, transB=false)
    bs = size(a, 3)
    transA && (a = permutedims(a, [2, 1, 3]))
    transB && (b = permutedims(b, [2, 1, 3]))
    c = []
    for i in 1:bs
        push!(c, a[:, :, i] * b[:, :, i])
    end
    return cat(c...; dims=3)
end

function bmm_adjtest(a, b; adjA=false, adjB=false)
    bs = size(a, 3)
    c = []
    for i in 1:bs
        ai = adjA ? adjoint(a[:, :, i]) : a[:, :, i]
        bi = adjB ? adjoint(b[:, :, i]) : b[:, :, i]
        push!(c, ai * bi)
    end
    return cat(c...; dims=3)
end

function half_batched_mul(x, y)
    @assert size(y, 3) == 1
    d = size(x, 2)
    x_mat = reshape(permutedims(x, (1, 3, 2)), :, d)
    y_mat = reshape(y, d, :)
    z_mat = x_mat * y_mat
    return permutedims(reshape(z_mat, size(x, 1), size(x, 3), :), (1, 3, 2))
end

perm_12(A) = PermutedDimsArray(A, (2, 1, 3))
perm_23(A) = PermutedDimsArray(A, (1, 3, 2))
