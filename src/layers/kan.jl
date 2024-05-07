# utilities for splines
@views function __extend_grid(grid::AbstractMatrix; k::Integer=0)
    k == 0 && return grid
    h = (grid[end:end, :] .- grid[1:1, :]) ./ (size(grid, 1) - 1)
    for _ in 1:k
        grid = vcat(grid[1:1, :] .- h, grid, grid[end:end, :] .+ h)
    end
    return grid
end

function __bspline_evaluate(
        x::AbstractMatrix, grid::AbstractMatrix; order::Integer=3, extend::Bool=true)
    grid = extend ? __extend_grid(grid; k=order) : grid

    return
end

function __coefficient_to_curve end

function __curve_to_coefficient end

# TODO: GPUs typically ship faster routines for batched least squares.
function __batched_least_squares end
