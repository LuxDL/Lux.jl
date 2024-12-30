abstract type AbstractPoolMode end

(m::AbstractPoolMode)(x) = calculate_pool_dims(m, x)

function calculate_pool_dims end

CRC.@non_differentiable calculate_pool_dims(::Any...)
EnzymeRules.inactive(::typeof(calculate_pool_dims), ::Any...) = true

@concrete struct GenericPoolMode <: AbstractPoolMode
    kernel_size <: Tuple{Vararg{IntegerType}}
    stride <: Tuple{Vararg{IntegerType}}
    pad <: Tuple{Vararg{IntegerType}}
    dilation <: Tuple{Vararg{IntegerType}}
end

function calculate_pool_dims(m::GenericPoolMode, x)
    return PoolDims(x, m.kernel_size; padding=m.pad, m.stride, m.dilation)
end

struct GlobalPoolMode <: AbstractPoolMode end

calculate_pool_dims(::GlobalPoolMode, x) = PoolDims(x, size(x)[1:(end - 2)])

@concrete struct AdaptivePoolMode <: AbstractPoolMode
    out_size <: Tuple{Vararg{IntegerType}}
end

function calculate_pool_dims(m::AdaptivePoolMode, x)
    in_size = size(x)[1:(end - 2)]
    stride = in_size .÷ m.out_size
    kernel_size = in_size .- (m.out_size .- 1) .* stride
    return PoolDims(x, kernel_size; padding=0, stride, dilation=1)
end

symbol_to_pool_mode(::StaticSymbol{:generic}) = GenericPoolMode
symbol_to_pool_mode(::StaticSymbol{:global}) = GlobalPoolMode
symbol_to_pool_mode(::StaticSymbol{:adaptive}) = AdaptivePoolMode

abstract type AbstractPoolOp end

struct MaxPoolOp <: AbstractPoolOp end

(m::MaxPoolOp)(x, pdims) = maxpool(x, pdims)
function (m::MaxPoolOp)(x, ::GlobalPoolMode)
    return maximum(x; dims=1:(ndims(x) - 2), init=eltype(x)(-Inf))
end

struct MeanPoolOp <: AbstractPoolOp end

(m::MeanPoolOp)(x, pdims) = meanpool(x, pdims)
(m::MeanPoolOp)(x, ::GlobalPoolMode) = mean(x; dims=1:(ndims(x) - 2))

@concrete struct LpPoolOp <: AbstractPoolOp
    p
end

(m::LpPoolOp)(x, pdims) = lpnormpool(x, pdims; m.p)
(m::LpPoolOp)(x, ::GlobalPoolMode) = lpnormpool(x, PoolDims(x, size(x)[1:(end - 2)]); m.p)

symbol_to_pool_op(::StaticSymbol{:max}, _) = MaxPoolOp()
symbol_to_pool_op(::StaticSymbol{:mean}, _) = MeanPoolOp()
symbol_to_pool_op(::StaticSymbol{:lp}, p) = LpPoolOp(p)

@concrete struct PoolingLayer <: AbstractLuxLayer
    mode <: AbstractPoolMode
    op <: AbstractPoolOp
end

function PoolingLayer(mode::SymbolType, op::SymbolType,
        arg::Union{Nothing, Tuple{Vararg{IntegerType}}}=nothing;
        stride=arg, pad=0, dilation=1, p=2)
    return PoolingLayer(symbol_to_pool_mode(static(mode)),
        symbol_to_pool_op(static(op), p), arg; stride, pad, dilation)
end

function PoolingLayer(::Type{GenericPoolMode}, op::AbstractPoolOp,
        kernel_size::Tuple{Vararg{IntegerType}}; stride=kernel_size, pad=0, dilation=1)
    stride = Utils.expand(Val(length(kernel_size)), stride)
    pad = calc_padding(pad, kernel_size, dilation, stride)
    dilation = Utils.expand(Val(length(kernel_size)), dilation)
    @argcheck allequal(length, (stride, kernel_size, dilation))

    return PoolingLayer(GenericPoolMode(kernel_size, stride, pad, dilation), op)
end

function PoolingLayer(::Type{AdaptivePoolMode}, op::AbstractPoolOp,
        out_size::Tuple{Vararg{IntegerType}}; kwargs...)
    return PoolingLayer(AdaptivePoolMode(out_size), op)
end

function PoolingLayer(::Type{GlobalPoolMode}, op::AbstractPoolOp, ::Nothing; kwargs...)
    return PoolingLayer(GlobalPoolMode(), op)
end

(m::PoolingLayer)(x, _, st::NamedTuple) = m.op(x, m.mode(x)), st

for layer_op in (:Max, :Mean, :LP)
    op = Symbol(lowercase(string(layer_op)))

    no_gpu_danger = layer_op == :LP ? """

    !!! danger "GPU Support"

        This layer is currently only supported on CPU.
    """ : ""

    layer_name = Symbol(layer_op, :Pool)
    extra_kwargs = layer_op == :LP ? ", p=2" : ""
    layer_docstring = """
        $(layer_name)(window; stride=window, pad=0, dilation=1$(extra_kwargs))

    $(layer_op) Pooling layer, which replaces all pixels in a block of size `window` with
    the reduction operation: $(op).

    ## Arguments

      - `window`: Tuple of integers specifying the size of the window. Eg, for 2D pooling
        `length(window) == 2`

    ## Keyword Arguments

      - `stride`: Should each be either single integer, or a tuple with `N` integers
      - `dilation`: Should each be either single integer, or a tuple with `N` integers

      - `pad`: Specifies the number of elements added to the borders of the data array. It can
        be

          + a single integer for equal padding all around,
          + a tuple of `N` integers, to apply the same padding at begin/end of each spatial
            dimension,
          + a tuple of `2*N` integers, for asymmetric padding, or
          + the singleton `SamePad()`, to calculate padding such that
            `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial
            dimension.

    $(no_gpu_danger)

    # Extended Help

    ## Inputs

      - `x`: Data satisfying `ndims(x) == N + 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

    ## Returns

      - Output of the pooling `y` of size `(O_N, ..., O_1, C, N)` where

    ```math
        O_i = \\left\\lfloor\\frac{I_i + p_i + p_{(i + N) \\% |p|} - d_i \\times (k_i - 1)}{s_i} + 1\\right\\rfloor
    ```

      - Empty `NamedTuple()`
    """

    global_layer_name = Symbol(:Global, layer_name)
    extra_kwargs = layer_op == :LP ? "; p=2" : ""
    global_pooling_docstring = """
        $(global_layer_name)($(extra_kwargs))

    Global $(layer_op) Pooling layer. Transforms `(w, h, c, b)`-shaped input into
    `(1, 1, c, b)`-shaped output, by performing mean pooling on the complete `(w, h)`-shaped
    feature maps.

    $(no_gpu_danger)

    ## Inputs

      - `x`: Data satisfying `ndims(x) > 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`

    ## Returns

      - Output of the pooling `y` of size `(1, ..., 1, C, N)`
      - Empty `NamedTuple()`
    """

    adaptive_layer_name = Symbol(:Adaptive, layer_name)
    adaptive_pooling_docstring = """
        $(adaptive_layer_name)(output_size$(extra_kwargs))

    Adaptive $(layer_op) Pooling layer. Calculates the necessary window size such that
    its output has `size(y)[1:N] == output_size`.

    ## Arguments

      - `output_size`: Size of the first `N` dimensions for the output

    $(no_gpu_danger)

    ## Inputs

      - `x`: Expects as input an array with `ndims(x) == N + 2`, i.e. channel and batch
        dimensions, after the `N` feature dimensions, where `N = length(output_size)`.

    ## Returns

      - Output of size `(out..., C, N)`
      - Empty `NamedTuple()`
    """

    @eval begin
        # Generic Pooling Layer
        @doc $(layer_docstring) @concrete struct $(layer_name) <:
                                                 AbstractLuxWrapperLayer{:layer}
            layer <: PoolingLayer
        end

        Experimental.layer_map_leaf(::KeyPath, ::$(layer_name)) = true

        function $(layer_name)(
                window::Tuple{Vararg{IntegerType}}; stride=window, pad=0, dilation=1, p=2)
            return $(layer_name)(PoolingLayer(static(:generic), static($(Meta.quot(op))),
                window; stride, pad, dilation, p))
        end

        function Base.show(io::IO, m::$(layer_name))
            (; mode, op) = m.layer
            (; kernel_size, pad, stride, dilation) = mode
            print(io, string($(Meta.quot(layer_name))), "($(kernel_size)")
            all(==(0), pad) || print(io, ", pad=", PrettyPrinting.tuple_string(pad))
            stride == kernel_size ||
                print(io, ", stride=", PrettyPrinting.tuple_string(stride))
            all(==(1), dilation) ||
                print(io, ", dilation=", PrettyPrinting.tuple_string(dilation))
            $(Meta.quot(op)) == :lp && (op.p == 2 || print(io, ", p=", op.p))
            print(io, ")")
        end

        PrettyPrinting.isa_printable_leaf(::$(layer_name)) = true

        # Global Pooling Layer
        @doc $(global_pooling_docstring) @concrete struct $(global_layer_name) <:
                                                          AbstractLuxWrapperLayer{:layer}
            layer <: PoolingLayer
        end

        Experimental.layer_map_leaf(::KeyPath, ::$(global_layer_name)) = true

        function $(global_layer_name)(; p=2)
            return $(global_layer_name)(PoolingLayer(static(:global), $(Meta.quot(op)); p))
        end

        function Base.show(io::IO, g::$(global_layer_name))
            (; op) = g.layer
            print(io, string($(Meta.quot(global_layer_name))), "(")
            $(Meta.quot(op)) == :lp && (op.p == 2 || print(io, ", p=", op.p))
            print(io, ")")
        end

        PrettyPrinting.isa_printable_leaf(::$(global_layer_name)) = true

        # Adaptive Pooling Layer
        @doc $(adaptive_pooling_docstring) @concrete struct $(adaptive_layer_name) <:
                                                            AbstractLuxWrapperLayer{:layer}
            layer <: PoolingLayer
        end

        Experimental.layer_map_leaf(::KeyPath, ::$(adaptive_layer_name)) = true

        function $(adaptive_layer_name)(out_size::Tuple{Vararg{IntegerType}}; p=2)
            return $(adaptive_layer_name)(PoolingLayer(
                static(:adaptive), $(Meta.quot(op)), out_size; p))
        end

        function Base.show(io::IO, a::$(adaptive_layer_name))
            (; mode, op) = a.layer
            print(io, string($(Meta.quot(adaptive_layer_name))), "(", mode.out_size)
            $(Meta.quot(op)) == :lp && (op.p == 2 || print(io, ", p=", op.p))
            print(io, ")")
        end

        PrettyPrinting.isa_printable_leaf(::$(adaptive_layer_name)) = true
    end
end
