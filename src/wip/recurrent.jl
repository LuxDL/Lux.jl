struct RNNCell{F1,F2,F3,F4} <: AbstractExplicitLayer
    λ::F1
    in_dims::Int
    out_dims::Int
    initW::F2
    initb::F3
    init_state::F4
end

function RNNCell((in, out)::Pair, λ=tanh; init=Flux.glorot_uniform, initb=zeros32, init_state=zeros32)
    return RNNCell(λ, in, out, init, initb, init_state)
end

function initialparameters(rng::AbstractRNG, r::RNNCell)
    return (
        Wi=r.initW(rng, r.out_dims, r.in_dims), Wh=r.initW(rng, r.out_dims, r.out_dims), bi=r.initb(rng, r.out_dims, 1)
    )
end

initialstates(rng::AbstractRNG, r::RNNCell) = (h=r.init_state(rng, r.out_dims, 1),)

function (r::RNNCell)(x::Union{AbstractVecOrMat,OneHotArray}, ps::NamedTuple, st::NamedTuple)
    λ = NNlib.fast_act(r.λ, x)
    h = λ.(ps.Wi * x .+ ps.Wh * st.h .+ b)
    return reshape_cell_output(h, x), (h=h,)
end
