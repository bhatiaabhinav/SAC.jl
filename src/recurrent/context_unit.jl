using Flux

export GRUContextRNN

"""Last layer should be GRU layer"""
mutable struct GRUContextRNN
    model # last layer should be a GRU unit
    prev_state
end
GRUContextRNN(model) = GRUContextRNN(model, model.layers[end].state)

Flux.@functor GRUContextRNN (model, )

function Flux.reset!(crnn::GRUContextRNN)
    Flux.reset!(crnn.model)
    nothing
end

@inline get_rnn_state(crnn::GRUContextRNN) = crnn.model.layers[end].state

function set_rnn_state!(crnn::GRUContextRNN, hs::AbstractMatrix{Float32})
    if size(hs) == size(get_rnn_state(crnn))
        copy!(crnn.model.layers[end].state, hs)
    else
        crnn.model.layers[end].state = copy(hs)
    end
    nothing
end

function (crnn::GRUContextRNN)(x::AbstractMatrix{Float32})
    crnn.prev_state = get_rnn_state(crnn)
    crnn.model(x)
end

function undo!(crnn::GRUContextRNN)
    @assert crnn.prev_state !== nothing "cannot undo anymore"
    set_rnn_state!(crnn, crnn.prev_state)
    crnn.prev_state = nothing
    return get_rnn_state(crnn)
end