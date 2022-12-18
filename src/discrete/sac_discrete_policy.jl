using MDPs
using StatsBase
using Flux

export SACDiscretePolicy

struct SACDiscretePolicy{T<:AbstractFloat} <: AbstractPolicy{Vector{T}, Int}
    actor_model  # maps states to action log probabilities
    deterministic::Bool
end

Flux.@functor SACDiscretePolicy (actor_model, )
Flux.gpu(p::SACDiscretePolicy{T}) where {T}  = SACDiscretePolicy{T}(Flux.gpu(p.actor_model))
Flux.cpu(p::SACDiscretePolicy{T}) where {T}  = SACDiscretePolicy{T}(Flux.cpu(p.actor_model))


function (p::SACDiscretePolicy{T})(rng::AbstractRNG, s::Vector{T})::Int where {T}
    𝐬 = reshape(s, :, 1)
    𝐚 = p(rng, 𝐬)
    return 𝐚[1]
    # argmax(p.actor_model(tof32(s)))
end

function (p::SACDiscretePolicy{T})(s::Vector{T}, a::Int) where {T}
    𝐬 = reshape(s, :, 1)
    return p(𝐬, :)[a, 1]
end



function (p::SACDiscretePolicy{T})(rng::AbstractRNG, 𝐬::AbstractMatrix{<:AbstractFloat})::Vector{Int} where {T}
    𝐬 = tof32(𝐬)
    probabilities = p(𝐬, :)
    n, batch_size = size(probabilities)
    π = zeros(Int, n)
    for i in 1:batch_size
        π[i] = sample(rng, 1:n, ProbabilityWeights(probabilities[:, i]))
    end
    return π
end

function (p::SACDiscretePolicy{T})(𝐬::AbstractMatrix{<:AbstractFloat}, 𝐚::AbstractVector{Int})::AbstractVector{Float32} where {T}
    probabilities = p(𝐬, :)
    batch_size = length(𝐚)
    action_indices = [CartesianIndex(𝐚[i], i) for i in 1:batch_size]
    return probabilities[action_indices]
end


function (p::SACDiscretePolicy{T})(𝐬::AbstractMatrix{<:AbstractFloat}, ::Colon)::AbstractMatrix{Float32} where {T}
    𝐬 = tof32(𝐬)
    logits = p.actor_model(𝐬)
    if p.deterministic
        logits = logits * 1.0f6
    end
    probabilities = Flux.softmax(logits)
    return probabilities
end

function get_probs_logprobs(p::SACDiscretePolicy{T}, 𝐬::AbstractMatrix{<:AbstractFloat})::Tuple{AbstractMatrix{Float32}, AbstractMatrix{Float32}} where {T}
    𝐬 = tof32(𝐬)
    logits = p.actor_model(𝐬)
    if p.deterministic
        logits = logits * 1.0f6
    end
    probabilities = Flux.softmax(logits)
    logprobabilities = Flux.logsoftmax(logits)
    return probabilities, logprobabilities
end


