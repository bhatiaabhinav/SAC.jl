using MDPs
using Flux
using Flux.Zygote
using Random
using UnPack
using DataStructures
using StatsBase

export SACPolicy, get_μ_σ_logσ, sample_action_logπ

struct SACPolicy{Tₛ <: AbstractFloat, Tₐ <: AbstractFloat} <: AbstractPolicy{Vector{Tₛ}, Vector{Tₐ}}
    actor_model # latent, μ, logσ
    deterministic::Bool
    action_space::MDPs.VectorSpace{Tₐ}
end

function (p::SACPolicy{Tₛ, Tₐ})(rng::AbstractRNG, s::Vector{Tₛ})::Vector{Tₐ} where {Tₛ, Tₐ}
    s = reshape(s, :, 1)
    a = p(rng, s)
    return convert(Vector{Tₐ}, a[:, 1])
end

function (p::SACPolicy{Tₛ, Tₐ})(s::Vector{Tₛ}, a::Vector{Tₐ})::Float64 where {Tₛ, Tₐ}  # returns probability density
    s = reshape(s, :, 1)
    a = reshape(a, :, 1)
    return p(s, a)[1]
end





function (p::SACPolicy)(rng::AbstractRNG, s::Matrix{<:AbstractFloat})::AbstractMatrix{Float32}
    a, _ = sample_action_logπ(p, rng, tof32(s))
    return a
end

function (p::SACPolicy)(s::Matrix{<:AbstractFloat}, a::Matrix{<:AbstractFloat})::AbstractVector{Float32}  # returns probability densities
    s = s |> tof32
    a = a |> tof32
    @unpack lows, highs = p.action_space
    shift = (lows + highs) / 2  |> tof32
    scale = (highs - lows) / 2  |> tof32
    a_unshifted_unscaled = (a .- shift) ./ scale
    a_untanhed = atanh.(a_unshifted_unscaled)

    μ, σ, logσ = get_μ_σ_logσ(p, s)
    logπ = sum(log_nomal_prob.(a_untanhed, μ, σ, logσ), dims=1)
    logπ = logπ - sum(log.(1f0 .- a_unshifted_unscaled.^2 .+ 1f-6), dims=1)

    return exp(logπ[1, :])
end


function get_μ_σ_logσ(p::SACPolicy, states::AbstractMatrix{<:AbstractFloat})::Tuple{AbstractMatrix{Float32}, AbstractMatrix{Float32}, AbstractMatrix{Float32}}
    _latent, _μ, _logσ = p.actor_model
    # println(p.actor_model)
    latent::AbstractMatrix{Float32} = _latent(states)
    μ::AbstractMatrix{Float32} = _μ(latent)
    logσ::AbstractMatrix{Float32} = !p.deterministic ? _logσ(latent) : Zygote.@ignore fill(-Inf32, size(μ))
    logσ = clamp.(logσ, -20f0, 2f0)
    return μ, exp.(logσ), logσ
end

function sample_action_logπ(p::SACPolicy, rng::AbstractRNG, μ::AbstractMatrix{Float32}, σ::AbstractMatrix{Float32}, logσ::AbstractMatrix{Float32})::Tuple{AbstractMatrix{Float32}, AbstractVector{Float32}}
    ξ::AbstractMatrix{Float32} = Zygote.@ignore randn(rng, Float32, size(σ))
    a::AbstractMatrix{Float32} = μ + σ .* ξ  # reparametrized sampling
    logπ::AbstractMatrix{Float32} = sum(log_nomal_prob.(a, μ, σ, logσ), dims=1)  # probability of the action
    a = tanh.(a)  # squashed gaussian policy
    logπ = logπ - sum(log.(1f0 .- a.^2 .+ 1f-6), dims=1)  # probability of squashed guassian policy action
    
    @unpack lows, highs = p.action_space
    shift = (lows + highs) / 2  |> tof32
    scale = (highs - lows) / 2  |> tof32
    a = scale .* a .+ shift

    return a, logπ[1, :]
end

function sample_action_logπ(p::SACPolicy, rng::AbstractRNG, states::AbstractMatrix{Float32})
    μ, σ, logσ = get_μ_σ_logσ(p, states)
    return sample_action_logπ(p, rng, μ, σ, logσ)
end