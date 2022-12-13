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
    shift::AbstractVector{Float32}
    scale::AbstractVector{Float32}
    function SACPolicy{Tₛ, Tₐ}(actor_model, deterministic::Bool, shift::AbstractVector{Float32}, scale::AbstractVector{Float32}) where {Tₛ, Tₐ}
        new{Tₛ, Tₐ}(actor_model, deterministic, shift, scale)
    end
    function SACPolicy{Tₛ, Tₐ}(actor_model, deterministic::Bool, aspace::MDPs.VectorSpace{Tₐ}) where {Tₛ, Tₐ}
        @unpack lows, highs = aspace
        shift = (lows + highs) / 2  |> tof32
        scale = (highs - lows) / 2  |> tof32
        new{Tₛ, Tₐ}(actor_model, deterministic, shift, scale)
    end
end

Flux.@functor SACPolicy (actor_model, )
Flux.gpu(p::SACPolicy{Tₛ, Tₐ}) where {Tₛ, Tₐ}  = SACPolicy{Tₛ, Tₐ}(Flux.gpu(p.actor_model), p.deterministic, Flux.gpu(p.shift), Flux.gpu(p.scale))
Flux.cpu(p::SACPolicy{Tₛ, Tₐ}) where {Tₛ, Tₐ}  = SACPolicy{Tₛ, Tₐ}(Flux.cpu(p.actor_model), p.deterministic, Flux.cpu(p.shift), Flux.cpu(p.scale))

function (p::SACPolicy{Tₛ, Tₐ})(rng::AbstractRNG, s::Vector{Tₛ})::Vector{Tₐ} where {Tₛ, Tₐ}
    s = reshape(s, :, 1)
    a = p(rng, s)
    return convert(Vector{Tₐ}, a[:, 1])
end

function (p::SACPolicy{Tₛ, Tₐ})(s::Vector{Tₛ}, a::Vector{Tₐ})::Float64 where {Tₛ, Tₐ}  # returns log probability density
    s = reshape(s, :, 1)
    a = reshape(a, :, 1)
    return p(s, a)[1]
end





function (p::SACPolicy)(rng::AbstractRNG, s::AbstractMatrix{<:AbstractFloat})::AbstractMatrix{Float32}
    a, _ = sample_action_logπ(p, rng, tof32(s))
    return a
end

function (p::SACPolicy)(s::AbstractMatrix{<:AbstractFloat}, a::AbstractMatrix{<:AbstractFloat})::AbstractVector{Float32}  # returns log probability densities
    s = s |> tof32
    a = a |> tof32
    a_unshifted_unscaled = (a .- p.shift) ./ p.scale
    a_untanhed = atanh.(a_unshifted_unscaled)

    μ, σ, logσ = get_μ_σ_logσ(p, s)
    logπ = sum(log_nomal_prob.(a_untanhed, μ, σ, logσ), dims=1)
    logπ = logπ - sum(log.(1f0 .- a_unshifted_unscaled.^2 .+ 1f-6), dims=1)

    return logπ[1, :]
end


function get_μ_σ_logσ(p::SACPolicy, states::AbstractMatrix{<:AbstractFloat})::Tuple{AbstractMatrix{Float32}, AbstractMatrix{Float32}, AbstractMatrix{Float32}}
    _latent, _μ, _logσ = p.actor_model
    # println(p.actor_model)
    latent::AbstractMatrix{Float32} = _latent(states)
    μ::AbstractMatrix{Float32} = _μ(latent)
    logσ::AbstractMatrix{Float32} = !p.deterministic ? _logσ(latent) : Zygote.@ignore convert(typeof(μ), fill(-Inf32, size(μ)))
    logσ = clamp.(logσ, -20f0, 2f0)
    return μ, exp.(logσ), logσ
end

function sample_action_logπ(p::SACPolicy, rng::AbstractRNG, μ::AbstractMatrix{Float32}, σ::AbstractMatrix{Float32}, logσ::AbstractMatrix{Float32}; ξ::Union{Nothing, AbstractMatrix{Float32}}=nothing)::Tuple{AbstractMatrix{Float32}, AbstractVector{Float32}}
    if isnothing(ξ)
        ξ = Zygote.@ignore convert(typeof(σ), randn(rng, Float32, size(σ)))
    end
    a::AbstractMatrix{Float32} = μ + σ .* ξ  # reparametrized sampling
    logπ::AbstractMatrix{Float32} = sum(log_nomal_prob.(a, μ, σ, logσ), dims=1)  # probability of the action
    a = tanh.(a)  # squashed gaussian policy
    logπ = logπ - sum(log.(1f0 .- a.^2 .+ 1f-6), dims=1)  # probability of squashed guassian policy action
    
    a = p.scale .* a .+ p.shift

    return a, logπ[1, :]
end

function sample_action_logπ(p::SACPolicy, rng::AbstractRNG, states::AbstractMatrix{Float32}; ξ::Union{Nothing, AbstractMatrix{Float32}}=nothing)
    μ, σ, logσ = get_μ_σ_logσ(p, states)
    return sample_action_logπ(p, rng, μ, σ, logσ; ξ=ξ)
end