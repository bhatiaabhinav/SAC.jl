import MDPs: preexperiment, preepisode, prestep, poststep, postepisode, postexperiment

export SACLearner

mutable struct SACLearner{Tₛ<:AbstractFloat, Tₐ<:AbstractFloat} <: AbstractHook
    π::SACPolicy{Tₛ, Tₐ}
    critics
    γ::Float32
    α::Float32
    ρ::Float32
    min_explore_steps::Int
    train_interval::Int
    batch_size::Int

    s::Union{Vector{Tₛ}, Nothing}
    buff::CircularBuffer{Tuple{Vector{Tₛ}, Vector{Tₐ}, Float64, Vector{Tₛ}, Bool}}
    critics′
    optim_actor::Adam
    optim_critics::Adam

    stats::Dict{Symbol, Float32}

    function SACLearner(π::SACPolicy{Tₛ, Tₐ}, critic, γ::Real, α, η_actor, η_critic; polyak=0.995, min_explore_steps=10000, train_interval=1, batch_size=32, buffer_size=1000000) where {Tₛ <: AbstractFloat, Tₐ <: AbstractFloat}
        buff = CircularBuffer{Tuple{Vector{Tₛ}, Vector{Tₐ}, Float64, Vector{Tₛ}, Bool}}(buffer_size)
        new{Tₛ, Tₐ}(π, (critic, deepcopy(critic)), γ, α, polyak, min_explore_steps, train_interval, batch_size, nothing, buff, (deepcopy(critic), deepcopy(critic)), Adam(η_actor), Adam(η_critic), Dict{Symbol, Float32}())
    end
end

function prestep(sac::SACLearner; env::AbstractMDP, kwargs...)
    sac.s = copy(state(env))
end

function poststep(sac::SACLearner{Tₛ, Tₐ}; env::AbstractMDP{Vector{Tₛ}, Vector{Tₐ}}, steps::Int, returns, rng::AbstractRNG, kwargs...) where {Tₛ, Tₐ}
    @unpack π, critics, γ, α, ρ, batch_size, s, critics′ = sac

    a, r, s′, d = copy(action(env)), reward(env), copy(state(env)), in_absorbing_state(env)
    push!(sac.buff, (s, a, r, s′, d))

    if steps >= sac.min_explore_steps && steps % sac.train_interval == 0
        replay_batch = rand(rng, sac.buff, batch_size)
        𝐬, 𝐚, 𝐫, 𝐬′, 𝐝 = map(i -> reduce((𝐱, y) -> cat(𝐱, y; dims=ndims(y) + 1), map(experience -> experience[i], replay_batch)), 1:5)
        𝐬, 𝐚, 𝐫, 𝐬′, 𝐝 = tof32.((𝐬, 𝐚, 𝐫, 𝐬′, 𝐝))

        𝐚′, logπ𝐚′ = sample_action_logπ(π, rng, 𝐬′)
        𝐪′ = min.(map(critic -> critic(vcat(𝐬′, 𝐚′))[1, :], critics′)...)
        𝐯′ = 𝐪′ - α * logπ𝐚′
        𝐪 = 𝐫 + (1 .- 𝐝) * γ .* 𝐯′
        ϕ = Flux.params(critics...)
        ℓϕ, ∇ϕℓ = Flux.Zygote.withgradient(ϕ) do
            return sum(map(critic -> Flux.mse(critic(vcat(𝐬, 𝐚))[1, :], 𝐪), critics)) / length(critics)
        end
        Flux.update!(sac.optim_critics, ϕ, ∇ϕℓ)

        θ = Flux.params(π.actor_model)
        ℓθ, ∇θℓ = Flux.Zygote.withgradient(θ) do
            𝐚, logπ𝐚 = sample_action_logπ(π, rng, 𝐬)
            𝐪 = min.(map(critic -> critic(vcat(𝐬, 𝐚))[1, :], critics)...)
            𝐯 = 𝐪 - α * logπ𝐚
            return -mean(𝐯)
        end
        Flux.update!(sac.optim_actor, θ, ∇θℓ)

        H = -mean(sample_action_logπ(π, rng, 𝐬)[2])
        target_ent::Float32 = -1 / size(action_space(env), 1)
        α = clamp(exp(log(α) - 0.0003f0 * (H - target_ent)), 0.0001f0, 1000f0)
        sac.α = α

        ϕ′ = Flux.params(critics′)
        for (param, param′) in zip(ϕ, ϕ′); copy!(param′, ρ * param′ + (1 - ρ) * param); end

        if steps % 1000 == 0
            v̄ = -ℓθ
            sac.stats[:ℓϕ] = ℓϕ
            sac.stats[:ℓθ] = ℓθ
            sac.stats[:v̄] = v̄
            sac.stats[:H] = H
            sac.stats[:α] = α
            episodes = length(returns)
            @debug "learning stats" steps episodes ℓϕ ℓθ v̄ H α
        end
    end
    nothing
end
