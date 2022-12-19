import MDPs: preexperiment, preepisode, prestep, poststep, postepisode, postexperiment

export SACDiscreteLearner

mutable struct SACDiscreteLearner{T<:AbstractFloat} <: AbstractHook
    π::SACDiscretePolicy{T}
    critics
    γ::Float32
    α::Float32
    ρ::Float32
    min_explore_steps::Int
    train_interval::Int
    batch_size::Int
    auto_tune_α::Bool

    s::Union{Vector{T}, Nothing}
    buff::CircularBuffer{Tuple{Vector{T}, Int, Float64, Vector{T}, Bool}}
    critics′
    optim_actor::Adam
    optim_critics::Adam

    stats::Dict{Symbol, Float32}

    function SACDiscreteLearner(π::SACDiscretePolicy{T}, critic, γ::Real; α=0.1, η_actor=0.0003, η_critic=0.0003, polyak=0.995, min_explore_steps=10000, train_interval=1, batch_size=64, buffer_size=1000000, auto_tune_α=false) where {T <: AbstractFloat}
        buff = CircularBuffer{Tuple{Vector{T}, Int, Float64, Vector{T}, Bool}}(buffer_size)
        new{T}(π, (critic, deepcopy(critic)), γ, α, polyak, min_explore_steps, train_interval, batch_size, auto_tune_α, nothing, buff, (deepcopy(critic), deepcopy(critic)), Adam(η_actor), Adam(η_critic), Dict{Symbol, Float32}())
    end
end


function prestep(sac::SACDiscreteLearner; env::AbstractMDP, kwargs...)
    sac.s = copy(state(env))
end

function poststep(sac::SACDiscreteLearner{T}; env::AbstractMDP{Vector{T}, Int}, steps::Int, returns, rng::AbstractRNG, kwargs...) where {T}
    @unpack π, critics, γ, α, ρ, batch_size, s, critics′ = sac

    a, r, s′, d = copy(action(env)), reward(env), copy(state(env)), in_absorbing_state(env)
    push!(sac.buff, (s, a, r, s′, d))

    if steps >= sac.min_explore_steps && steps % sac.train_interval == 0
        replay_batch = rand(rng, sac.buff, batch_size)
        𝐬, 𝐚, 𝐫, 𝐬′, 𝐝 = map(i -> reduce((𝐱, y) -> cat(𝐱, y; dims=ndims(y) + 1), map(experience -> experience[i], replay_batch)), 1:5)
        𝐬, 𝐫, 𝐬′, 𝐝 = tof32.((𝐬, 𝐫, 𝐬′, 𝐝))
        𝐚 = map(i -> CartesianIndex(𝐚[i], i), 1:batch_size)

        𝛑′, log𝛑′ = get_probs_logprobs(π, 𝐬′)
        𝐪̂′ = min.(map(critic -> critic(𝐬′), critics′)...)
        𝐯̂′ =  sum(𝛑′ .* (𝐪̂′ - α * log𝛑′); dims=1)[1, :]
    
        ϕ = Flux.params(critics...)
        ℓϕ, ∇ϕℓ = Flux.Zygote.withgradient(ϕ) do
            𝐪̂¹, 𝐪̂² = critics[1](𝐬), critics[2](𝐬)
            println(size.((𝐫, 𝐝′, 𝐯̂′, 𝐚, 𝐪̂¹, 𝐪̂¹[𝐚])))
            𝛅¹ = (𝐫 + γ * (1f0 .- 𝐝) .* 𝐯̂′ - 𝐪̂¹[𝐚])
            𝛅² = (𝐫 + γ * (1f0 .- 𝐝) .* 𝐯̂′ - 𝐪̂²[𝐚])
            ℓϕ = 0.5f0 * (mean(𝛅¹.^2) + mean(𝛅².^2))
            return ℓϕ 
        end
        Flux.update!(sac.optim_critics, ϕ, ∇ϕℓ)

        θ = Flux.params(π)
        ℓθ, ∇θℓ = Flux.Zygote.withgradient(θ) do
            𝛑, log𝛑 = get_probs_logprobs(π, 𝐬)
            𝐪̂ = min.(map(critic -> critic(𝐬), critics)...)
            𝐯̂ = sum(𝛑 .* (𝐪̂ - α * log𝛑); dims=1)
            return -mean(𝐯̂)
        end
        Flux.update!(sac.optim_actor, θ, ∇θℓ)

        𝛑, log𝛑 = get_probs_logprobs(π, 𝐬)
        H = -mean(sum(𝛑 .* log𝛑; dims=1))
        if sac.auto_tune_α
            target_ent::Float32 = 0.98f0 * log(length(action_space(env)))
            α = clamp(exp(log(α) - 0.0003f0 * (H - target_ent)), 0.0001f0, 1000f0)
            sac.α = α
        end
        
        ϕ′ = Flux.params(critics′...)
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
