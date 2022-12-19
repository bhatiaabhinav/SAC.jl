import MDPs: preexperiment, preepisode, prestep, poststep, postepisode, postexperiment
using Random
export RecurrentSACDiscreteLearner

mutable struct RecurrentSACDiscreteLearner{T<:AbstractFloat} <: AbstractHook
    π::ContextualSACDiscretePolicy{T}
    critics
    critic_crnn
    γ::Float32
    α::Float32
    ρ::Float32
    min_explore_steps::Int
    batch_size::Int
    horizon::Int
    tbptt_horizon::Int
    auto_tune_α::Bool
    device

    buff::AbstractArray{Float32, 2}  # sequence of evidence
    buff_head::Int
    traj_start_points::Set{Int}
    minibatch                               # preallocated memory for sampling a minibatch. Tuple 𝐞, 𝐨, 𝐚, 𝐫, 𝐨′, 𝐝′, 𝐧′
    𝐜ᵃ::AbstractArray{Float32, 3}           # preallocated memory for recording actor's context during a rollout
    𝐜ᶜ::AbstractArray{Float32, 3}           # preallocated memory for recording actor's context during a rollout

    actor::SACDiscretePolicy{T}     # train this sac actor and periodically copy weights to the original actor contextual policy.
    actor_crnn::GRUContextRNN               # train this actor crnn and periodically copy weights to the original actor context rnn
    critics′                                # target critic
    optim_actor::Adam
    optim_critics::Adam

    stats::Dict{Symbol, Float32}

    function RecurrentSACDiscreteLearner(π::ContextualSACDiscretePolicy{T}, critic, critic_context_rnn, γ::Real, horizon::Int, aspace::MDPs.IntegerSpace, sspace; α=0.1, η_actor=0.0003, η_critic=0.0003, polyak=0.995, batch_size=32, min_explore_steps=horizon*batch_size, tbptt_horizon=horizon, buffer_size=10000000, buff_mem_MB_cap=Inf, auto_tune_α=false, device=Flux.cpu) where {T <: AbstractFloat}
        each_entry_size = 1 + length(aspace) + 1 + size(sspace, 1) + 1
        buffer_size = min(buffer_size, buff_mem_MB_cap * 2^20 / (4 * each_entry_size)) |> floor |> Int
        buff = zeros(Float32, each_entry_size, buffer_size)
        𝐞 = zeros(Float32, each_entry_size, horizon + 1, batch_size) |> device
        𝐨 = zeros(Float32, size(sspace, 1), horizon, batch_size) |> device
        𝐚 = zeros(Float32, size(aspace, 1), horizon, batch_size) |> device
        𝐫 = zeros(Float32, horizon, batch_size) |> device
        𝐨′ = zeros(Float32, size(sspace, 1), horizon, batch_size) |> device
        𝐝′ = zeros(Float32, horizon, batch_size) |> device
        𝐧′ = zeros(Float32, horizon, batch_size) |> device
        minibatch = (𝐞, 𝐨, 𝐚, 𝐫, 𝐨′, 𝐝′, 𝐧′)
        𝐜ᵃ = zeros(Float32, size(get_rnn_state(π.crnn), 1), horizon + 1, batch_size) |> device
        𝐜ᶜ = zeros(Float32, size(get_rnn_state(critic_context_rnn), 1), horizon + 1, batch_size) |> device
        new{T}(π, (device(deepcopy(critic)), device(deepcopy(critic))), device(deepcopy(critic_context_rnn)), γ, α, polyak, min_explore_steps, batch_size, horizon, tbptt_horizon, auto_tune_α, device, buff, 1, Set{Int}(), minibatch, 𝐜ᵃ, 𝐜ᶜ, device(deepcopy(π.π)), device(deepcopy(π.crnn)), (device(deepcopy(critic)), device(deepcopy(critic))), Adam(η_actor), Adam(η_critic), Dict{Symbol, Float32}())
    end
end

function increment_buff_head!(sac::RecurrentSACDiscreteLearner)
    cap = size(sac.buff, 2)
    sac.buff_head = ((sac.buff_head + 1) - 1) % cap + 1
    nothing
end

function push_to_buff!(sac::RecurrentSACDiscreteLearner, is_new_traj, prev_action::Int, prev_reward, cur_state, cur_state_terminal, aspace::MDPs.IntegerSpace)
    buff = sac.buff
    m = sac.buff_head
    n_actions = length(aspace)

    if Bool(buff[1, m])
        delete!(sac.traj_start_points, m)
    end

    buff[1, m] = is_new_traj
    buff[1+1:1+n_actions, m] .= 0f0
    buff[1+prev_action, m] = 1f0
    buff[1+n_actions+1, m] = prev_reward
    buff[1+n_actions+1+1:1+n_actions+1+length(cur_state), m] .= cur_state
    buff[end, m] = cur_state_terminal

    if Bool(is_new_traj)
        push!(sac.traj_start_points, m)
    end

    increment_buff_head!(sac)
    nothing
end

function sample_from_buff!(sac::RecurrentSACDiscreteLearner, env)
    (𝐞, 𝐨, 𝐚, 𝐫, 𝐨′, 𝐝′, 𝐧′), seq_len, batch_size = sac.minibatch, sac.horizon + 1, sac.batch_size
    cap = size(sac.buff, 2)
    @assert seq_len < cap
    n_actions = length(action_space(env))

    
    function isvalidstartpoint(p_begin)
        """Ensures that the buffer head doesn't lie in between any sampled trajectory:"""
        p_end = ((p_begin + seq_len - 1) - 1) % cap + 1
        if p_end > p_begin
            return !(p_begin <= sac.buff_head <= p_end)
        else
            return p_end < sac.buff_head < p_begin
        end
    end

    for n in 1:batch_size
        start_index = rand(sac.traj_start_points)
        while !isvalidstartpoint(start_index)
            start_index = rand(sac.traj_start_points)
        end
        indices = ((start_index:(start_index .+ seq_len - 1)) .- 1) .% cap .+ 1
        𝐞[:, :, n] .= sac.device(sac.buff[:, indices])
    end

    # Note: "actions" are onehot
    prev_actions = @view 𝐞[1+1:1+n_actions, :, :]
    prev_rewards = @view 𝐞[1+n_actions+1, :, :]
    cur_obs = @view 𝐞[1+n_actions+1+1:1+n_actions+1+length(state(env)), :, :]
    
    obs = @view cur_obs[:, 1:end-1, :]
    actions = @view prev_actions[:, 2:end, :]
    rewards = @view prev_rewards[2:end, :]
    next_obs = @view cur_obs[:, 2:end, :]
    next_isterminals = @view 𝐞[end, 2:end, :]
    next_is_newtrajs = @view 𝐞[1, 2:end, :]

    copy!(𝐨, obs); copy!(𝐚, actions); copy!(𝐫, rewards); copy!(𝐨′, next_obs); copy!(𝐝′, next_isterminals); copy!(𝐧′, next_is_newtrajs)

    return 𝐞, 𝐨, 𝐚, 𝐫, 𝐨′, 𝐝′, 𝐧′
end

function preepisode(sac::RecurrentSACDiscreteLearner; env, kwargs...)
    push_to_buff!(sac, true, 1, 0f0, state(env), in_absorbing_state(env), action_space(env))
end

function poststep(sac::RecurrentSACDiscreteLearner{T}; env::AbstractMDP{Vector{T}, Int}, steps::Int, returns, rng::AbstractRNG, kwargs...) where {T}
    @unpack actor, actor_crnn, critics, critic_crnn, γ, α, ρ, batch_size, horizon, tbptt_horizon, 𝐜ᶜ, 𝐜ᵃ, critics′ = sac

    push_to_buff!(sac, false, action(env), reward(env), state(env), in_absorbing_state(env), action_space(env))

    if steps >= sac.min_explore_steps && (steps % 50 == 0)
        @debug "sampling trajectories"
        𝐞, 𝐨, 𝐚, 𝐫, 𝐨′, 𝐝′, 𝐧′  = sample_from_buff!(sac, env)
        # note: 𝐚 is onehot!

        function actor_update()
            θ = Flux.params(actor, actor_crnn)
            Flux.reset!.((actor_crnn, critic_crnn))
            fill!(𝐜ᶜ, 0f0)
            for t in 1:horizon
                𝐜ᶜ[:, t, :] .= @views critic_crnn(𝐞[:, t, :])
            end
            𝐬ᶜ = @views reshape(vcat(𝐜ᶜ[:, 1:horizon, :], 𝐨), :, horizon * batch_size)
            𝐨 = reshape(𝐨, :, horizon * batch_size)
            entropy = 0f0
            ℓθ, ∇θℓ = withgradient(θ) do
                _𝐜ᵃ = reduce(hcat, map(1:horizon) do t
                    @views reshape(actor_crnn(𝐞[:, t, :]), :, 1, batch_size)
                end)
                _𝐜ᵃ = reshape(_𝐜ᵃ, :, horizon * batch_size)
                𝐬ᵃ = vcat(_𝐜ᵃ, 𝐨)
                𝛑, log𝛑 = get_probs_logprobs(actor, 𝐬ᵃ)
                𝐪̂ = min.(map(critic -> critic(𝐬ᶜ), critics)...)
                𝐯̂ = sum(𝛑 .* (𝐪̂ - α * log𝛑); dims=1)
                entropy += -mean(sum(𝛑 .* log𝛑; dims=1))
                ℓθ = -mean(𝐯̂)
                return ℓθ
            end
            Flux.update!(sac.optim_actor, θ, ∇θℓ)
            return ℓθ, entropy 
        end

        function critic_update()
            ϕ = Flux.params(critics..., critic_crnn)
            Flux.reset!.((actor_crnn, critic_crnn))
            fill!(𝐜ᵃ, 0f0)
            fill!(𝐜ᶜ, 0f0)
            for t in 1:(horizon+1)
                𝐜ᵃ[:, t, :] .= @views actor_crnn(𝐞[:, t, :])
                𝐜ᶜ[:, t, :] .= @views critic_crnn(𝐞[:, t, :])
            end
            𝐜ᵃ′ = @view 𝐜ᵃ[:, 2:end, :]
            𝐜ᶜ′ = @view 𝐜ᶜ[:, 2:end, :]
            𝐬ᵃ′ = reshape(vcat(𝐜ᵃ′, 𝐨′), :, horizon * batch_size)
            𝐬ᶜ′ = reshape(vcat(𝐜ᶜ′, 𝐨′), :, horizon * batch_size)

            𝛑′, log𝛑′ = get_probs_logprobs(actor, 𝐬ᵃ′)
            𝐪̂′ = min.(map(critic -> critic(𝐬ᶜ′), critics′)...)
            𝐯̂′ =  sum(𝛑′ .* (𝐪̂′ - α * log𝛑′); dims=1)[1, :]

            𝐨 = reshape(𝐨, :, horizon * batch_size)
            𝐚 = argmax(reshape(𝐚, :, horizon * batch_size), dims=1)[1, :] # CartesianIndices
            𝐫 = reshape(𝐫, horizon * batch_size)
            𝐝′ = reshape(𝐝′, horizon * batch_size)
            𝐧′ = reshape(𝐧′, horizon * batch_size)
            Flux.reset!.((actor_crnn, critic_crnn))
            ℓϕ, ∇ϕℓ = withgradient(ϕ) do
                _𝐜ᶜ = reduce(hcat, map(1:horizon) do t
                    @views reshape(critic_crnn(𝐞[:, t, :]), :, 1, batch_size)
                end)
                _𝐜ᶜ = reshape(_𝐜ᶜ, :, horizon * batch_size)
                𝐬ᶜ = vcat(_𝐜ᶜ, 𝐨)
                𝐪̂¹, 𝐪̂² = critics[1](𝐬ᶜ), critics[2](𝐬ᶜ)
                𝛅¹ = (𝐫 + γ * (1f0 .- 𝐝′) .* 𝐯̂′ - 𝐪̂¹[𝐚]) .* (1f0 .- 𝐧′)
                𝛅² = (𝐫 + γ * (1f0 .- 𝐝′) .* 𝐯̂′ - 𝐪̂²[𝐚]) .* (1f0 .- 𝐧′)
                ℓϕ = 0.5f0 * (mean(𝛅¹.^2) + mean(𝛅².^2))
                return ℓϕ
            end
            Flux.update!(sac.optim_critics, ϕ, ∇ϕℓ)
            return ℓϕ
        end

        function alpha_update(current_ent)
            target_ent::Float32 = 0.98f0 * log(length(action_space(env)))
            α = clamp(exp(log(α) - 0.0003f0 * (current_ent - target_ent)), 0.0001f0, 1000f0)
            sac.α = α
        end

        function target_critics_update()
            ϕ = Flux.params(sac.critics)
            ϕ′ = Flux.params(sac.critics′)
            for (param, param′) in zip(ϕ, ϕ′)
                copy!(param′, ρ * param′ + (1 - ρ) * param)
            end
        end

        function copy_back_actor_params()
            θ = Flux.params(actor, actor_crnn)
            θ′ = Flux.params(sac.π.π, sac.π.crnn)
            for (param, param′) in zip(θ, θ′)
                copy!(param′, param)
            end
        end

        @debug "actor update"
        ℓθ, H = actor_update()

        @debug "critic update"
        ℓϕ = critic_update()
        
        if sac.auto_tune_α
            @debug "temperature update"
            α = alpha_update(H)
        end

        @debug "target critics update"
        target_critics_update()

        @debug "actor parameters back to the actor"
        copy_back_actor_params()


        v̄ = -ℓθ
        sac.stats[:ℓϕ] = ℓϕ
        sac.stats[:ℓθ] = ℓθ
        sac.stats[:v̄] = v̄
        sac.stats[:H] = H
        sac.stats[:α] = α
        episodes = length(returns)
        @debug "learning stats" steps episodes ℓϕ ℓθ v̄ H α
    end
    nothing
end
