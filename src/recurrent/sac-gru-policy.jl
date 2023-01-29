using MDPs
using Flux
using Flux.Zygote
using Random
using UnPack
using DataStructures
using StatsBase

export ContextualSACPolicy

mutable struct ContextualSACPolicy{Tₛ <: AbstractFloat, Tₐ <: AbstractFloat} <: AbstractPolicy{Vector{Tₛ}, Vector{Tₐ}}
    π::SACPolicy{Tₛ, Tₐ}
    crnn::GRUContextRNN
    context::Matrix{Float32}
    isnewtraj::Float32
    prev_a::Vector{Float32}
    prev_r::Float32
    function ContextualSACPolicy(π::SACPolicy{Tₛ, Tₐ}, crnn::GRUContextRNN) where {Tₛ, Tₐ}
        new{Tₛ, Tₐ}(π, crnn, zeros(Float32, size(get_start_state(crnn), 1), 1), 1f0, zeros(Float32, size(π.shift)), 0f0)
    end
end


function (p::ContextualSACPolicy{Tₛ, Tₐ})(rng::AbstractRNG, o::Vector{Tₛ})::Vector{Tₐ} where {Tₛ, Tₐ}
    set_rnn_state!(p.crnn, p.context)
    # assuming the policy will not be called on a terminal state
    evidence = reshape(vcat(p.isnewtraj, p.prev_a, p.prev_r, tof32(o), 0f0), :, 1)
    c = p.crnn(evidence)
    o = reshape(o, :, 1) |> tof32
    s = vcat(c, o)
    a = p.π(rng, s)
    return convert(Vector{Tₐ}, a[:, 1])
end

function (p::ContextualSACPolicy{Tₛ, Tₐ})(o::Vector{Tₛ}, a::Vector{Tₐ})::Float64 where {Tₛ, Tₐ}  # returns log probability density
    set_rnn_state!(p.crnn, p.context)
    evidence = reshape(vcat(p.isnewtraj, p.prev_a, p.prev_r, tof32(o), 0f0), :, 1)
    c = p.crnn(evidence)
    o = reshape(o, :, 1) |> tof32
    s = vcat(c, o)
    a = reshape(a, :, 1)
    return p.π(s, a)[1]
end


function MDPs.preepisode(p::ContextualSACPolicy{Tₛ, Tₐ}; kwargs...) where {Tₛ, Tₐ}
    @debug "Resetting policy"
    p.isnewtraj = 1f0
    fill!(p.prev_a, 0f0)
    p.prev_r = 0
    copy!(p.context, get_start_state(p.crnn))
    nothing
end

function MDPs.poststep(p::ContextualSACPolicy{Tₛ, Tₐ}; env, kwargs...) where {Tₛ, Tₐ}
    p.isnewtraj = 0f0
    copy!(p.prev_a, action(env))
    p.prev_r = reward(env)
    copy!(p.context, get_rnn_state(p.crnn))
    @debug "Storing action, reward, grustate in policy struct"
    nothing
end

