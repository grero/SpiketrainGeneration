module SpiketrainGeneration
using JLD2
using Distributions
using LinearAlgebra
using Makie

struct IntegrateAndFireProcess
    θ::Float64
    σ::Float64
end

get_threshold(q::IntegrateAndFireProcess) = q.θ
do_step(q::IntegrateAndFireProcess, λ, dt) = λ*dt + q.σ*randn()

struct LeakyIntegrateAndFireProcess
    θ::Float64
    σ::Float64
    τ::Float64
    x0::Float64
end

LeakyIntegrateAndFireProcess(θ, σ, τ) = LeakyIntegrateAndFireProcess(θ, σ, τ, 0.0)

get_threshold(q::LeakyIntegrateAndFireProcess) = q.θ
do_step(q::LeakyIntegrateAndFireProcess, x, λ, dt) = (-(x-q.x0)/q.τ+λ)*dt + q.σ*randn()

struct ExponentialProcess
end

get_threshold(q::ExponentialProcess) = -log(rand())
do_step(q::ExponentialProcess, λ, dt) = λ*dt


"""
    generate_spikes(spike_process::SpikeProcess, λ::Vector{Float64};tmin=0.0, tmax=Inf, dt=0.1)

Generate a spike train using the supplied `spike_process` and rate vector `λ`
"""
function generate_spikes(spike_process::SpikeProcess, λ::Vector{Float64};tmin=0.0, tmax=Inf, dt=0.1)
    _sp = Float64[]
    q = get_threshold(spike_process)
    x = 0.0
    t = tmin
    for i in axes(λ,1)
        x += do_step(spike_process, λ[i], dt)
        t += dt
        if x >= q
            push!(_sp, t)
            q = get_threshold(spike_process)
            x = 0.0
        end
    end
    _sp
end

function generate_spikes(spike_process, λ::Matrix{Float64};kvs...)
    ntrials = size(λ,2)
    spikes = Vector{Vector{Float64}}(undef, ntrials)
    for i in 1:ntrials
        spikes[i] = generate_spikes(spike_process, λ[:,i];kvs...)
    end
    spikes
end

end