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


end
