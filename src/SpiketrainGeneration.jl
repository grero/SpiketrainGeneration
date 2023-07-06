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




function filter_spikes(spiketrain::SpiketrainClustering.PopulationSpiketrain, t0, t1)
    sp = Vector{Vector{Float64}}(undef, length(spiketrain))
    for ii in 1:length(spiketrain)
        spikes = spiketrain[ii].spikes
        sp[ii] = filter(tt->t0 .< tt .<= t1, spikes)
    end
    SpiketrainClustering.PopulationSpiketrain(sp)
end

function plot_spiketrains!(lg, spiketrains;color=nothing)
    ntrials = length(spiketrains)
    ncells = length(spiketrains[1])
    axes = [Axis(lg[i,1]) for i in 1:ncells]
    linkaxes!(axes...)
    for (ii,ax) in enumerate(axes)
        spikes = [sp[ii].spikes for sp in spiketrains]
        trialid = [fill(jj, length(spikes[jj])) for jj in 1:length(spikes)]
        spikes = cat(spikes..., dims=1)
        trialid = cat(trialid..., dims=1)
        if color === nothing
            scatter!(ax, spikes, trialid, color="black", markersize=7px)
        else
            scatter!(ax, spikes, trialid, color=color[trialid], markersize=7px)
        end
        if ii < ncells
            ax.xticklabelsvisible = false
        end
    end
    lg
end

function get_counts(spiketrains, nbins::Int64;tmax::Union{Nothing,Float64})
    if tmax === nothing 
        tmax = -Inf
        for sp in spiketrains
            for spp in sp
                if !isempty(spp)
                    tmax = max(maximum(spp), tmax)
                end
            end
        end
    end
    bins = range(0.0, stop=tmax, length=nbins)
    counts = fill(0.0, nbins, length(spiketrains[1]), length(spiketrains))
    for (i,sp) in enumerate(spiketrains)
        for (j,spp) in enumerate(sp)
            if !isempty(spp)
                for sppi in spp.spikes
                    ii = searchsortedfirst(bins, sppi)
                    if ii !== nothing
                        counts[ii,j,i] += 1
                    end
                end
           end
        end
    end
    counts
end

end
