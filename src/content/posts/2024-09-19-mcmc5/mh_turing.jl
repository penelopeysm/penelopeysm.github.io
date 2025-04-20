using AbstractMCMC
using Distributions
using Random
using Statistics: mean, std
using Turing

years = range(1970, 2017)
latlongs = [
    (51.74055, -2.40512),
    (51.75693, -2.3889),
    (51.78825, -1.53601),
    (51.77742, -2.24422),
    (51.77258, -2.67131),
    (51.90814, -1.44842),
    (51.93837, -1.12551),
    (51.84574, -1.79373),
    (51.81055, -2.06563),
    (52.02035, -1.18665),
    (52.07196, -0.99282),
    (51.84266, -1.87047),
    (52.32209, 3.271671),
    (52.22631, 2.1483),
    (52.17919, 0.458765),
    (52.14845, 1.091358),
    (52.29695, 1.681535),
    (52.65615, 3.164477),
    (52.13287, 1.186956),
    (52.19713, 0.052245),
    (52.85175, -1.74943),
    (52.98735, 0.296592),
    (52.68782, 3.358744),
    (52.67473, 4.383683),
    (52.41917, 4.093512),
    (52.51428, 3.582494),
    (52.44482, 3.515402),
    (52.47698, 3.829205),
    (52.46947, 4.746665),
    (52.40592, 4.393268),
    (52.37406, 4.570991),
    (52.43313, 3.32205),
    (52.39657, 3.302281),
    (52.34543, 3.638763),
    (52.45495, 4.447601),
    (52.34317, 3.644572),
    (52.35402, 4.803036),
    (52.4026, 4.896817),
    (52.46815, 4.738608),
    (52.36552, 4.097582),
    (52.11678, 3.730362),
    (52.9878, 6.72993),
    (52.55287, 5.514886),
    (52.58836, 6.272805),
    (52.11357, 4.959362),
    (52.63933, 6.397523),
    (51.95124, 5.30471),
    (52.87718, 7.639933),
]
lats = [ll[1] for ll in latlongs]
longs = [ll[2] for ll in latlongs]
scaled_lats = (lats .- mean(lats)) ./ std(lats)
scaled_longs = (longs .- mean(longs)) ./ std(longs)
scaled_years = (years .- mean(years)) ./ std(years)

@model function demo2(x)
    x ~ Normal(a, 1)
    a ~ Normal(0, 1)
end
chain = sample(demo2(1), MH(), 100)

struct VarInfo
    names::Vector{Symbol}
    is_observed::Dict{Symbol,Bool}
    values::Dict{Symbol,Union{Float64,<:AbstractVector{Float64}}}
    logprobs::Dict{Symbol,Float64}
    distributions::Dict{Symbol,Function}
end

function α_dist(vi::VarInfo)
    return Normal(0, 1)
end
function β_dist(vi::VarInfo)
    return Normal(0, 1)
end
function D_dist(vi::VarInfo)
    # Now we need to use `vi` to extract the values of α, β, and t
    α = vi.values[:α]
    β = vi.values[:β]
    t = vi.values[:t]
    return MvNormal(α .+ β .* t, 1)
end
function setup_varinfo(t, D)
    return VarInfo(
        # names
        [:α, :β, :t, :D],
        # is_observed
        Dict(:α => false, :β => false, :t => true, :D => true),
        # values
        Dict(:α => 0.0, :β => 0.0, :t => t, :D => D),
        # logprobs -- we'll fill these in later
        Dict(),
        # distributions
        Dict(:α => α_dist, :β => β_dist, :D => D_dist),
    )
end
lats_vi = setup_varinfo(scaled_years, scaled_lats)

# For unobserved variables like α and β
function assume!(rng::Random.AbstractRNG, name::Symbol, vi::VarInfo; value = nothing)
    dist = vi.distributions[name](vi)
    if value === nothing
        value = rand(rng, dist)
    end
    logprob = logpdf(dist, value)
    vi.values[name], vi.logprobs[name] = value, logprob
end

# For observed variables like D
function observe!(name::Symbol, vi::VarInfo)
    dist = vi.distributions[name](vi)
    value = vi.values[name]
    logprob = logpdf(dist, value)
    vi.values[name], vi.logprobs[name] = value, logprob
end

function sample!(rng::Random.AbstractRNG, vi::VarInfo; values = nothing)
    for name in vi.names
        # This part deals with α and β
        if !vi.is_observed[name]
            value = values === nothing ? nothing : get(values, name, nothing)
            assume!(rng, name, vi; value = value)
            # This part deals with D. Note the `name in keys(vi.distributions)`
            # check -- this avoids t.
        elseif vi.is_observed[name] && name in keys(vi.distributions)
            observe!(name, vi)
        end
    end
end

function logdensity(vi::VarInfo)
    return sum(values(vi.logprobs))
end


# --------------------

# Sampler setup
struct SimpleMHSampler <: AbstractMCMC.AbstractSampler
    σ::Float64   # standard deviation of the proposal distribution
end

struct Transition
    value::Dict{Symbol,Union{Float64,<:AbstractVector{Float64}}}
    lp::Float64   # log probability density up to additive constant
end

struct SwanModelWithVarInfo <: AbstractMCMC.AbstractModel
    vi::VarInfo
end

swan_model_lats = SwanModelWithVarInfo(setup_varinfo(scaled_years, scaled_lats))

# Method 1: Initial step
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::SimpleMHSampler;
    kwargs...,
)
    sample!(rng, model.vi)
    transition = Transition(model.vi.values, logdensity(model.vi))
    return (transition, model.vi)
end

# Method 2: Subsequent steps
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::SimpleMHSampler,
    vi::VarInfo;
    kwargs...,
)
    # Make a copy of the previous state
    prev_vi = deepcopy(vi)
    # Generate a new proposal value for each of the unobserved variables
    proposal_values = Dict()
    for name in vi.names
        if !vi.is_observed[name]
            proposal_dist = Normal(vi.values[name], sampler.σ)
            proposal_values[name] = rand(rng, proposal_dist)
        end
    end
    # Then sample with these values being fixed. This call mutates the object,
    # which is why we needed to save the previous state above.
    sample!(rng, vi; values = proposal_values)
    # Determine whether to accept
    if log(rand(Float64)) < logdensity(vi) - logdensity(prev_vi)
        successful_transition = Transition(vi.values, logdensity(vi))
        return (successful_transition, vi)
    else
        failed_transition = Transition(prev_vi.values, logdensity(prev_vi))
        return (failed_transition, prev_vi)
    end
end

# Swan models

chain = sample(swan_model_lats, SimpleMHSampler(1.0), 1000)
β_samples = [transition.value[:β] for transition in chain]
mean(β_samples) * std(lats) / std(years)

swan_model_longs = SwanModelWithVarInfo(setup_varinfo(scaled_years, scaled_longs))
chain = sample(swan_model_longs, SimpleMHSampler(1.0), 1000)
mean([transition.value[:β] for transition in chain]) * std(longs) / std(years)


# Coin flip model

function coinflip_vi(y)
    return VarInfo(
        # names
        [:p, :y],
        # is_observed
        Dict(:p => false, :y => true),
        # values
        Dict(:p => 0.5, :y => y),
        # logprobs -- we'll fill these in later
        Dict(),
        # distributions
        Dict(
            :p => (vi) -> Beta(1, 1),
            :y =>
                (vi) -> filldist(
                    Bernoulli(min(max(vi.values[:p], 0), 1)),
                    length(vi.values[:y]),
                ),
        ),
    )
end

@model function coinflip(y)
    p ~ Beta(1, 1)
    y ~ filldist(Bernoulli(p), length(y))
end

data = Float64.(rand(Bernoulli(0.8), 200))
struct CoinFlipModel <: AbstractMCMC.AbstractModel
    vi::VarInfo
end
cf = CoinFlipModel(coinflip_vi(data))
chain = sample(cf, SimpleMHSampler(1.0), 1000)
mean([transition.value[:p] for transition in chain])
